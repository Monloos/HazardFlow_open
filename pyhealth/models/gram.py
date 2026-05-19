import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from pyhealth.models.base_model import BaseModel
from pyhealth.datasets import SampleEHRDataset


def build_gram_trees(tree_file_prefix: str, device: torch.device) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """load ancestor tree from .pickled files"""
    leaves_list, ancestors_list = [], []
    for i in range(5, 0, -1):
        tree_path = f"{tree_file_prefix}.level{i}.pk"
        if not os.path.exists(tree_path): continue
        try:
            with open(tree_path, 'rb') as f: tree_map = pickle.load(f)
        except EOFError: tree_map = {}
        if not tree_map:
            print(f"Warning: Tree file {tree_path} is empty. Skipping this level.")
            continue
        ancestors = np.array(list(tree_map.values())).astype('int32')
        if ancestors.ndim < 2:
            print(f"Warning: Data in {tree_path} is malformed (not 2D). Skipping this level.")
            continue
        anc_size = ancestors.shape[1]
        leaves = np.array([[k] * anc_size for k in tree_map.keys()]).astype('int32')
        leaves_list.append(torch.tensor(leaves, dtype=torch.long, device=device))
        ancestors_list.append(torch.tensor(ancestors, dtype=torch.long, device=device))
    if not leaves_list: raise ValueError(f"All tree files with prefix '{tree_file_prefix}' were empty or invalid.")
    return leaves_list, ancestors_list


class RNNLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, rnn_type: str = "GRU", num_layers: int = 1, dropout: float = 0.5, bidirectional: bool = False):
        super(RNNLayer, self).__init__()
        self.dropout_layer = nn.Dropout(dropout)
        rnn_module = getattr(nn, rnn_type)
        self.rnn = rnn_module(input_size, hidden_size, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional, batch_first=True)
        self.bidirectional = bidirectional
        if bidirectional: self.down_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.hidden_size = hidden_size
    def forward(self, x: torch.tensor, mask: Optional[torch.tensor] = None) -> Tuple[torch.tensor, torch.tensor]:
        x = self.dropout_layer(x)
        batch_size = x.size(0)
        lengths = torch.sum(mask.int(), dim=-1).cpu() if mask is not None else torch.full(size=(batch_size,), fill_value=x.size(1), dtype=torch.int64)
        is_valid_len = lengths > 0
        valid_x, valid_lengths = x[is_valid_len], lengths[is_valid_len]
        if valid_x.size(0) == 0:
            outputs, last_outputs = torch.zeros(batch_size, x.size(1), self.hidden_size, device=x.device), torch.zeros(batch_size, self.hidden_size, device=x.device)
            return outputs, last_outputs
        packed_x = rnn_utils.pack_padded_sequence(valid_x, valid_lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.rnn(packed_x)
        valid_outputs, _ = rnn_utils.pad_packed_sequence(packed_outputs, batch_first=True, total_length=x.size(1))
        outputs, last_outputs = torch.zeros(batch_size, x.size(1), self.hidden_size, device=x.device), torch.zeros(batch_size, self.hidden_size, device=x.device)
        outputs[is_valid_len] = valid_outputs
        valid_indices = torch.arange(valid_lengths.size(0))
        valid_last_outputs = self.get_last_outputs(valid_outputs, valid_lengths, valid_indices)
        last_outputs[is_valid_len] = valid_last_outputs
        return outputs, last_outputs
    def get_last_outputs(self, outputs, lengths, indices):
        if not self.bidirectional: return outputs[indices, (lengths - 1), :]
        else:
            outputs = outputs.view(outputs.shape[0], outputs.shape[1], 2, -1)
            f_last, b_last = outputs[indices, (lengths - 1), 0, :], outputs[:, 0, 1, :]
            return self.down_projection(torch.cat([f_last, b_last], dim=-1))

class GRAMLayer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, attention_dim: int):
        super(GRAMLayer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.W_attention = nn.Linear(embedding_dim * 2, attention_dim)
        self.v_attention = nn.Parameter(torch.randn(attention_dim))
        
    def forward(self, leaves_list: List[torch.Tensor], ancestors_list: List[torch.Tensor]) -> torch.Tensor:
        graph_emb_matrix = self.embedding.weight.clone()

        for leaves, ancestors in zip(leaves_list, ancestors_list):
            leaf_indices = leaves[:, 0]
        
            leaf_emb = self.embedding(leaves)
            anc_emb = self.embedding(ancestors)
            
            attention_input = torch.cat([leaf_emb, anc_emb], dim=2)
            mlp_output = torch.tanh(self.W_attention(attention_input))
            pre_attention = torch.matmul(mlp_output, self.v_attention)
            attention_weights = F.softmax(pre_attention, dim=1)

            updated_embs_for_leaves = (anc_emb * attention_weights.unsqueeze(-1)).sum(dim=1)
            
            graph_emb_matrix.index_copy_(0, leaf_indices, updated_embs_for_leaves)

        return graph_emb_matrix

class GRAM(BaseModel):
    def __init__(self, dataset: SampleEHRDataset, feature_key: str, label_key: str, mode: str, tree_file_prefix: str, num_ancestors: int, embedding_dim: int = 128, hidden_dim: int = 128, attention_dim: int = 128, **kwargs):
        super(GRAM, self).__init__(dataset=dataset, feature_keys=[feature_key], label_key=label_key, mode=mode)
        self.embedding_dim, self.hidden_dim, self.attention_dim, self.feature_key = embedding_dim, hidden_dim, attention_dim, feature_key
        self.feat_tokenizers, self.embeddings, self.linear_layers = {}, nn.ModuleDict(), nn.ModuleDict()
        input_info = self.dataset.input_info[self.feature_key]
        self.add_feature_transform_layer(self.feature_key, input_info)
        self.tokenizer = self.feat_tokenizers[self.feature_key]
        num_leaves = self.tokenizer.get_vocabulary_size()
        num_embeddings = num_leaves + num_ancestors
        leaves_list, ancestors_list = build_gram_trees(tree_file_prefix, self.device)
        for i, (l, a) in enumerate(zip(leaves_list, ancestors_list)):
            self.register_buffer(f"leaves_level_{i}", l)
            self.register_buffer(f"ancestors_level_{i}", a)
        self.num_levels = len(leaves_list)
        self.gram_layer = GRAMLayer(num_embeddings, embedding_dim, attention_dim)
        self.rnn = RNNLayer(input_size=embedding_dim, hidden_size=hidden_dim, **kwargs)
        output_size = self.get_output_size(self.get_label_tokenizer())
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        leaves_list, ancestors_list = [getattr(self, f"leaves_level_{i}") for i in range(self.num_levels)], [getattr(self, f"ancestors_level_{i}") for i in range(self.num_levels)]
        
        graph_emb_matrix = self.gram_layer(leaves_list, ancestors_list)
        
        patient_visits = kwargs[self.feature_key]
        tokenized_visits = self.tokenizer.batch_encode_3d(patient_visits)
        tokenized_visits = torch.tensor(tokenized_visits, dtype=torch.long, device=self.device)
        
        embedded_codes = F.embedding(tokenized_visits, graph_emb_matrix)
        
        visit_embeddings = torch.tanh(torch.sum(embedded_codes, dim=2))
        mask = torch.any(tokenized_visits != 0, dim=2)
        _, patient_emb = self.rnn(visit_embeddings, mask)
        logits = self.fc(patient_emb)
        
        y_true = self.prepare_labels(kwargs[self.label_key], self.get_label_tokenizer())
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        return {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
