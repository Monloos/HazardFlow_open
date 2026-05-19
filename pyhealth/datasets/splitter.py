from itertools import chain
from typing import Optional, Tuple, Union, List

import numpy as np
import torch

from pyhealth.datasets import SampleBaseDataset


# TODO: train_dataset.dataset still access the whole dataset which may leak information
# TODO: add more splitting methods


def split_by_visit(
    dataset: SampleBaseDataset,
    ratios: Union[Tuple[float, float, float], List[float]],
    seed: Optional[int] = None,
):
    """Splits the dataset by visit (i.e., samples).

    Args:
        dataset: a `SampleBaseDataset` object
        ratios: a list/tuple of ratios for train / val / test
        seed: random seed for shuffling the dataset

    Returns:
        train_dataset, val_dataset, test_dataset: three subsets of the dataset of
            type `torch.utils.data.Subset`.

    Note:
        The original dataset can be accessed by `train_dataset.dataset`,
            `val_dataset.dataset`, and `test_dataset.dataset`.
    """
    if seed is not None:
        np.random.seed(seed)
    assert sum(ratios) == 1.0, "ratios must sum to 1.0"
    index = np.arange(len(dataset))
    np.random.shuffle(index)
    train_index = index[: int(len(dataset) * ratios[0])]
    val_index = index[
        int(len(dataset) * ratios[0]) : int(len(dataset) * (ratios[0] + ratios[1]))
    ]
    val_len = int(len(dataset)*ratios[1])     
    test_len = int(len(dataset)-len(dataset)*(ratios[0]+ratios[1]))      
    test_index = index[int(len(dataset) * (ratios[0] + ratios[1])) :]
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)
    return train_dataset, val_dataset, test_dataset, val_len, test_len



from sklearn.model_selection import StratifiedKFold
from typing import List, Tuple, Union, Optional
from itertools import chain
import numpy as np
import torch

def stratified_k_split_by_patient(
    dataset,
    ratios: Union[Tuple[float, float, float], List[float]],
    seed: Optional[int] = None,
    n_splits: int = 5,
    fold_index: int = 0,
):
    """
    Stratified split the dataset by patient ID using StratifiedKFold.
    
    Args:
        dataset: A `SampleBaseDataset` object with `patient_to_index` and `patient_labels` attributes.
        ratios: A tuple/list for train/val/test ratios (must sum to 1.0).
        seed: Random seed.
        n_splits: Number of stratified folds.
        fold_index: Which fold to use as the test set.

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    assert sum(ratios) == 1.0, "Ratios must sum to 1.0"
    if seed is not None:
        np.random.seed(seed)

    patient_ids = np.array(list(dataset.patient_to_index.keys()))

    patient_to_label = {}
    for item in dataset:
        pid = item["patient_id"]
        label = item["label"]
        patient_to_label[pid] = label
    # print("dataset", dataset[0])
    # patient_labels = np.array([dataset.label[pid] for pid in patient_ids])  # 0 or 1
    patient_labels = np.array([patient_to_label[pid] for pid in patient_ids])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(patient_ids, patient_labels))
    train_val_idx, test_idx = splits[fold_index]
    
    train_val_patients = patient_ids[train_val_idx]
    test_patients = patient_ids[test_idx]

    num_train_val = len(train_val_patients)
    train_end = int(num_train_val * (ratios[0] / (ratios[0] + ratios[1])))
    shuffled = train_val_patients.copy()
    np.random.shuffle(shuffled)
    train_patients = shuffled[:train_end]
    val_patients = shuffled[train_end:]

    def get_indices(patient_list):
        return list(chain(*[dataset.patient_to_index[pid] for pid in patient_list]))

    train_index = get_indices(train_patients)
    val_index = get_indices(val_patients)
    test_index = get_indices(test_patients)

    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)

    return train_dataset, val_dataset, test_dataset




def split_by_patient(
    dataset: SampleBaseDataset,
    ratios: Union[Tuple[float, float, float], List[float]],
    seed: Optional[int] = None,
):
    """Splits the dataset by patient.

    Args:
        dataset: a `SampleBaseDataset` object
        ratios: a list/tuple of ratios for train / val / test
        seed: random seed for shuffling the dataset

    Returns:
        train_dataset, val_dataset, test_dataset: three subsets of the dataset of
            type `torch.utils.data.Subset`.

    Note:
        The original dataset can be accessed by `train_dataset.dataset`,
            `val_dataset.dataset`, and `test_dataset.dataset`.
    """
    if seed is not None:
        np.random.seed(seed)
    assert sum(ratios) == 1.0, "ratios must sum to 1.0"
    patient_indx = list(dataset.patient_to_index.keys())
    num_patients = len(patient_indx)
    np.random.shuffle(patient_indx)
    train_patient_indx = patient_indx[: int(num_patients * ratios[0])]
    val_patient_indx = patient_indx[
        int(num_patients * ratios[0]) : int(num_patients * (ratios[0] + ratios[1]))
    ]
    test_patient_indx = patient_indx[int(num_patients * (ratios[0] + ratios[1])) :]
    train_index = list(
        chain(*[dataset.patient_to_index[i] for i in train_patient_indx])
    )
    val_index = list(chain(*[dataset.patient_to_index[i] for i in val_patient_indx]))
    test_index = list(chain(*[dataset.patient_to_index[i] for i in test_patient_indx]))
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)
    return train_dataset, val_dataset, test_dataset


def split_by_sample(
    dataset: SampleBaseDataset,
    ratios: Union[Tuple[float, float, float], List[float]],
    seed: Optional[int] = None,
    get_index: Optional[bool] = False,
):
    """Splits the dataset by sample

    Args:
        dataset: a `SampleBaseDataset` object
        ratios: a list/tuple of ratios for train / val / test
        seed: random seed for shuffling the dataset

    Returns:
        train_dataset, val_dataset, test_dataset: three subsets of the dataset of
            type `torch.utils.data.Subset`.

    Note:
        The original dataset can be accessed by `train_dataset.dataset`,
            `val_dataset.dataset`, and `test_dataset.dataset`.
    """
    if seed is not None:
        np.random.seed(seed)
    assert sum(ratios) == 1.0, "ratios must sum to 1.0"
    index = np.arange(len(dataset))
    np.random.shuffle(index)
    train_index = index[: int(len(dataset) * ratios[0])]
    val_index = index[
                int(len(dataset) * ratios[0]): int(
                    len(dataset) * (ratios[0] + ratios[1]))
                ]
    test_index = index[int(len(dataset) * (ratios[0] + ratios[1])):]
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)
    
    if get_index:
        return torch.tensor(train_index), torch.tensor(val_index), torch.tensor(test_index)
    else:
        return train_dataset, val_dataset, test_dataset
