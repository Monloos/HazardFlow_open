import os
import argparse
from copy import deepcopy
import pickle
from pathlib import Path
from argparse import Namespace

import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from models import DrFuseTrainer
from utils import EHRDiscretizer, EHRNormalizer, get_ehr_datasets, load_cxr_ehr, load_discretized_header

def main(args):
    # ===================================================================
    #  Step 1: SETUP (Data, Seed, etc.)
    # ===================================================================
    
    if args.task == 'phe':
        args.data_dir = 'your path/mortality_prediction/drfuse-main/dataset/mimic4extract/data/phenotyping'

    torch.set_num_threads(5)
    L.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_root = Path(args.data_dir)
    ehr_dir = data_root
    cxr_dir = Path(args.cxr_resized_data_dir) 

    # --- Data Discretizer and Normalizer Setup ---
    discretizer = EHRDiscretizer(timestep=1.0,
                                 store_masks=True,
                                 impute_strategy='previous',
                                 start_time='zero')

    discretizer_header = load_discretized_header(discretizer=discretizer, ehr_dir=ehr_dir) 
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if "->" not in x]

    normalizer = EHRNormalizer(fields=cont_channels)
    normalizer_state = args.normalizer_state
    if normalizer_state is None:
        script_dir = os.path.dirname(__file__)
        normalizer_state = os.path.join(script_dir, 'normalizers', f'ph_ts{args.timestep}.input_str_previous.start_time_zero.normalizer')
    
    if os.path.exists(normalizer_state):
        normalizer.load_params(normalizer_state)
    else:
        print(f"Warning: Normalizer state file not found at {normalizer_state}. Proceeding without normalization.")

    # --- Data Loading ---
    if args.task == 'mor':
        print("In-hospital Mortality (mor)")
        ehr_datasets = get_ehr_datasets(discretizer, normalizer,
                                        ehr_data_dir=ehr_dir,
                                        ehr_pkl_fpath_train=ehr_dir/'ehr_in-hospital-mortality_48h_train.pkl',
                                        ehr_pkl_fpath_val=ehr_dir/'ehr_in-hospital-mortality_48h_val.pkl',
                                        ehr_pkl_fpath_test=ehr_dir/'ehr_in-hospital-mortality_48h_test.pkl')
        meta_pkl_path = data_root/'metas_with_labels_in-hospital-mortality_first_48h.pkl'
        
    else: # task == 'phe'
        print("Phenotyping (phe)")
        ehr_datasets = get_ehr_datasets(discretizer, normalizer,
                                        ehr_data_dir=ehr_dir,
                                        ehr_pkl_fpath_train=ehr_dir/'ehr_phenotyping_48h_train.pkl',
                                        ehr_pkl_fpath_val=ehr_dir/'ehr_phenotyping_48h_val.pkl',
                                        ehr_pkl_fpath_test=ehr_dir/'ehr_phenotyping_48h_test.pkl')
        meta_pkl_path = data_root/'metas_with_labels_phenotyping_first_48h.pkl'

    dataloaders = load_cxr_ehr(cxr_resized_data_dir=args.cxr_resized_data_dir,
                              data_pairs_train=args.data_pair,
                              data_ratio=args.data_ratio,
                              batch_size=args.batch_size,
                              meta_pkl_fpath=meta_pkl_path,
                              ehr_datasets=ehr_datasets,
                              cxr_pkl_fpath_train=ehr_dir/f'cxr_{args.task}_48h_train.pkl',
                              cxr_pkl_fpath_val=ehr_dir/f'cxr_{args.task}_48h_val.pkl',
                              cxr_pkl_fpath_test=ehr_dir/f'cxr_{args.task}_48h_test.pkl',
                              num_workers=args.num_workers)

    train_dl, val_dl, test_dl_partial, test_dl_paired = dataloaders
    
    label_names = train_dl.dataset.CLASSES
    if args.task == 'mor':
        label_names = ['in-hospital-mortality'] 

    # ===================================================================
    #  Step 2: PHASE 1
    # ===================================================================


    model_phase1 = DrFuseTrainer(args=args, label_names=label_names, training_phase=1)

    early_stop_callback = EarlyStopping(monitor='val_PRAUC_avg_over_dxs/final', min_delta=0.00, patience=args.patience, verbose=True, mode="max")
    checkpoint_callback_phase1 = ModelCheckpoint(
        dirpath='./lightning_logs/phase1_checkpoints',
        filename='best_model-{epoch:02d}-{val_PRAUC_avg_over_dxs/final:.4f}',
        save_top_k=1,
        monitor='val_PRAUC_avg_over_dxs/final',
        mode='max',
    )
    
    trainer_phase1 = L.Trainer(
        devices=[args.cuda],
        accelerator='gpu',
        max_epochs=50, 
        callbacks=[early_stop_callback, checkpoint_callback_phase1],
        default_root_dir='./lightning_logs/phase1'
    )

    trainer_phase1.fit(model=model_phase1, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best_model_path_phase1 = checkpoint_callback_phase1.best_model_path
    print(f"\nPhase 1 finished. Best model saved at: {best_model_path_phase1}\n")
    
    if not best_model_path_phase1:
        print("Error: Could not find the best model from Phase 1. Exiting.")
        return

    # ===================================================================
    #  Step 3: PHASE 2
    # ===================================================================
    print("\n" + "="*50)
    print("                STARTING TRAINING PHASE 2")
    print("="*50 + "\n")

    model_phase2 = DrFuseTrainer.load_from_checkpoint(
        checkpoint_path=best_model_path_phase1,
        args=args, 
        label_names=label_names,
        training_phase=2, 
    )

    checkpoint_callback_phase2 = ModelCheckpoint(
        dirpath='./lightning_logs/phase2_checkpoints',
        filename='best_model-{epoch:02d}-{val_PRAUC_avg_over_dxs/final:.4f}',
        save_top_k=1,
        monitor='val_PRAUC_avg_over_dxs/final',
        mode='max',
    )

    trainer_phase2 = L.Trainer(
        devices=[args.cuda],
        accelerator='gpu',
        max_epochs=20, 
        callbacks=[early_stop_callback, checkpoint_callback_phase2], 
        default_root_dir='./lightning_logs/phase2'
    )

    trainer_phase2.fit(model=model_phase2, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best_model_path_phase2 = checkpoint_callback_phase2.best_model_path
    print(f"\nPhase 2 finished. Best model saved at: {best_model_path_phase2}\n")

    if not best_model_path_phase2:
        print("Error: Could not find the best model from Phase 2. Using last model for testing.")
        best_model_path_phase2 = trainer_phase2.checkpoint_callback.last_model_path


    # ===================================================================
    #  Step 4: test
    # ===================================================================
    print("\n" + "="*50)
    print("                STARTING FINAL TESTING")
    print("="*50 + "\n")
    
    results = {}
    trainer_phase2.test(model=model_phase2, dataloaders=test_dl_partial, ckpt_path=best_model_path_phase2) 
    results['partial_test_results'] = deepcopy(model_phase2.test_results)

    trainer_phase2.test(model=model_phase2, dataloaders=test_dl_paired, ckpt_path=best_model_path_phase2)
    results['paired_test_results'] = deepcopy(model_phase2.test_results)
    
    logpath = trainer_phase2.logger.log_dir
    with open(os.path.join(logpath, 'test_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print("\n--- Final Test Results ---")
    print("Partial Test Set:")
    print(f"  AUROC: {results['partial_test_results']['auroc']:.4f}")
    print(f"  PRAUC: {results['partial_test_results']['prauc']:.4f}")
    print("\nPaired Test Set:")
    print(f"  AUROC: {results['paired_test_results']['auroc']:.4f}")
    print(f"  PRAUC: {results['paired_test_results']['prauc']:.4f}")
    
    mean_auroc = (results['partial_test_results']['auroc'] + results['paired_test_results']['auroc']) / 2
    mean_prauc = (results['partial_test_results']['prauc'] + results['paired_test_results']['prauc']) / 2
    
    print("\nAverage:")
    print(f"  Mean AUROC: {mean_auroc:.4f}")
    print(f"  Mean PRAUC: {mean_prauc:.4f}")
    print(f"\nResults saved in {logpath}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DrFuse Model Training')
    parser.add_argument('--data_dir', type=str, default='/your path/mortality_prediction/drfuse-main/dataset/mimic4extract/data/in-hospital-mortality')
    parser.add_argument('--cxr_resized_data_dir', type=str, default='/your path/mimic-cxr-jpg/2.0.0/resized')
    parser.add_argument('--data_ratio', type=float, default=1.0)
    parser.add_argument('--data_pair', type=str, default='paired', choices=['partial', 'paired'])
    parser.add_argument('--timestep', type=float, default=1.0)
    parser.add_argument('--lambda_disentangle_shared', type=float, default=1)
    parser.add_argument('--lambda_disentangle_ehr', type=float, default=1)
    parser.add_argument('--lambda_disentangle_cxr', type=float, default=1)
    parser.add_argument('--lambda_pred_ehr', type=float, default=1)
    parser.add_argument('--lambda_pred_cxr', type=float, default=1)
    parser.add_argument('--lambda_pred_shared', type=float, default=1)
    parser.add_argument('--aug_missing_ratio', type=float, default=0.3)
    parser.add_argument('--lambda_attn_aux', type=float, default=1)
    parser.add_argument('--ehr_n_layers', type=int, default=1)
    parser.add_argument('--ehr_n_head', type=int, default=4)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--normalizer_state', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task', type=str, default='mor')
    parser.add_argument('--cuda', type=int, default=0)
    
    args = parser.parse_args()
    main(args)