import os


#os.environ["NCCL_DEBUG"] = "INFO"
#os.environ["WANDB_MODE"] = "disabled"
os.environ["OMP_NUM_THREADS"] = "10" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "10" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "10" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "10" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "10"
os.environ["TRITON_CACHE_DIR"] = "/home/alishbaimran/triton/"
os.environ["WANDB_CACHE_DIR"] = "/home/alishbaimran/wandb/"

import warnings
import utils
#warnings.filterwarnings("ignore")
import pickle
from tqdm.auto import tqdm
import pandas as pd
from torch import nn, Tensor

from torch.utils.data import DataLoader, TensorDataset
import argparse

import sys
sys.path.append('../')
import torch
import numpy as np

from model import LitUCEModel
from data import MultiDatasetSentences, MultiDatasetSentenceCollator

# Lit Imports
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy, DDPStrategy
from lightning.pytorch.profilers import PyTorchProfiler, AdvancedProfiler
from callbacks import LogLR

def get_ESM2_embeddings(args):
    # Load in ESM2 embeddings and special tokens
    all_pe = torch.load(args.token_location)
    if all_pe.shape[0] == 143574:
        torch.manual_seed(23)
        #MASK_TENSOR = torch.normal(mean=0, std=1, size=(1, args.token_dim))
        CHROM_TENSORS = torch.normal(mean=0, std=1, size=(1895, args.token_dim))
        # 1894 is the total number of chromosome choices, it is hardcoded for now
        all_pe = torch.vstack((all_pe, CHROM_TENSORS)) # Add the chrom tensors to the end
        all_pe.requires_grad = False


    print("Loaded PE", all_pe.shape)
    # do not randomize it!
    # all_pe = torch.randn_like(all_pe) # random init :) 
    return all_pe

def main(args):
    # Setup Data
    datasets_df = pd.read_csv(args.dataset_path)
    sorted_dataset_names = sorted(datasets_df["names"])
    shapes_dict = utils.get_shapes_dict(dataset_path=args.dataset_path)
    TOTAL_N_CELL = 36238464 # 35755720 # 1087158 # 35755720 # 27178832 # 9059574 # 18119211 # 36238464
    EPOCH_LENGTH = int(TOTAL_N_CELL // args.batch_size // 24)
    warmup_steps = EPOCH_LENGTH * 6 # ? not sure why this needs to be included but seems empirical?? no clue why this is 6
    SAVE_EVERY = (EPOCH_LENGTH // 2) + 4  # avoid saving an extra time
    dataset = MultiDatasetSentences(sorted_dataset_names, shapes_dict, args)
    multi_dataset_sentence_collator = MultiDatasetSentenceCollator(args)
    
    # Make the dataloader outside of the 
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=multi_dataset_sentence_collator, num_workers=8, persistent_workers=True)
    
    # Setup Model
    model = LitUCEModel(token_dim=args.token_dim,
                         d_model=args.emsize,
                         nhead=args.nhead,
                         d_hid=args.d_hid,
                         nlayers=args.nlayers,
                         output_dim=args.output_dim,
                         dropout=args.dropout,
                         warmup_steps=warmup_steps, 
                         gradient_accumulation_steps=args.gradient_accumulation_steps,
                         compiled=False,
                         num_datasets=len(sorted_dataset_names),
                         max_lr=args.max_lr,
                         dataset_embedding_dim=args.dataset_embedding_dim
                       )

    all_pe = get_ESM2_embeddings(args)
    all_pe.requires_grad = args.fine_tune_embeds
    model.pe_embedding = nn.Embedding.from_pretrained(all_pe)

    print(model.pe_embedding.weight)
    
    model = model.train()

    if args.compiled:
        model = torch.compile(model, dynamic=False)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print(f"Run Name: {args.run_name}")
    print(f"Checkpoint Directory: {args.checkpoint_dir}")
    # Init ModelCheckpoint callback, monitoring 'val_loss'

    model_run_name = f"UCE_plain"
    print(model_run_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f"{args.run_name}" + "-{epoch}-{step}.pt",
        every_n_train_steps=50000,
        save_top_k=-1
    )

    wandb_logger = WandbLogger(project="VCI", name=args.run_name)
    #wandb_logger.watch(model, log_freq=5000)
    trainer = L.Trainer(max_epochs=args.n_epochs, 
                        
                        callbacks=[checkpoint_callback,  
                                   LogLR(100),
                                   RichProgressBar()],
                        #devices=1,
                        # Accumulation
                        gradient_clip_val=args.max_grad_norm,
                        accumulate_grad_batches=args.gradient_accumulation_steps,
                        precision="bf16-mixed",
                        strategy=DDPStrategy(process_group_backend="nccl"),
                        # Logging
                        logger=wandb_logger,
                        num_nodes=args.num_nodes,
                        #profiler=PyTorchProfiler(export_to_chrome=True),
                       )
    trainer.fit(model=model, train_dataloaders=dataloader)
    trainer.save_checkpoint(f"{args.checkpoint_dir}/{args.run_name}_final.pt")

if __name__=="__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Lightning Train Universal Cell Embedding')

    # Define command-line arguments
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--num_gpus', type=int, default=8, help='Number of GPUs')
    parser.add_argument('--n_epochs', type=int, default=8, help='Number of epochs')
    parser.add_argument('--sample_size', type=int, default=1024, help='Sample size')
    parser.add_argument('--token_dim', type=int, default=5120, help='Token dimension')
    parser.add_argument('--token_location', type=str, default="/scratch/ctc/ML/uce/all_species_pe_tokens.torch", help='Protein Embedding stacked token file')
    parser.add_argument('--emb_model_name', type=str, default="ESM2", help='PE model name')
    
    parser.add_argument('--emsize', type=int, default=1280, help='Embedding dimension')
    parser.add_argument('--d_hid', type=int, default=5120, help='Transformer hidden dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Transformer number of heads')
    
    parser.add_argument('--nlayers', type=int, default=2, help='Number of layers')
    
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--mask_prop', type=float, default=0.2, help='Mask probability for training')
    
    parser.add_argument('--output_dim', type=int, default=1280, help='Output dimension')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient Clipping Max Norm')
    parser.add_argument('--max_lr', type=float, default=3e-4, help='Max Learning rate')
    
    
    parser.add_argument('--local', type=str, default='local', help='Path to local')
    parser.add_argument('--compiled', action='store_true', help='Whether the code is compiled')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps')
    
    parser.add_argument('--P', type=int, default=512, help='Value of P')
    parser.add_argument('--N', type=int, default=512, help='Value of N')
    parser.add_argument("--pad_token_idx", type=int, default=0, help="PAD token index")
    parser.add_argument("--chrom_token_left_idx", type=int, default=1, help="Chrom token left index")
    parser.add_argument("--chrom_token_right_idx", type=int, default=2, help="Chrom token right index")
    parser.add_argument("--cls_token_idx", type=int, default=3, help="CLS token index")
    parser.add_argument("--CHROM_TOKEN_OFFSET", type=int, default=143574, help="Offset index, tokens after this mark are chromosome identifiers")  
    parser.add_argument("--pad_length", type=int, default=2048, help="PAD length")
    parser.add_argument('--dataset_path', type=str,
                    default="/scratch/ctc/ML/uce/full_train_datasets.csv",
                    help='Path to the dataset')

    parser.add_argument('--fine_tune_embeds', type=bool, default=False, help='Allow embeddings to be finetuned')
    
    parser.add_argument('--dataset_embedding_dim', type=int, default=0, help='Dataset Embedding dimension in binary decoder')
    # Multi Node Setup
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of training nodes')
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--run_name', type=str, required=True)
    args = parser.parse_args()
    torch.set_float32_matmul_precision("medium")
    main(args)
