import os


#os.environ["NCCL_DEBUG"] = "INFO"
os.environ["WANDB_MODE"] = "disabled"
os.environ["OMP_NUM_THREADS"] = "10" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "10" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "10" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "10" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "10"
os.environ["TRITON_CACHE_DIR"] = "/lfs/local/0/yanay/triton/"
os.environ["WANDB_CACHE_DIR"] = "/lfs/local/0/yanay/wandb"
os.environ["WANDB_CACHE_DIR"] = "/lfs/local/0/yanay/wandb"

import utils


import warnings
warnings.filterwarnings("ignore")
import pickle
from tqdm.auto import tqdm
import pandas as pd
from torch import nn, Tensor
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset
import argparse

import sys
sys.path.append('../')
import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from model import TransformerModel
from data import MultiDatasetSentences, MultiDatasetSentenceCollator


def set_up_accelerator(args, warmup_steps):
    hp = {"num_epochs":args.n_epochs,
          "batch_size":args.batch_size,
          "warmup_steps":warmup_steps,
          "dropout":args.dropout,\
          "nhead":args.nhead,
          "n_layers":args.nlayers,
          "d_hid":args.d_hid,
          "emsize":args.emsize,
          "token_dim":args.token_dim,
          "sample_size":args.sample_size,
          "P":args.P,
          "N":args.N,
          "output_dim":args.output_dim,
          "compiled":args.compiled,
          "gradient_accumulation_steps":args.gradient_accumulation_steps
         }
    accelerator = Accelerator(project_dir=f"/lfs/{args.local}/0/yanay/reload_accel_new_dl_chrom_4",
                              log_with="wandb", gradient_accumulation_steps=args.gradient_accumulation_steps)
    accelerator.init_trackers(
        "training_chrom_setup",
        config=hp,
        init_kwargs={
            "wandb": {
                "notes": "training with chromsome sort setup"
            }
        },
    )

    return accelerator

def get_ESM2_embeddings(args):
    # Load in ESM2 embeddings and special tokens
    all_pe = torch.load(f"/dfs/project/uce/all_species_pe_tokens.torch")
    if all_pe.shape[0] == 143574:
        torch.manual_seed(23)
        #MASK_TENSOR = torch.normal(mean=0, std=1, size=(1, args.token_dim))
        CHROM_TENSORS = torch.normal(mean=0, std=1, size=(1895, args.token_dim))
        # 1894 is the total number of chromosome choices, it is hardcoded for now
        all_pe = torch.vstack((all_pe, CHROM_TENSORS)) # Add the chrom tensors to the end
        all_pe.requires_grad = False


    #print("Loaded PE", all_pe.shape)
    # randomize it!
    all_pe = torch.randn_like(all_pe) # random init :) 
    return all_pe

def get_species_names():
    with open("/dfs/project/uce/all_species_offsets.pkl", "rb") as f:
        species_to_offsets = pickle.load(f)

    with open("/dfs/project/uce/species_to_nn_idxs.pkl", "rb") as f:
        species_to_idxs = pickle.load(f)

    sorted_species_names = sorted(species_to_offsets.keys())
    return sorted_species_names

class Scheduler(_LRScheduler):
    # https://kikaben.com/transformers-training-details/
    def __init__(self,
                 optimizer: Optimizer,
                 dim_embed: int,
                 warmup_steps: int,
                 last_epoch: int = -1,
                 verbose: bool = False,
                 gradient_accumulation_steps: int = 4
                ) -> None:
        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> float:
        lr = self.calc_lr(self._step_count, self.dim_embed, self.warmup_steps)
        lr *= np.sqrt(self.gradient_accumulation_steps) # add this
        return [lr] * self.num_param_groups


    def calc_lr(self, step, dim_embed, warmup_steps):
        return dim_embed ** (-0.5) * min(step ** (-0.5),
                                         step * warmup_steps ** (-1.5)) 

def main(args):
    if args.d_hid is None:
        args.d_hid = args.emsize * 4
    if args.N is None:
        args.N = args.P
    if args.pad_length is None:
        args.pad_length = args.sample_size * 2

    datasets_df = pd.read_csv(args.dataset_path)
    sorted_dataset_names = sorted(datasets_df["names"])
    shapes_dict = utils.get_shapes_dict(dataset_path=args.dataset_path)
    TOTAL_N_CELL = 36238464 # 35755720 # 1087158 # 35755720 # 27178832 # 9059574 # 18119211 # 36238464
    EPOCH_LENGTH = int(TOTAL_N_CELL // args.batch_size // 24)
    warmup_steps = EPOCH_LENGTH * 6 # ? not sure why this needs to be included but seems empirical?? no clue why this is 6
    SAVE_EVERY = (EPOCH_LENGTH // 2) + 4  # avoid saving an extra time
    
    accelerator = set_up_accelerator(args, warmup_steps)
    accelerator.print("*****SETUP MODEL*****")
    accelerator.print(f"EPOCH_LENGTH: {EPOCH_LENGTH}")
    accelerator.print(f"warmup_steps: {warmup_steps}")
    accelerator.print(f"SAVE_EVERY: {SAVE_EVERY}")

    model = TransformerModel(token_dim=args.token_dim,
                             d_model=args.emsize,
                             nhead=args.nhead,
                             d_hid=args.d_hid,
                             nlayers=args.nlayers,
                             output_dim=args.output_dim,
                             dropout=args.dropout)

    all_pe = get_ESM2_embeddings(args)
    all_pe.requires_grad= False
    model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
    model = model.train()

    if args.compiled:
        model = torch.compile(model, dynamic=True)

    optimizer = Adam(model.parameters(),
                     betas = (0.9, 0.98),
                     eps = 1.0e-9)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.97)
    scheduler = Scheduler(optimizer, dim_embed=args.emsize,
                          warmup_steps=warmup_steps)

    # Loading Setup for big data
    accelerator.print("******STARTING TRAINING******")
    train(accelerator, model, optimizer, scheduler, sorted_dataset_names,
          shapes_dict, SAVE_EVERY)



    # Redundant/Unused code
    """
    token = -1 * torch.ones(args.token_dim)
    #lr = 1e-4  # learning rate
    epoch = -1
    
    # dataset = MultiDataset({k:torch.tensor(v.layers["counts"]) for k,v in dataset_to_adatas.items()})
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=multi_dataset_collate_fn)
    """

def train(accelerator, model, optimizer, scheduler, sorted_dataset_names,
          shapes_dict, SAVE_EVERY):
    lrs = []
    criterion = nn.BCEWithLogitsLoss()

    dataset = MultiDatasetSentences(sorted_dataset_names, shapes_dict, args)
    multi_dataset_sentence_collator = MultiDatasetSentenceCollator(args)
    
    # Make the dataloader outside of the 
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=multi_dataset_sentence_collator, num_workers=8, persistent_workers=True)
    
    model, optimizer, scheduler, dataloader = accelerator.prepare(model, optimizer,
                                                      scheduler, dataloader)

    # Register the LR scheduler
    accelerator.register_for_checkpointing(scheduler)
    # accelerator.save_state(f"/lfs/{local}/0/yanay/mammal_accel_new_dl_chrom_15_cxg")
    run_step = 0
    # print(f"LR: {optimizer.param_groups[0]['lr']}")
    #accelerator.load_state("/lfs/local/0/yanay/mammal_accel_new_dl_chrom_33_1024")
    #print("loaded state")
    # print(f"LR: {optimizer.param_groups[0]['lr']}")
    # Assume the checkpoint was saved 100 steps into the epoch

    
    
    for epoch in np.arange(1, args.n_epochs + 1):
        # dataset_to_idx_epoch = get_ordering(datasets_to_counts, shuffle=True)
        # for split in np.arange(num_splits):
        # dataset_to_idx = {k:v[split] for k,v in dataset_to_idx_epoch.items()}
        # accelerator.print(f"Epoch {epoch} Split {split} Loading")
        # expression, _ = load_local(list(dataset_to_idx.keys()), dataset_to_idx)
        pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process,
                    smoothing=0.02)
        '''
        if epoch <= 4:
            # skip the first 4
            run_step = 1006624 + 1
            # shuffle the dataloader again?
            continue
            #accelerator.skip_first_batches(dataloader, 128874)
        '''
        #else:
           
            #dataloader = accelerator.skip_first_batches(dataloader, 1006624)
        
        epoch_losses = []
        running_losses = []
        
        #pbar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process,
        #           smoothing=0.05) # Smoother
        epoch_step = 0
        for batch in pbar:
            with accelerator.accumulate(model):
                loss = 0
                
                batch_sentences = batch[0]
                mask = batch[1]
                cell_outputs_X_pe = batch[2]
                cell_outputs_Y = batch[3]

                batch_sentences = model.module.pe_embedding(
                    batch_sentences.long())
                cell_outputs_X_pe = model.module.pe_embedding(
                    cell_outputs_X_pe.long())
                
                batch_sentences = nn.functional.normalize(batch_sentences, dim=2) # Normalize token outputs now
                
                _, embedding = model.forward(batch_sentences, mask=mask)
                
                X = cell_outputs_X_pe
                Y = cell_outputs_Y
                X = model.module.gene_embedding_layer(X)
                embs = embedding.unsqueeze(1).repeat(1, X.shape[1], 1)
                combine = torch.cat((X, embs), dim=2)
                decs = model.module.binary_decoder(combine)
                loss += (criterion(input=decs.squeeze(), target=Y) * batch_sentences.shape[0])

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                curr_loss = float(loss.detach().cpu())
                #epoch_losses.append(curr_loss)
                running_losses.append(curr_loss)
                scheduler.step()
                curr_lr = optimizer.param_groups[0]['lr']
                optimizer.zero_grad()

            
            '''
            loss_string = [f"Epoch {epoch}"]
            loss_string += [
                f"Avg Loss: {round(np.average(epoch_losses) / args.batch_size, 5)}"]
            loss_string += [f"LR: {round(curr_lr, 8)}"]
            loss_string = ", ".join(loss_string)

            pbar.set_description(loss_string)
            ''' # This is a huge slow down
            
            
            if (epoch_step % 50) == 0  and accelerator.is_main_process:
                accelerator.log({"train_loss": curr_loss / args.batch_size,
                                 "learning_rate": curr_lr,
                                 "seq_len": int(batch_sentences.shape[1])},
                                step=run_step)
                
            if epoch_step > 0 and (epoch_step % SAVE_EVERY) == 0 and accelerator.is_main_process:
                accelerator.log({"train_loss": curr_loss / args.batch_size,
                                 "learning_rate": curr_lr,
                                 "seq_len": int(batch_sentences.shape[1])},
                                step=run_step)
                accelerator.save_state(
                    f"/lfs/{args.local}/0/yanay/reload_accel_new_dl_chrom_4")
                accelerator.print(
                    f"Saved checkpoint with"
                    f" RL: {round(np.average(running_losses) / args.batch_size, 5)}")
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(),
                                 f"/dfs/project/uce/deepspeed_models/unwrapped_chrom_model_state_dict_step_{run_step}_epoch_{epoch}_nlayers_{args.nlayers}_sample_size_{args.sample_size}_CLS_CXG.torch")
                running_losses = []
            
            epoch_step += 1
            run_step += 1
        # Save the model again at the end of the epoch
        accelerator.log({"train_loss": curr_loss / args.batch_size,
                         "learning_rate": curr_lr,
                         "seq_len": int(batch_sentences.shape[0])},
                        step=run_step)
        accelerator.save_state(
            f"/lfs/{args.local}/0/yanay/reload_accel_new_dl_chrom_4")
        accelerator.print(
            f"Epoch: {epoch}, Saved checkpoint with loss"
            f" RL: {round(np.average(running_losses) / args.batch_size, 5)}")
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(unwrapped_model.state_dict(),                  f"/dfs/project/uce/deepspeed_models/unwrapped_chrom_model_state_dict_step_{run_step}_epoch_{epoch}_nlayers_{args.nlayers}_sample_size_{args.sample_size}_CLS_CXG.torch")
    accelerator.end_training()

if __name__=="__main__":
    # Parse command-line arguments
    
    parser = argparse.ArgumentParser(description='Universal Cell Embedding')

    # Define command-line arguments
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_gpus', type=int, default=8, help='Number of GPUs')
    parser.add_argument('--n_epochs', type=int, default=16, help='Number of epochs')
    parser.add_argument('--sample_size', type=int, default=1024, help='Sample size')
    parser.add_argument('--token_dim', type=int, default=5120, help='Token dimension')
    parser.add_argument('--emsize', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--d_hid', type=int, default=1024, help='Transformer hidden dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Transformer number of heads')
    
    parser.add_argument('--nlayers', type=int, default=4, help='Number of layers')
    
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--output_dim', type=int, default=256, help='Output dimension')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient Clipping Max Norm')
    
    parser.add_argument('--local', type=str, default='local', help='Path to local')
    parser.add_argument('--compiled', action='store_true', help='Whether the code is compiled')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Number of gradient accumulation steps')

    parser.add_argument('--P', type=int, default=512, help='Value of P')
    parser.add_argument('--N', type=int, default=None, help='Value of N')
    parser.add_argument("--pad_token_idx", type=int, default=0, help="PAD token index")
    parser.add_argument("--chrom_token_left_idx", type=int, default=1, help="Chrom token left index")
    parser.add_argument("--chrom_token_right_idx", type=int, default=2, help="Chrom token right index")
    parser.add_argument("--cls_token_idx", type=int, default=3, help="CLS token index")
    parser.add_argument("--CHROM_TOKEN_OFFSET", type=int, default=143574, help="Offset index, tokens after this mark are chromosome identifiers")  
    parser.add_argument("--pad_length", type=int, default=None, help="PAD length")
    parser.add_argument('--dataset_path', type=str,
                    default="/dfs/project/uce/response/full_train_datasets.csv",
                    help='Path to the dataset')
    args = parser.parse_args()
    torch.set_float32_matmul_precision("medium")
    main(args)
