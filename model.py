"""
Model class

"""
import torch._dynamo
torch._dynamo.config.optimize_ddp = False

import warnings
warnings.filterwarnings("ignore")
import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, BCEWithLogitsLoss

import sys
sys.path.append('../')
from typing import Any
import torch
import lightning as L
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ChainedScheduler, LinearLR, LRScheduler, CosineAnnealingLR
from torch.optim import Optimizer
import numpy as np
from loss import MMDLoss

def full_block(in_features, out_features, p_drop=0.1):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.LayerNorm(out_features),
        nn.GELU(),
        nn.Dropout(p=p_drop),
    )

class SkipBlock(nn.Module):
    def __init__(self, in_features):
        """
        Given input X of size in_features
        - out = layernorm(x + MLP(MLP(X))
        
        """
        super().__init__()
        self.dim = in_features
        self.intermediate_dense = nn.Linear(in_features, in_features*2, bias=True)
        self.dense = nn.Linear(in_features*2, in_features, bias=True)
        self.activation = nn.SiLU()
        self.layer_norm = nn.LayerNorm(in_features)
    
    def forward(self, x):
        residual = x
        x = self.intermediate_dense(x) 
        x = self.activation(x)
        x = self.dense(x)
        x = self.layer_norm(x + residual)
        return x
'''
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
'''
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1536):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp \
            (torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class LitUCEModel(L.LightningModule):
    def __init__(self, token_dim: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, output_dim:int, args: Any, dropout: float = 0.0, 
                 warmup_steps: int = 0, gradient_accumulation_steps: int = 1,
                 compiled: bool = False, num_datasets: int = 0, dataset_embedding_dim: int = 16, max_lr=4e-4):
        super().__init__()
        self.save_hyperparameters()
        self.compiled = compiled
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.dropout = dropout
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_lr = max_lr
        self.args = args
        # Encodes Tokens
        self.encoder = nn.Sequential(#SkipBlock(token_dim), # Add an extra layer here with skip connection
                                     nn.Linear(token_dim, d_model, bias=True),
                                     nn.LayerNorm(d_model), # Moved before activation
                                     nn.GELU(), # Revert to GELU
                                    )



        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True, activation="gelu") # switch to gelu activation
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        if compiled:
            self.transformer_encoder = torch.compile(self.transformer_encoder)


        self.d_model = d_model
        self.dropout = dropout
        
        self.decoder = nn.Sequential(full_block(d_model, output_dim, self.dropout),
                                     full_block(output_dim, output_dim, self.dropout),
                                     full_block(output_dim, output_dim, self.dropout),
                                     nn.Linear(output_dim, output_dim)
                                     )

        if compiled:
            self.decoder = torch.compile(self.decoder)

        self.dataset_embedding_dim = dataset_embedding_dim
        #self.dataset_num_embedding = nn.Embedding(num_datasets, self.dataset_embedding_dim, max_norm=True)
        
        self.binary_decoder = nn.Sequential(
            full_block(output_dim + d_model, 2048, self.dropout),
            full_block(2048, 512, self.dropout),
            full_block(512, 128, self.dropout),
            nn.Linear(128, 1)
        )

        if compiled:
            self.binary_decoder = torch.compile(self.binary_decoder)

        # Encodes Tokens for Decoder
        self.gene_embedding_layer = nn.Sequential(
                                     nn.Linear(token_dim, d_model, bias=True),
                                     nn.LayerNorm(d_model), # Moved before activation
                                     nn.GELU(), # Revert to GELU
                                    ) # Don't reuse this layer

        if compiled:
            self.gene_embedding_layer = torch.compile(self.gene_embedding_layer)

        self.pe_embedding = None
        

    def forward(self, src: Tensor, mask: Tensor):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=mask)
        gene_output = self.decoder(output) # batch x seq_len x 128
        # embedding = torch.mul(gene_output, mask.t().unsqueeze(2)).sum(0) # average over non zero genes
        # In the new format, the cls token, which is at the 0 index mark, is the output.
        embedding = gene_output[:, 0, :] # select only the CLS token.
        embedding = nn.functional.normalize(embedding, dim=1) # Normalize.
        return gene_output, embedding


    def predict(self, cell_embedding, gene_embeddings):
        gene_embeddings = self.gene_embedding_layer(gene_embeddings)
        dec = self.binary_decoder \
            (torch.hstack((cell_embedding, gene_embeddings)))
        return dec

    def shared_step(self, batch, batch_idx):
        batch_sentences = batch[0]
        mask = batch[1]
        cell_outputs_X_pe = batch[2]
        cell_outputs_Y = batch[3]

        dataset_nums = batch[5]

        batch_sentences = self.pe_embedding(
            batch_sentences.long())
        cell_outputs_X_pe = self.pe_embedding(
            cell_outputs_X_pe.long())
        #dataset_num_emb = self.dataset_num_embedding(dataset_nums) # batch x emb shap
        
        batch_sentences = nn.functional.normalize(batch_sentences, dim=2) # Normalize token outputs now # TODO YANAY EXPERIMENT WITH REMOVING THIS
        
        _, embedding = self.forward(batch_sentences, mask=mask)
        
        X = cell_outputs_X_pe
        Y = cell_outputs_Y

        X = self.gene_embedding_layer(X)
        embs = embedding.unsqueeze(1).repeat(1, X.shape[1], 1)
        # add dataset num to decoder
        #dataset_num_emb = dataset_num_emb.unsqueeze(1).repeat(1, X.shape[1], 1) # batch x (P+N) x emb
        print("shape of X")
        print(X.shape)
        print("shape of embs")
        print(embs.shape)
        #combine = torch.cat((X, embs, dataset_num_emb), dim=2)
        combine = torch.cat((X, embs), dim=2) # remove dataset value
        if torch.isnan(X).any():
            print(f"[ERROR] NaN detected in 'X' at batch {batch_idx}")

        if torch.isnan(embs).any():
            print(f"[ERROR] NaN detected in 'embs' at batch {batch_idx}")
        
        if torch.isnan(combine).any():
            print(f"[ERROR] NaN detected in 'combine' at batch {batch_idx}")

        decs = self.binary_decoder(combine)

        # if torch.isnan(decs).any():
        #     print(f"[ERROR] NaN detected in 'decs' at batch {batch_idx}")

        if self.args.loss_name == "cross_entropy":
            if torch.isnan(decs).any():
                print(f"[ERROR] NaN detected in 'decs' at batch {batch_idx}")

            bce_loss = BCEWithLogitsLoss()(decs.squeeze(), Y)
            total_loss = bce_loss
        
        elif self.args.loss_name == "only_mmd":
            print("decs")
            print(decs.squeeze().shape)
            print(cell_outputs_Y.shape)
            mmd_loss = MMDLoss(kernel="energy")(decs.squeeze(), cell_outputs_Y)
            total_loss = mmd_loss
        
        elif self.args.loss_name == "bce_mmd":
            # print("combine shape before filtering:", combine.shape)
            # print(combine)
            # Split combine into two halves along the last dimension
            # Split along the second dimension (genes), so first 512 genes are "expressed", last 512 are "unexpressed"
            combine_left, combine_right = torch.split(combine, 512, dim=1)

            # Flatten for MMD
            expressed_embeddings = combine_left.reshape(-1, combine.shape[-1])  # shape [24 * 512, 1024]
            unexpressed_embeddings = combine_right.reshape(-1, combine.shape[-1])  # shape [24 * 512, 1024]

            # print(f"Shape of expressed_embeddings (first half of genes): {expressed_embeddings.shape}")
            # print(f"Shape of unexpressed_embeddings (second half of genes): {unexpressed_embeddings.shape}")

            # Print some example values from both (first 3 genes from each)
            # print("First 3 expressed embeddings:")
            # print(expressed_embeddings[:3])

            # print("First 3 unexpressed embeddings:")
            # print(unexpressed_embeddings[:3])

            mmd_loss = MMDLoss(kernel="energy")(expressed_embeddings, unexpressed_embeddings)

            bce_loss = BCEWithLogitsLoss()(decs.squeeze(), Y)

            total_loss = bce_loss - self.args.mmd_weight * mmd_loss
        
        else:
            raise ValueError(f"Unsupported loss name: {self.args.loss.name}")

        sch = self.lr_schedulers()
        sch.step()
        return total_loss, batch_sentences.shape[1]

    @torch.compile(disable=True)
    def training_step(self, batch, batch_idx):
        loss, seq_len = self.shared_step(batch, batch_idx)
        self.log("train_loss", loss)
        self.log("seq_len", seq_len)
        return loss
        

    def configure_optimizers(self):
        # Marcel Code
        max_lr = self.max_lr
        optimizer = torch.optim.AdamW(self.parameters(), lr=max_lr, weight_decay=0.01)       
        total_steps = self.trainer.estimated_stepping_batches * 2 # not sure why need to do this
        
        linear_warmup = LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=int(0.03 * total_steps))
        cosine_decay = CosineAnnealingLR(optimizer, eta_min=max_lr * 0.3, T_max=total_steps)
        scheduler = ChainedScheduler([
            linear_warmup, cosine_decay
        ])

        lr_scheduler = {
            'scheduler': scheduler,
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1,
        }
        return [optimizer], [lr_scheduler]