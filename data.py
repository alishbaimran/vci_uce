"""
Dataloaders

"""

import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../')
from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np
import pickle
import torch.utils.data as data
import torch.nn.functional as F


class MultiDatasetSentences(data.Dataset):
    def __init__(self, sorted_dataset_names, shapes_dict, args) -> None:
        super(MultiDatasetSentences, self).__init__()
        # self.xs = {}
        self.num_cells = {}
        self.num_genes = {}
        self.shapes_dict = shapes_dict
        self.args = args

        self.total_num_cells = 0
        for name in sorted_dataset_names:
            num_cells, num_genes = self.shapes_dict[name]
            # self.xs[name] = X
            self.num_cells[name] = num_cells
            self.num_genes[name] = num_genes

            self.total_num_cells += num_cells

        self.datasets = sorted_dataset_names

        # TODO: preferably not hard-coded here
        self.dataset_to_protein_embeddings = torch.load(
            f"/scratch/ctc/ML/uce/reduced_datasets_to_pe_chrom_{args.token_dim}_new.torch")
        
        for k, v in self.dataset_to_protein_embeddings.items():
            if torch.isnan(v).any():
                print(f"[ERROR] NaN detected in dataset_to_protein_embeddings for {k}")
                raise ValueError(f"NaNs detected in pre-loaded embeddings for {k}")

        with open("/scratch/ctc/ML/uce/dataset_to_chroms_new.pkl", "rb") as f:
            self.dataset_to_chroms = pickle.load(f)
        with open("/scratch/ctc/ML/uce/dataset_to_starts_new.pkl", "rb") as f:
            self.dataset_to_starts = pickle.load(f)

        self.datasets_to_num = {k:v for k,v in zip(self.datasets, range(len(self.datasets)))}

    def __getitem__(self, idx):
        if isinstance(idx, int):
            for dataset in sorted(self.datasets):
                if idx < self.num_cells[dataset]:
                    cts = np.memmap(f"/large_experiments/ctc/ML/data/cell/observational/" + f"{dataset}_counts.npz",
                            dtype='int64', mode='r', shape=self.shapes_dict[dataset])
                    counts = cts[idx]
                    counts = torch.tensor(counts).unsqueeze(0)
                    weights = torch.log1p(counts)
                    weights = (weights / torch.sum(weights))
                    #cell_outputs_X_pe is getting task_sentence and cell_outputs_Y gets the task_counts
                    batch_sentences, mask, cell_outputs_X_pe, cell_outputs_Y = sample_cell_sentences(
                        counts, dataset, self.args, self.dataset_to_protein_embeddings
                    )
                    seq_len = self.args.pad_length
                    dataset_num = self.datasets_to_num[dataset]
                    return batch_sentences, mask, cell_outputs_X_pe, cell_outputs_Y, idx, seq_len, dataset_num
                else:
                    idx -= self.num_cells[dataset]
            raise IndexError
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return self.total_num_cells

    def get_dim(self) -> Dict[str, int]:
        return self.num_genes


class MultiDatasetSentenceCollator(object):
    def __init__(self, args):
        self.pad_length = args.pad_length
        self.P = args.P
        self.N = args.N


    def __call__(self, batch):
        batch_size = len(batch)
        batch_sentences = torch.zeros((batch_size, self.pad_length))
        mask = torch.zeros((batch_size, self.pad_length), dtype=bool)


        idxs = torch.zeros(batch_size)
        Xs = torch.zeros((batch_size, (self.P + self.N)))
        Ys = torch.zeros((batch_size, (self.P + self.N)))

        dataset_nums = torch.zeros(batch_size)

        i = 0
        max_len = 0
        for bs, msk, xx, yy, idx, seq_len, dataset_num in batch:
            batch_sentences[i, :] = bs

            max_len = max(max_len, seq_len)
            mask[i, :] = msk
            idxs[i] = idx

            xx = xx#.squeeze()
            yy = yy.squeeze()

            Xs[i] = xx#[pn_idx]
            Ys[i] = yy#[pn_idx]
            dataset_nums[i] = dataset_num
            i += 1
        
        return batch_sentences[:, :max_len] , mask[:, :max_len], Xs, Ys, idxs, dataset_nums.long()


def sample_cell_sentences(counts, dataset, args, dataset_to_protein_embeddings):
    
    if torch.isnan(counts).any():
        raise ValueError(f"NaN values in counts for dataset {dataset}")

    if torch.any(counts < 0):
        print("less than 0 count found")
        counts = F.relu(counts)

    max_val = torch.max(counts).item()
    min_val = torch.min(counts).item()

    if max_val > 20:
        counts = torch.log1p(counts)
    # else:
    #     print(f"Counts are already log-transformed (max={max_val:.2f}, min={min_val:.2f})")


    expression_weights = counts / torch.sum(counts)

    ds_emb_idxs = dataset_to_protein_embeddings[dataset]
    cell_sentences = torch.zeros((counts.shape[0], args.pad_length))
    task_counts = torch.zeros((counts.shape[0], args.P + args.N))
    task_sentence = torch.zeros((counts.shape[0], args.P + args.N))
    mask = torch.zeros((counts.shape[0], args.pad_length), dtype=torch.bool)

    for c, cell in enumerate(counts):
        num_pos_genes = torch.sum(cell > 0)
        start_sentence = min((args.pad_length - 1) // 2, num_pos_genes)

        genes_ranked_exp = torch.argsort(cell, descending=True)

        cell_sentences[c, 0] = args.cls_token_idx
        cell_sentences[c, 1: start_sentence + 1] = genes_ranked_exp[:start_sentence]

        cell_sentences[c, start_sentence + 1:] = torch.multinomial(
            expression_weights, args.pad_length - start_sentence - 1, replacement=True
        )

        cell_sentences[c, :] = ds_emb_idxs[cell_sentences[c, :].to(torch.int32)]

        exp_genes = torch.where(cell > 0)[0]
        if len(exp_genes) > args.P:
            task_sentence[c, :args.P] = exp_genes[torch.randperm(len(exp_genes))[:args.P]]
        else:
            task_sentence[c, :args.P] = exp_genes[torch.randint(len(exp_genes), (args.P,))]

        unexp_genes = torch.where(cell == 0)[0]
        if len(unexp_genes) == 0:
            print("using fallback for unexpressed genes")
            unexp_genes = torch.where(cell < 1)[0]

        if len(unexp_genes) > args.N:
            task_sentence[c, args.P:] = unexp_genes[torch.randperm(len(unexp_genes))[:args.N]]
        else:
            task_sentence[c, args.P:] = unexp_genes[torch.randint(len(unexp_genes), (args.N,))]

        task_counts[c] = cell[task_sentence[c].to(torch.int32)]

        if args.loss_name in ["cross_entropy", "bce_mmd"]:
            task_counts[c] = (task_counts[c] > 0).float()

        task_sentence[c] = ds_emb_idxs[task_sentence[c].to(torch.int32)]

        task_gene_set = torch.tensor(task_sentence[c].tolist(), dtype=cell_sentences.dtype)
        potential_mask = torch.isin(cell_sentences[c], task_gene_set)

        target_mask_count = int(args.mask_target_pct * args.pad_length)
        current_mask_count = potential_mask.sum().item()

        if current_mask_count > target_mask_count:
            mask_indices = torch.where(potential_mask[1:])[0] + 1
            keep_indices = torch.randperm(len(mask_indices))[:target_mask_count]
            selected_indices = mask_indices[keep_indices]

            final_mask = torch.zeros_like(potential_mask)
            final_mask[selected_indices] = True
            mask[c] = final_mask
        elif current_mask_count < target_mask_count:
            non_masked = ~potential_mask
            non_masked_indices = torch.where(non_masked[1:])[0] + 1

            additional_needed = target_mask_count - current_mask_count
            additional_needed = min(additional_needed, len(non_masked_indices))

            if len(non_masked_indices) > 0 and additional_needed > 0:
                additional_indices = non_masked_indices[torch.randperm(len(non_masked_indices))[:additional_needed]]
                potential_mask[additional_indices] = True

            mask[c] = potential_mask
        else:
            mask[c] = potential_mask

        mask[c, 0] = False

    return cell_sentences.long(), mask, task_sentence.long(), task_counts
