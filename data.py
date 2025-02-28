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
                    batch_sentences, mask, cell_outputs_X_pe, \
                    cell_outputs_Y, seq_len = \
                        sample_cell_sentences(counts, weights, dataset, self.args,
                            dataset_to_protein_embeddings= self.dataset_to_protein_embeddings,
                            dataset_to_chroms=self.dataset_to_chroms,
                            dataset_to_starts=self.dataset_to_starts)
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



def sample_cell_sentences(counts, batch_weights, dataset, args,
                          dataset_to_protein_embeddings,
                          dataset_to_chroms,
                          dataset_to_starts):
    
    dataset_idxs = dataset_to_protein_embeddings[dataset]
    cell_sentences = torch.zeros((counts.shape[0], args.pad_length))
    # pos = adata.X > 0
    mask = torch.zeros((counts.shape[0], args.pad_length), dtype=bool)

    chroms = dataset_to_chroms[dataset]
    starts = dataset_to_starts[dataset]

    longest_seq_len = 0
    cell_outputs_X = torch.zeros((counts.shape[0], args.P + args.N))
    cell_outputs_Y = torch.zeros((counts.shape[0], args.P + args.N))

    for c, cell in enumerate(counts):
        pos_genes = torch.where(counts[c] > 0)[0]
        neg_genes = torch.where(counts[c] < 1)[0]
        if len(pos_genes) == 0:
            pos_genes = neg_genes

        weights = batch_weights[c].numpy()
        # pos_gene_embeds = adata_pe[pos_genes]

        # randomly choose some pos genes to mask out.

        # start with 20% random dropout
        mask_weights = np.random.choice(pos_genes,
                                        size=round(len(pos_genes) * args.mask_prop),
                                        replace=False)
        # Clip so no value is 10x larger than smaller value.
        #min_val = np.min(cell[pos_genes])
        #new_max_val = (min_val * 10)
        #proportion_clipped = np.mean(cell[pos_genes] > new_max_val)
        
        weights[mask_weights] = 0  # drop these out
        weights = weights / sum(weights)  # RE NORM after mask
        # clip
        #weights = np.clip(weights, a_min=0, a_max=0.005) # P(binomial(1024, 0.005) >= 10) = 0.036
        #weights = weights / sum(weights)  # RE NORM after clip
        # mask.append(torch.ones(sample_size))
        choice_idx = np.random.choice(np.arange(len(weights)),
                                      size=args.sample_size, p=weights,
                                      replace=True)
        choosen_chrom = chroms[choice_idx]
        chrom_sort = np.argsort(choosen_chrom)  # order by chromsome
        choice_idx = choice_idx[chrom_sort]  # now ordered by chrom

        # sort by start
        new_chrom = chroms[choice_idx]
        choosen_starts = starts[choice_idx]

        ordered_choice_idx = np.full((args.pad_length),
                                     args.cls_token_idx)  # start with cls
        # i= 0 first token is CLS
        i = 1  # continue on to the rest of the sequence with left bracket being assumed.\
        # Shuffle the chroms now
        uq_chroms = np.unique(new_chrom)
        np.random.shuffle(uq_chroms) # shuffle
        for chrom in uq_chroms:
            # Open Chrom
            ordered_choice_idx[i] = int(chrom) + args.CHROM_TOKEN_OFFSET # token of this chromosome # i = 1 next token is a chrom open
            i += 1
            # now sort the by start order within the chroms
            loc = np.where(new_chrom == chrom)[0]
            sort_by_start = np.argsort(
                choosen_starts[loc])  # start locations for these chromsomes

            to_add = choice_idx[loc[sort_by_start]]
            ordered_choice_idx[i:(i + len(to_add))] = dataset_idxs[to_add]  # convert
            i += len(to_add)
            ordered_choice_idx[i] = args.chrom_token_right_idx # add the chrom sep again
            i += 1  # add the closing token again

        longest_seq_len = max(longest_seq_len, i)
        remainder_len = (args.pad_length - i)

        cell_mask = torch.concat((torch.zeros(i, dtype=bool),
                                  # pay attention to all of these tokens, ignore the rest!
                                  torch.ones(remainder_len, dtype=bool)))

        mask[c, :] = cell_mask

        ordered_choice_idx[i:] = args.pad_token_idx  # mask

        # sample_row = pos_gene_embeds[choice_idx, :]
        cell_sentences[c, :] = torch.from_numpy(ordered_choice_idx)
        choice_idx_ouput_p = mask_weights  # use the masked genes as task
        if len(choice_idx_ouput_p) > args.P:
            choice_idx_ouput_p = np.random.choice(choice_idx_ouput_p,
                                                  replace=False,
                                                  size=args.P)  # subset of masked genes
        elif len(choice_idx_ouput_p) < args.P:
            remainder = args.P - len(choice_idx_ouput_p)  # remaining to be choosen
            choice_idx_ouput_p = np.append(choice_idx_ouput_p,
                                           np.random.choice(pos_genes,
                                                            size=remainder,
                                                            replace=True))  # choose more

        # choice_idx_ouput_p = pos_genes[choice_idx_ouput_p]
        if args.N <= len(neg_genes):
            choice_idx_ouput_n = np.random.choice(np.arange(len(neg_genes)),
                                                  size=args.N, replace=False)
        else:
            choice_idx_ouput_n = np.random.choice(np.arange(len(neg_genes)),
                                                  size=args.N, replace=True)

        choice_idx_ouput_n = neg_genes[choice_idx_ouput_n]

        cell_outputs_X[c] = torch.tensor(
            np.concatenate((choice_idx_ouput_p, choice_idx_ouput_n)))
        cell_outputs_Y[c] = torch.cat((torch.ones(args.P), torch.zeros(args.N)))

    cell_sentences_pe = cell_sentences.long()  # .unsqueeze(2) # all_pe[cell_sentences.long(), :]
    cell_outputs_X_pe = dataset_idxs[
        cell_outputs_X.long()]  # .unsqueeze(2) # all_pe[dataset_idxs[cell_outputs_X.long()], :]

    return cell_sentences_pe, mask, cell_outputs_X_pe, cell_outputs_Y, args.pad_length
