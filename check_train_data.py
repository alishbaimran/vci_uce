import utils
import pandas as pd
from tqdm.auto import tqdm
import pickle
from multiprocessing import Pool
import numpy as np

dataset_path = "/dfs/project/uce/response/full_train_datasets.csv"
datasets_df = pd.read_csv(dataset_path)
shapes_dict = utils.get_shapes_dict(dataset_path=dataset_path)

def proc(i):
    dataset = datasets_df["names"][i]
    shape = shapes_dict[dataset]
    cts = np.memmap(f"/lfs/local/0/yanay/cxg_npzs/" + f"{dataset}_counts.npz", dtype='int64', mode='r', shape=shape)
    #cts = np.array(cts) # read into memory
    # count number of unique genes
    # trackers
    n_uniq_all = []
    counts_all = []
    n_genes_masked_all = []
    n_uq_choices_all = []
    max_count_all = []
    
    max_val_all = []
    proportion_clipped_all = []
    for j in range(shape[0]):
        cell = cts[j]
        # count number of unique genes
        n_uniq = np.sum(cts > 0)
        n_uniq_all.append(n_uniq)
        # do a sample
        pos_genes = np.where(cell > 0)[0]
        cell = np.log1p(cell)
        weights = cell / sum(cell)
        # New clipping scheme
        # The max value should not be more than 10x the min value. That represents 2^10 = 1000x more counts
        min_val = np.min(cell[pos_genes])
        new_max_val = (min_val * 10)
        proportion_clipped = np.mean(cell[pos_genes] > new_max_val)
        max_val_all.append(new_max_val)
        proportion_clipped_all.append(proportion_clipped)
        
        weights = weights = np.clip(weights, a_min=0, a_max=new_max_val) # P(binomial(1024, 0.005) >= 10) = 0.036
        mask_weights = np.random.choice(pos_genes,
                                        size=round(len(pos_genes) * 0.2),
                                        replace=False)
        weights[mask_weights] = 0  # drop these out
        weights = weights / sum(weights)  # RE NORM after mask
        # clip
        # after clip
        # mask.append(torch.ones(sample_size))
        choice_idx = np.random.choice(np.arange(len(weights)), size=1024, p=weights, replace=True)
        n_genes_masked = len(mask_weights)
        n_genes_masked_all.append(n_genes_masked)
        uq, counts = np.unique(choice_idx, return_counts=True)
        n_uq_choices = len(uq)
        n_uq_choices_all.append(n_uq_choices)
        max_count = max(counts)
        max_count_all.append(max_count)
        #counts_all.append(counts) # the distribution of counts

    save_path = f"/lfs/local/0/yanay/data_info_uce_train_clip_max10/{dataset}_"
    with open(save_path + 'n_uniq_all.pickle', 'wb+') as handle:
        pickle.dump(n_uniq_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #with open(save_path + 'counts_all.pickle', 'wb+') as handle:
    #    pickle.dump(counts_all, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    with open(save_path + 'n_genes_masked_all.pickle', 'wb+') as handle:
        pickle.dump(n_genes_masked_all, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    with open(save_path + 'n_uq_choices_all.pickle', 'wb+') as handle:
        pickle.dump(n_uq_choices_all, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    with open(save_path + 'max_count_all.pickle', 'wb+') as handle:
        pickle.dump(max_count_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(save_path + 'max_val_all.pickle', 'wb+') as handle:
        pickle.dump(max_val_all, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    with open(save_path + 'proportion_clipped_all.pickle', 'wb+') as handle:
        pickle.dump(proportion_clipped_all, handle, protocol=pickle.HIGHEST_PROTOCOL) 

if __name__ == '__main__':
    I = np.arange(len(datasets_df))
    np.random.shuffle(I) # shuffle rows so we don't overflow memory
    with Pool(48) as p:
        with tqdm(total=len(I)) as pbar:
                for _ in p.imap_unordered(proc, I):
                    pbar.update()