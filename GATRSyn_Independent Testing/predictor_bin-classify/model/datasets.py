import numpy as np
import torch

import random

from torch.utils.data import Dataset
from .utils import read_map
from sklearn.decomposition import PCA

# class FastTensorDataLoader:
#     """
#     A DataLoader-like object for a set of tensors that can be much faster than
#     TensorDataset + DataLoader because dataloader grabs individual indices of
#     the dataset and calls cat (slow).
#     Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
#     """
#
#     def __init__(self, *tensors, batch_size=32, shuffle=False):
#         """
#         Initialize a FastTensorDataLoader.
#         :param *tensors: tensors to store. Must have the same length @ dim 0.
#         :param batch_size: batch size to load.
#         :param shuffle: if True, shuffle the data *in-place* whenever an
#             iterator is created out of this object.
#         :returns: A FastTensorDataLoader.
#         """
#         assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
#         self.tensors = tensors
#
#         self.dataset_len = self.tensors[0].shape[0]
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#
#         # Calculate # batches
#         n_batches, remainder = divmod(self.dataset_len, self.batch_size)
#         if remainder > 0:
#             n_batches += 1
#         self.n_batches = n_batches
#
#     def __iter__(self):
#         if self.shuffle:
#             r = torch.randperm(self.dataset_len)
#             self.tensors = [t[r] for t in self.tensors]
#         self.i = 0
#         return self
#
#     def __next__(self):
#         if self.i >= self.dataset_len:
#             raise StopIteration
#         batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
#         self.i += self.batch_size
#         return batch
#
#     def __len__(self):
#         return self.n_batches
class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors and lists that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors_and_lists, batch_size=32, shuffle=False, drop_last=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors_and_lists: tensors and lists to store. Tensors must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :param drop_last: if True, drop the last incomplete batch.
        :returns: A FastTensorDataLoader.
        """
        self.tensors = []
        self.lists = []
        self.is_tensor = []

        for item in tensors_and_lists:
            if isinstance(item, torch.Tensor):
                self.tensors.append(item)
                self.is_tensor.append(True)
            else:
                self.lists.append(item)
                self.is_tensor.append(False)

        assert all(t.shape[0] == self.tensors[0].shape[0] for t in
                   self.tensors), "All tensors must have the same length at dim 0."
        assert all(
            len(l) == len(self.tensors[0]) for l in self.lists), "All lists must have the same length as tensors."

        self.drop_last = drop_last
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate number of batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if not self.drop_last or remainder == 0:
            self.n_batches = n_batches
        else:
            self.n_batches = n_batches - 1

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
            self.lists = [[l[i] for i in r] for l in self.lists]
        self.i = 0
        return self

    def __next__(self):
        if self.drop_last and self.i + self.batch_size > self.dataset_len:
            raise StopIteration
        if self.i >= self.dataset_len:
            raise StopIteration

        batch_tensors = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
        batch_lists = tuple(
            [l[self.i + j] for j in range(min(self.batch_size, self.dataset_len - self.i))] for l in self.lists)

        self.i += self.batch_size
        return (*batch_tensors, *batch_lists)

    def __len__(self):
        return self.n_batches


class EmbDataset(Dataset):

    def __init__(self, drug2id_file, cell2id_file, synergy_score_file, use_folds):
        self.drug2id = read_map(drug2id_file)
        self.cell2id = read_map(cell2id_file)
        self.samples = []
        valid_drugs = set(self.drug2id.keys())
        valid_cells = set(self.cell2id.keys())
        with open(synergy_score_file, 'r') as f:
            f.readline()
            for line in f:
                drug1, drug2, cellname, score, fold = line.rstrip().split('\t')
                if drug1 in valid_drugs and drug2 in valid_drugs and cellname in valid_cells:
                    if int(fold) in use_folds:
                        sample = [self.drug2id[drug1], self.drug2id[drug2], self.cell2id[cellname], float(score)]
                        self.samples.append(sample)
                        sample = [self.drug2id[drug2], self.drug2id[drug1], self.cell2id[cellname], float(score)]
                        self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        drug1_id, drug2_id, cell_id, score = self.samples[item]
        drug1_feat = torch.LongTensor([drug1_id])
        drug2_feat = torch.LongTensor([drug2_id])
        cell_feat = torch.LongTensor([cell_id])
        score = torch.FloatTensor([score])
        return  drug1_feat, drug2_feat, cell_feat, score,drug1_id, drug2_id, cell_id


class PPIDataset(Dataset):

    def __init__(self, exp_file):
        self.expression = np.load(exp_file)

    def __len__(self):
        return self.expression.shape[0]

    def __getitem__(self, item):
        return torch.LongTensor([item]), torch.FloatTensor(self.expression[item])


class AEDataset(Dataset):

    def __init__(self, feat_file):
        self.feat = np.load(feat_file)

    def __len__(self):
        return self.feat.shape[0]

    def __getitem__(self, item):
        return torch.FloatTensor(self.feat[item]), torch.FloatTensor(self.feat[item])


class SynergyDataset(Dataset):

    def __init__(self, drug2id_file, cell2id_file, drug_feat_file, cell_feat_file, synergy_score_file, use_folds,
                 train=True):
        self.drug2id = read_map(drug2id_file)
        self.cell2id = read_map(cell2id_file)
        self.drug_feat = np.load(drug_feat_file)
        self.cell_feat = np.load(cell_feat_file)
        self.samples = []
        valid_drugs = set(self.drug2id.keys())
        valid_cells = set(self.cell2id.keys())
        with open(synergy_score_file, 'r') as f:
            f.readline()
            for line in f:
                drug1, drug2, cellname, score, fold = line.rstrip().split('\t')
                if drug1 in valid_drugs and drug2 in valid_drugs and cellname in valid_cells:
                    if int(fold) in use_folds:
                        sample = [self.drug2id[drug1], self.drug2id[drug2], self.cell2id[cellname], float(score)]
                        self.samples.append(sample)
                        if train:
                            sample = [self.drug2id[drug2], self.drug2id[drug1], self.cell2id[cellname], float(score)]
                            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        drug1_id, drug2_id, cell_id, score = self.samples[item]
        drug1_feat = torch.from_numpy(self.drug_feat[drug1_id]).float()
        drug2_feat = torch.from_numpy(self.drug_feat[drug2_id]).float()
        cell_feat = torch.from_numpy(self.cell_feat[cell_id]).float()
        score = torch.FloatTensor([score])
        return drug1_feat, drug2_feat, cell_feat, score

    def drug_feat_len(self):
        return self.drug_feat.shape[-1]

    def cell_feat_len(self):
        return self.cell_feat.shape[-1]


class FastSynergyDataset(Dataset):

    def __init__(self, drug2id_file, cell2id_file, drug_feat_file, cell_feat_file, synergy_score_file, use_folds,
                 train=True):
        self.drug2id = read_map(drug2id_file)
        self.cell2id = read_map(cell2id_file)
        self.drug_feat = np.load(drug_feat_file)
        self.cell_feat = np.load(cell_feat_file)

        # # 对药物特征进行降维
        # # drug_pca = PCA(n_components=100)
        # # drug_feat_reduced = drug_pca.fit_transform(self.drug_feat)
        # # 对细胞特征进行降维
        # cell_pca = PCA(n_components=1000)
        # self.cell_feat = cell_pca.fit_transform(self.cell_feat)

        self.samples = []
        self.raw_samples = []
        self.train = train
        valid_drugs = set(self.drug2id.keys())
        valid_cells = set(self.cell2id.keys())
        with open(synergy_score_file, 'r') as f:
            f.readline()
            for line in f:
                drug1, drug2, cellname, score, fold = line.rstrip().split('\t')
                if drug1 in valid_drugs and drug2 in valid_drugs and cellname in valid_cells:
                    if int(fold) in use_folds:
                        sample = [
                            torch.from_numpy(self.drug_feat[self.drug2id[drug1]]).float(),
                            torch.from_numpy(self.drug_feat[self.drug2id[drug2]]).float(),
                            torch.from_numpy(self.cell_feat[self.cell2id[cellname]]).float(),
                            torch.FloatTensor([float(score)]),
                            drug1,  # 添加药物1名称
                            drug2,  # 添加药物2名称
                            cellname  # 添加细胞名称
                        ]
                        self.samples.append(sample)
                        raw_sample = [self.drug2id[drug1], self.drug2id[drug2], self.cell2id[cellname], score]
                        self.raw_samples.append(raw_sample)
                        if train:
                            sample = [
                                torch.from_numpy(self.drug_feat[self.drug2id[drug2]]).float(),
                                torch.from_numpy(self.drug_feat[self.drug2id[drug1]]).float(),
                                torch.from_numpy(self.cell_feat[self.cell2id[cellname]]).float(),
                                torch.FloatTensor([float(score)]),
                                # drug1,  # 添加药物1名称
                                # drug2,  # 添加药物2名称
                                # cellname  # 添加细胞名称
                            ]
                            self.samples.append(sample)
                            raw_sample = [self.drug2id[drug2], self.drug2id[drug1], self.cell2id[cellname], score]
                            self.raw_samples.append(raw_sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]

    def drug_feat_len(self):
        return self.drug_feat.shape[-1]

    def cell_feat_len(self):
        return self.cell_feat.shape[-1]

    # 修改 tensor_samples 方法（返回4个张量）
    def tensor_samples(self, indices=None):
        if indices is None:
            indices = list(range(len(self)))
        d1 = torch.cat([torch.unsqueeze(self.samples[i][0], 0) for i in indices], dim=0)
        d2 = torch.cat([torch.unsqueeze(self.samples[i][1], 0) for i in indices], dim=0)
        c = torch.cat([torch.unsqueeze(self.samples[i][2], 0) for i in indices], dim=0)
        y = torch.cat([torch.unsqueeze(self.samples[i][3], 0) for i in indices], dim=0)
        # 删除多余的返回项（drug1_ids, drug2_ids, cell_ids）
        return d1, d2, c, y  # 只返回4个张量

    # def tensor_samples(self, indices=None):
    #     if indices is None:
    #         indices = list(range(len(self)))
    #     d1 = torch.cat([torch.unsqueeze(self.samples[i][0], 0) for i in indices], dim=0)
    #     d2 = torch.cat([torch.unsqueeze(self.samples[i][1], 0) for i in indices], dim=0)
    #     c = torch.cat([torch.unsqueeze(self.samples[i][2], 0) for i in indices], dim=0)
    #     y = torch.cat([torch.unsqueeze(self.samples[i][3], 0) for i in indices], dim=0)
    #     drug1_ids = [self.samples[i][4] for i in indices]
    #     drug2_ids = [self.samples[i][5] for i in indices]
    #     cell_ids = [self.samples[i][6] for i in indices]
    #
    #     return d1, d2, c, y, drug1_ids, drug2_ids, cell_ids

class DSDataset(Dataset):

    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, item):
        return torch.FloatTensor(self.samples[item]), torch.FloatTensor([self.labels[item]])
























# import numpy as np
# import torch
#
# import random
#
# from torch.utils.data import Dataset
# from .utils import read_map
# from sklearn.decomposition import PCA
#
# class FastTensorDataLoader:
#     """
#     A DataLoader-like object for a set of tensors that can be much faster than
#     TensorDataset + DataLoader because dataloader grabs individual indices of
#     the dataset and calls cat (slow).
#     Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
#     """
#
#     def __init__(self, *tensors, batch_size=32, shuffle=False,drop_last=False):
#         """
#         Initialize a FastTensorDataLoader.
#         :param *tensors: tensors to store. Must have the same length @ dim 0.
#         :param batch_size: batch size to load.
#         :param shuffle: if True, shuffle the data *in-place* whenever an
#             iterator is created out of this object.
#         :returns: A FastTensorDataLoader.
#         """
#         assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
#         self.tensors = tensors
#         self.drop_last = drop_last
#         self.dataset_len = self.tensors[0].shape[0]
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#
#         # Calculate # batches
#         n_batches, remainder = divmod(self.dataset_len, self.batch_size)
#         # if remainder > 0:
#         #     n_batches += 1
#         # self.n_batches = n_batches
#         if not self.drop_last or remainder == 0:
#                 self.n_batches = n_batches
#         else:
#                 self.n_batches = n_batches - 1
#
#     def __iter__(self):
#         if self.shuffle:
#             r = torch.randperm(self.dataset_len)
#             self.tensors = [t[r] for t in self.tensors]
#         self.i = 0
#         return self
#
#     def __next__(self):
#         if self.drop_last and self.i + self.batch_size > self.dataset_len:
#             raise StopIteration
#         if self.i >= self.dataset_len:
#             raise StopIteration
#         batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
#         self.i += self.batch_size
#         return batch
#
#     def __len__(self):
#         return self.n_batches
#
#
# class EmbDataset(Dataset):
#
#     def __init__(self, drug2id_file, cell2id_file, synergy_score_file, use_folds):
#         self.drug2id = read_map(drug2id_file)
#         self.cell2id = read_map(cell2id_file)
#         self.samples = []
#         valid_drugs = set(self.drug2id.keys())
#         valid_cells = set(self.cell2id.keys())
#         with open(synergy_score_file, 'r') as f:
#             f.readline()
#             for line in f:
#                 drug1, drug2, cellname, score, fold = line.rstrip().split('\t')
#                 if drug1 in valid_drugs and drug2 in valid_drugs and cellname in valid_cells:
#                     if int(fold) in use_folds:
#                         sample = [self.drug2id[drug1], self.drug2id[drug2], self.cell2id[cellname], float(score)]
#                         self.samples.append(sample)
#                         sample = [self.drug2id[drug2], self.drug2id[drug1], self.cell2id[cellname], float(score)]
#                         self.samples.append(sample)
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, item):
#         drug1_id, drug2_id, cell_id, score = self.samples[item]
#         drug1_feat = torch.LongTensor([drug1_id])
#         drug2_feat = torch.LongTensor([drug2_id])
#         cell_feat = torch.LongTensor([cell_id])
#         score = torch.FloatTensor([score])
#         return drug1_feat, drug2_feat, cell_feat, score
#
#
# class PPIDataset(Dataset):
#
#     def __init__(self, exp_file):
#         self.expression = np.load(exp_file)
#
#     def __len__(self):
#         return self.expression.shape[0]
#
#     def __getitem__(self, item):
#         return torch.LongTensor([item]), torch.FloatTensor(self.expression[item])
#
#
# class AEDataset(Dataset):
#
#     def __init__(self, feat_file):
#         self.feat = np.load(feat_file)
#
#     def __len__(self):
#         return self.feat.shape[0]
#
#     def __getitem__(self, item):
#         return torch.FloatTensor(self.feat[item]), torch.FloatTensor(self.feat[item])
#
#
# class SynergyDataset(Dataset):
#
#     def __init__(self, drug2id_file, cell2id_file, drug_feat_file, cell_feat_file, synergy_score_file, use_folds,
#                  train=True):
#         self.drug2id = read_map(drug2id_file)
#         self.cell2id = read_map(cell2id_file)
#         self.drug_feat = np.load(drug_feat_file)
#         self.cell_feat = np.load(cell_feat_file)
#         self.samples = []
#         valid_drugs = set(self.drug2id.keys())
#         valid_cells = set(self.cell2id.keys())
#         with open(synergy_score_file, 'r') as f:
#             f.readline()
#             for line in f:
#                 drug1, drug2, cellname, score, fold = line.rstrip().split('\t')
#                 if drug1 in valid_drugs and drug2 in valid_drugs and cellname in valid_cells:
#                     if int(fold) in use_folds:
#                         sample = [self.drug2id[drug1], self.drug2id[drug2], self.cell2id[cellname], float(score)]
#                         self.samples.append(sample)
#                         if train:
#                             sample = [self.drug2id[drug2], self.drug2id[drug1], self.cell2id[cellname], float(score)]
#                             self.samples.append(sample)
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, item):
#         drug1_id, drug2_id, cell_id, score = self.samples[item]
#         drug1_feat = torch.from_numpy(self.drug_feat[drug1_id]).float()
#         drug2_feat = torch.from_numpy(self.drug_feat[drug2_id]).float()
#         cell_feat = torch.from_numpy(self.cell_feat[cell_id]).float()
#         score = torch.FloatTensor([score])
#         return drug1_feat, drug2_feat, cell_feat, score
#
#     def drug_feat_len(self):
#         return self.drug_feat.shape[-1]
#
#     def cell_feat_len(self):
#         return self.cell_feat.shape[-1]
#
#
# class FastSynergyDataset(Dataset):
#
#     def __init__(self, drug2id_file, cell2id_file, drug_feat_file, cell_feat_file, synergy_score_file, use_folds,
#                  train=True):
#         self.drug2id = read_map(drug2id_file)
#         self.cell2id = read_map(cell2id_file)
#         self.drug_feat = np.load(drug_feat_file)
#         self.cell_feat = np.load(cell_feat_file)
#
#
#         self.samples = []
#         self.raw_samples = []
#         self.train = train
#         valid_drugs = set(self.drug2id.keys())
#         valid_cells = set(self.cell2id.keys())
#         with open(synergy_score_file, 'r') as f:
#             f.readline()
#             for line in f:
#                 drug1, drug2, cellname, score, fold = line.rstrip().split('\t')
#                 if drug1 in valid_drugs and drug2 in valid_drugs and cellname in valid_cells:
#                     if int(fold) in use_folds:
#                         sample = [
#                             torch.from_numpy(self.drug_feat[self.drug2id[drug1]]).float(),
#                             torch.from_numpy(self.drug_feat[self.drug2id[drug2]]).float(),
#                             torch.from_numpy(self.cell_feat[self.cell2id[cellname]]).float(),
#                             torch.FloatTensor([float(score)]),
#                             drug1,  # 添加药物1名称
#                             drug2,  # 添加药物2名称
#                             cellname  # 添加细胞名称
#                         ]
#                         self.samples.append(sample)
#                         raw_sample = [self.drug2id[drug1], self.drug2id[drug2], self.cell2id[cellname], score]
#                         self.raw_samples.append(raw_sample)
#                         if train:
#                             sample = [
#                                 torch.from_numpy(self.drug_feat[self.drug2id[drug2]]).float(),
#                                 torch.from_numpy(self.drug_feat[self.drug2id[drug1]]).float(),
#                                 torch.from_numpy(self.cell_feat[self.cell2id[cellname]]).float(),
#                                 torch.FloatTensor([float(score)]),
#                                 drug1,  # 添加药物1名称
#                                 drug2,  # 添加药物2名称
#                                 cellname  # 添加细胞名称
#                             ]
#                             self.samples.append(sample)
#                             raw_sample = [self.drug2id[drug2], self.drug2id[drug1], self.cell2id[cellname], score]
#                             self.raw_samples.append(raw_sample)
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, item):
#         return self.samples[item]
#
#     def drug_feat_len(self):
#         return self.drug_feat.shape[-1]
#
#     def cell_feat_len(self):
#         return self.cell_feat.shape[-1]
#
#     def tensor_samples(self, indices=None):
#         if indices is None:
#             indices = list(range(len(self)))
#         d1 = torch.cat([torch.unsqueeze(self.samples[i][0], 0) for i in indices], dim=0)
#         d2 = torch.cat([torch.unsqueeze(self.samples[i][1], 0) for i in indices], dim=0)
#         c = torch.cat([torch.unsqueeze(self.samples[i][2], 0) for i in indices], dim=0)
#         y = torch.cat([torch.unsqueeze(self.samples[i][3], 0) for i in indices], dim=0)
#         drug1_ids = [self.samples[i][4] for i in indices]
#         drug2_ids = [self.samples[i][5] for i in indices]
#         cell_ids = [self.samples[i][6] for i in indices]
#         return d1, d2, c, y, drug1_ids, drug2_ids, cell_ids
#
# class DSDataset(Dataset):
#
#     def __init__(self, samples, labels):
#         self.samples = samples
#         self.labels = labels
#
#     def __len__(self):
#         return self.samples.shape[0]
#
#     def __getitem__(self, item):
#         return torch.FloatTensor(self.samples[item]), torch.FloatTensor([self.labels[item]])
