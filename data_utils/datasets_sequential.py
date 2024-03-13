from scipy.sparse import coo_matrix, dok_matrix
import torch.utils.data as data
from config.configurator import configs
import numpy as np
import torch
#import dgl
import pandas as pd
from collections import defaultdict, Counter
import scipy.sparse as sp
from os import path
from sklearn.metrics.pairwise import cosine_similarity



class SequentialDataset(data.Dataset):
    def __init__(self, user_seqs, dynamic_context, static_context, mode='train', user_seqs_aug=None):
        self.mode = mode

        self.max_dynamic_context_length = configs['data']['max_context_length']
        self.max_seq_len = configs['model']['max_seq_len']
        self.user_history_lists = {user: seq for user,
                                   seq in zip(user_seqs["uid"], user_seqs["item_seq"])}
        self.user_history_time_delta_lists = {user: time_delta for user,
                                   time_delta in zip(user_seqs["uid"], user_seqs["time_delta"])}
        self.static_context = static_context
        self.dynamic_context = dynamic_context
        if user_seqs_aug is not None:
            self.uids = user_seqs_aug["uid"]
            self.seqs = user_seqs_aug["item_seq"]
            self.last_items = user_seqs_aug["item_id"]
            self.time_delta = user_seqs_aug["time_delta"]
        else:
            self.uids = user_seqs["uid"]
            self.seqs = user_seqs["item_seq"]
            self.last_items = user_seqs["item_id"]
            self.time_delta = user_seqs["time_delta"]
            self.common_item = user_seqs["common_item"]

        if mode == 'test':
            self.test_users = self.uids
            self.user_pos_lists = np.asarray(
                self.last_items, dtype=np.int32).reshape(-1, 1).tolist()

    def _pad_seq(self, seq):
        if len(seq) >= self.max_seq_len:
            seq = seq[-self.max_seq_len:]
        else:
            # pad at the head
            seq = [0] * (self.max_seq_len - len(seq)) + seq
        return seq
    
    def _pad_time_delta(self, seq, time_delta):
        if len(seq) >= self.max_seq_len:
            time_delta = time_delta[-self.max_seq_len:]
        else:
            # pad at the head
            time_delta = [0] * (self.max_seq_len - len(seq)) + time_delta
        return time_delta

    def _pad_context(self, lst):
        if len(lst) < self.max_dynamic_context_length:
            zeros_before = (self.max_dynamic_context_length - len(lst))
            return [0] * zeros_before + lst
        else:
            return lst

    def sample_negs(self):
        if 'neg_samp' in configs['data'] and configs['data']['neg_samp']:
            self.negs = []
            for i in range(len(self.uids)):
                u = self.uids[i]
                seq = self.user_history_lists[u]
                last_item = self.last_items[i]
                while True:
                    iNeg = np.random.randint(1, configs['data']['item_num'])
                    if iNeg not in seq and iNeg != last_item:
                        break
                self.negs.append(iNeg)
        else:
            pass

    def _process_context(self, context_i, context_type):
        context_keys = context_i.keys()
        context_values = [context_i[key] for key in context_keys]
        if context_type == 'dynamic':
            context = [self._pad_context(inner_list) for inner_list in context_values]
            return context
        else:   
            return context_values

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        try:
            seq_i = self.seqs[idx]
            time_delta_i = self.time_delta[idx]
            padded_dynamic_context = self._process_context(self.dynamic_context[idx], context_type='dynamic')
            static_context= self._process_context(self.static_context[idx], context_type='static')
            
            if self.mode == 'train' and 'neg_samp' in configs['data'] and configs['data']['neg_samp']:
                result = (
                    self.uids[idx],
                    torch.LongTensor(self._pad_seq(seq_i)),
                    self.last_items[idx],
                    torch.LongTensor(self._pad_time_delta(seq_i, time_delta_i)),
                    torch.LongTensor(padded_dynamic_context),
                    torch.LongTensor(static_context),
                    len(seq_i),
                    self.common_item[idx],
                    self.negs[idx]
                )
            else:
                result = (
                    self.uids[idx],
                    torch.LongTensor(self._pad_seq(seq_i)),
                    self.last_items[idx],
                    torch.LongTensor(self._pad_time_delta(seq_i, time_delta_i)),
                    torch.LongTensor(padded_dynamic_context),
                    torch.LongTensor(static_context),
                    len(seq_i),
                    self.common_item[idx]
                )
            
            # print("uid:", result[0])
            # print("padded_seq shape:", result[1].shape)
            # print("last_item:", result[2])
            # print("padded_time_delta shape:", result[3].shape)
            # print("context_values shape:", result[4].shape if len(result) > 4 else None)

            return result
        except Exception as e:
            print("Error:", e)
            raise