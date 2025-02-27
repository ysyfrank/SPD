import torch
import random
import datasets
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Union, List, Tuple, Dict

class PointDataset(Dataset):
    def __init__(self, filename, max_seq_length, tokenizer, dataset_script_dir, dataset_cache_dir):
        self._filename = filename
        self._max_seq_length = max_seq_length
        self._tokenizer = tokenizer
        self.ir_dataset = datasets.load_dataset(
            f'{dataset_script_dir}/json.py',
            data_files=self._filename,
            ignore_verifications=False,
            cache_dir=dataset_cache_dir,
            features=datasets.Features({
                "queryid": datasets.Value("string"),
                'query_ids': [datasets.Value("int32")],
                "docid": datasets.Value("string"),
                'doc_ids': [datasets.Value("int32")],
            })
        )['train']
        self.total_len = len(self.ir_dataset)  
      
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, item):
        data = self.ir_dataset[item]
        data = self.add_padding(data)
        return {
            "queryid": data["queryid"],
            "query_ids": np.array(data["query_ids"]),
            "query_attention_mask": np.array(data["query_attention_mask"]),
            "docid": data["docid"],
            "doc_ids": np.array(data["doc_ids"]),
            "doc_attention_mask": np.array(data["doc_attention_mask"]),
            "labels": np.array(data["labels"])
        }

    def add_padding(self, training_instance):
        padded_query_ids = [0 for i in range(self._max_seq_length)]
        padded_query_attention = [0 for i in range(self._max_seq_length)]
        padded_doc_ids = [0 for i in range(self._max_seq_length)]
        padded_doc_attention = [0 for i in range(self._max_seq_length)]
        labels = [1]

        query_ids = training_instance['query_ids'][:-1][:(self._max_seq_length-1)] + [1]
        for i, iid in enumerate(query_ids):
            padded_query_ids[i] = iid
            padded_query_attention[i] = 1

        doc_ids = training_instance['doc_ids'][:-1][:(self._max_seq_length-1)] + [1]
        for i, iid in enumerate(doc_ids):
            padded_doc_ids[i] = iid
            padded_doc_attention[i] = 1

        new_instance = {
            "query_ids": padded_query_ids,
            "query_attention_mask": padded_query_attention,
            "queryid": training_instance["queryid"],
            "doc_ids": padded_doc_ids,
            "doc_attention_mask": padded_doc_attention,
            "docid": training_instance["docid"],
            "labels": labels
        }
        return new_instance
