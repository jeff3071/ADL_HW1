from typing import List, Dict

import torch
from torch.utils.data import Dataset

from utils import Vocab
from utils import pad_to_len

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.data_, self.label = self.collate_fn(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        if not self.label:
            return self.data_[index]
        return self.data_[index], self.label[index]

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        text = [data['text'].split() for data in samples]
        text2idx = torch.LongTensor(self.vocab.encode_batch(text ,self.max_len))
        if 'intent' not in samples[-1].keys(): # test data
            return text2idx, None
        intent2idx = [self.label2idx(data['intent']) for data in samples]
        return text2idx, intent2idx


    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

class SlotDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: tag for tag, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.data_, self.label, self.text_len = self.collate_fn(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        if self.label is None:
          return self.data_[index], self.text_len[index]
        return self.data_[index], self.label[index], self.text_len[index]

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        text = []
        text_len = []
        for data in samples:
          text.append(data['tokens'])
          text_len.append(len(data['tokens']))
        
        text2idx = torch.LongTensor(self.vocab.encode_batch(text ,self.max_len))
        
        if 'tags' not in (samples[-1].keys()):
            return text2idx,None, text_len
        
        tags_list = [data['tags'] for data in samples]
        tag2idx = []
        for tags in tags_list:
          now_tags = [9]*self.max_len   # pad data, 9 is pad idx
          for i, tag in enumerate(tags):
            now_tags[i] = self.label2idx(tag)
          tag2idx.append(now_tags)

        tag2idx = torch.LongTensor(tag2idx)
        return text2idx, tag2idx, text_len #accuray and predict result need


    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

