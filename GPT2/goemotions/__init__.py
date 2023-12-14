import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Literal, Optional
from os import path
import json
import re
import random


def preprocess_series(series: pd.Series):
    hashtags = re.compile(r"^#\S+|\s#\S+")
    mentions = re.compile(r"^@\S+|\s@\S+")
    urls = re.compile(r"https?://\S+")

    def process_text(text):
        text = re.sub(r'http\S+', '', text)
        text = hashtags.sub(' hashtag', text)
        text = mentions.sub(' entity', text)
        return text.strip()

    return series.apply(process_text)


class GoEmotionsDataset(Dataset):
    def __init__(self, split: Literal['train', 'dev', 'test'], data_path: Optional[str]=None):
        cwd = path.dirname(__file__)

        if data_path is None:
            data_path = path.join(cwd, 'data')
        
        tsv_path = path.join(data_path, f"{split}.tsv")
        self.data_df = pd.read_csv(tsv_path, sep="\t", header=None)
        self.data_df[0] = preprocess_series(self.data_df[0])

        with open(path.join(cwd, 'emotions.txt'), 'r') as fp:
            self.emotions = fp.read().splitlines()
        # self.mapping, self.labels = self.__reverse_mapping()
        self.labels = self.emotions

    def __reverse_mapping(self):
        with open(path.join(path.dirname(__file__), 'sentiment_mapping.json'), 'r') as fp:
            mapping = json.load(fp)

        reverse_mapping = {}
        labels = []
        for idx, (key, value) in enumerate(mapping.items()):
            for v in value:
                reverse_mapping[v] = idx
            labels.append(key)
        reverse_mapping['neutral'] = len(mapping)
        labels.append('neutral')
        
        return reverse_mapping, labels

    def __len__(self):
        return self.data_df.shape[0]
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        text = row[0]
        label_index = int(random.choice(row[1].split(',')))  # select one randomly
        # label_index = int(row[1].split(',')[0])
        # label = self.mapping[self.emotions[label_index]]

        return text, label_index
