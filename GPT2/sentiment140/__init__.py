import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Literal, Optional
from os import path
import re


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


class Sentiment140Dataset(Dataset):
    def __init__(self, split: Literal['train', 'dev', 'test'], data_path: Optional[str]=None):
        cwd = path.dirname(__file__)

        if data_path is None:
            data_path = path.join(cwd, 'data')
        
        csv_path = path.join(data_path, f"{split}.csv")
        self.data_df = pd.read_csv(csv_path)
        self.data_df["text"] = preprocess_series(self.data_df["text"])
        self.labels = ['negative', 'positive']

    def __len__(self):
        return self.data_df.shape[0]
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        text = row["text"]
        label = row["sentiment"]

        return text, label
