MODEL = "/home/ecal/scratch/model_sent140_def.pth"
SAVE = "/home/ecal/scratch/sent140-probs.npy"

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from transformers import BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased", cache_dir="/home/ecal/scratch/"
)


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(tokenizer.vocab_size, 512)
        self.transformer = nn.Transformer(d_model=512, nhead=8)
        self.fc = nn.Linear(512, 2)

    def forward(self, x, mask):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, x, src_key_padding_mask=mask)
        x = x.permute(1, 0, 2)
        x = x.mean(dim=1)
        return self.fc(x)


model = Transformer().to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(MODEL))
model.eval()
test = torch.load("/home/ecal/scratch/sent140-test.pth")
test_dl = DataLoader(test, batch_size=16)

all_outs = []
with torch.no_grad():
    for inputs, mask, _ in test_dl:
        inputs = inputs.to(device)
        mask = mask.to(device)
        outputs = model(inputs, mask)
        all_outs.extend([*outputs.cpu().detach().numpy()])
    
np.save(SAVE, np.array(all_outs))
