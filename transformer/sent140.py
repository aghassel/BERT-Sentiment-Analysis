import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from transformers import BertTokenizer
from tqdm import tqdm

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
        x = x.permute(1, 0, 2)  # Transpose for transformer
        x = self.transformer(x, x, src_key_padding_mask=mask)
        x = x.permute(1, 0, 2)  # Transpose back
        x = x.mean(dim=1)
        return self.fc(x)


def load_data():
    import os
    if not os.path.exists("/home/ecal/scratch/sent140-train.pth"):
        import re

        df = pd.read_csv(
            "/home/ecal/scratch/data/training.1600000.processed.noemoticon.csv",
            encoding="latin-1",
            header=None
        )

        df.columns = ["label", "id", "Date", "Query", "User", "text"]
        df = df.drop(columns=["id", "Date", "Query", "User"], axis=1)
        df["label"] = df["label"].replace(4, 1)

        hashtags = re.compile(r"^#\S+|\s#\S+")
        mentions = re.compile(r"^@\S+|\s@\S+")
        urls = re.compile(r"https?://\S+")

        def process_text(text):
            text = urls.sub("", text)
            text = hashtags.sub(" hashtag", text)
            text = mentions.sub(" entity", text)
            return text.strip().lower()

        df["text"] = df["text"].apply(process_text)

        labels = df["label"].values
        texts = df["text"].values

        input_ids = []
        attention_masks = []
        for text in texts:
            encoded = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=512,
                padding="max_length",
                return_attention_mask=True,
                truncation=True,
                return_tensors="pt"
            )
            input_ids.append(encoded["input_ids"])
            attention_masks.append(encoded["attention_mask"].bool())
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)

        print("Input IDs shape:", input_ids.shape)
        print("Attention Masks shape:", attention_masks.shape)
        print("Labels shape:", labels.shape)

        train_size = int(0.9 * len(input_ids))
        train = TensorDataset(
            input_ids[:train_size], attention_masks[:train_size], labels[:train_size]
        )
        test = TensorDataset(
            input_ids[train_size:], attention_masks[train_size:], labels[train_size:]
        )

        torch.save(train, "/home/ecal/scratch/sent140-train.pth")
        torch.save(test, "/home/ecal/scratch/sent140-test.pth")
        print("Saved train and test datasets")
    else:
        train = torch.load("/home/ecal/scratch/sent140-train.pth")
        test = torch.load("/home/ecal/scratch/sent140-test.pth")

    train_dl = DataLoader(train, batch_size=32, pin_memory=True)
    test_dl = DataLoader(test, batch_size=32, pin_memory=True)

    return train_dl, test_dl


def evaluate(model: Transformer, test_dl: DataLoader, criterion):
    model.eval()
    total_loss = 0.0
    preds, all_labels = [], []
    for inputs, mask, labels in test_dl:
        inputs = inputs.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs, mask)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        preds.append(outputs.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return total_loss / len(test_dl), preds, all_labels


if __name__ == "__main__":
    # Load and preprocess data
    print("Loading data...")
    train_dl, test_dl = load_data()

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", type=str, required=True, help="Path to output models")
    parser.add_argument("-l", "--lr", type=float, default=3e-4)
    args = parser.parse_args()

    # Training
    print("Training...")
    model = Transformer().to(device)
    model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in tqdm(range(1, 11)):
        epoch_loss = 0.0
        model.train()
        prog = tqdm(train_dl, desc=f"Epoch {epoch}", leave=False, disable=False)
        for inputs, mask, labels in prog:
            inputs = inputs.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, mask)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            prog.set_postfix(loss=loss.item())

        tqdm.write(f"Epoch {epoch}")

        epoch_loss /= len(train_dl)
        tqdm.write(f"Train Loss: {epoch_loss:.4f}")

        val_loss, _, _ = evaluate(model, test_dl, criterion)
        tqdm.write(f"Val Loss: {val_loss:.4f}")

        torch.save(
            model.state_dict(), f"/home/ecal/scratch/model_sent140_{args.out}_{epoch}.pth"
        )

    # Save the model
    torch.save(model.state_dict(), f"/home/ecal/scratch/model_sent140_{args.out}.pth")
