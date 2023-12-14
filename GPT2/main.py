import os
import argparse
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from copy import deepcopy

from goemotions import GoEmotionsDataset
from sentiment140 import Sentiment140Dataset
from model import Classifier


def pipeline(encoded_input):
    if args.finetune:
        gpt_outputs = model(**encoded_input)['last_hidden_state']
    else:
        with torch.no_grad():
            gpt_outputs = model(**encoded_input)['last_hidden_state']
    gpt_outputs = torch.flatten(gpt_outputs, 1)
    outputs = clf(gpt_outputs)
    return outputs


def eval_loop():
    clf.eval()
    labels = []
    pred = []
    running_loss = 0.0
    for inputs, label in tqdm(val_dataloader):
        encoded_input = tokenizer(inputs, return_tensors='pt', padding=True).to(device)
        outputs = pipeline(encoded_input)

        pred.append(outputs.argmax(1).cpu())
        labels.append(label)

        with torch.no_grad():
            running_loss += criterion(outputs.cpu(), label.cpu()).item()
    
    pred = torch.cat(pred)
    labels = torch.cat(labels)
    running_loss /= len(val_dataloader)

    clf.train()
    return pred, labels, running_loss


parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--longest_sequence", type=int, default=40)
parser.add_argument("--learning_rate", "--lr", type=float, default=0.001)
parser.add_argument("--device", choices=['cpu', 'cuda'])
parser.add_argument("--dataset", choices=['sentiment140', 'goemotions'], required=True)
parser.add_argument("--model", default='gpt2', choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
parser.add_argument("--output_directory", "-o", required=True)
parser.add_argument("--finetune", action='store_true', default=False)
parser.add_argument("--dropout", action='store_true')

args = parser.parse_args()

os.makedirs(args.output_directory, exist_ok=True)

with open(os.path.join(args.output_directory, 'args.txt'), 'w') as fp:
    fp.write(str(vars(args)))

device = args.device
if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

torch.multiprocessing.set_sharing_strategy('file_system')
if args.dataset == 'goemotions':
    train_dataset = GoEmotionsDataset('train')
    val_dataset = GoEmotionsDataset('dev')
else:
    train_dataset = Sentiment140Dataset('train')
    val_dataset = Sentiment140Dataset('dev')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

tokenizer = GPT2Tokenizer.from_pretrained(args.model)
model = GPT2Model.from_pretrained(args.model,)
max_length = min(tokenizer.model_max_length, args.longest_sequence)
embed_dim = model.embed_dim
clf = Classifier(max_length, embed_dim, len(train_dataset.labels), args.dropout)

model = model.to(device)
clf = clf.to(device)

tokenizer.pad_token = tokenizer.eos_token

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(clf.parameters(), lr=args.learning_rate)

metrics = []
best_model = None
best_model_val_acc = -1
for epoch in tqdm(range(args.epochs)):  # loop over the dataset multiple times
    clf.train()
    running_loss = 0.0
    correct = 0
    for i, data in enumerate(tqdm(train_dataloader), 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        encoded_input = tokenizer(inputs, return_tensors='pt', padding=True).to(device)
        labels = labels.to(device)

        outputs = pipeline(encoded_input)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # gather statistics
        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    
    running_loss /= len(train_dataloader)
    print(f'[training loss: {running_loss:.3f} training accuracy: {correct / len(train_dataset):.3f}]')

    pred, label, val_loss = eval_loop()
    accuracy = accuracy_score(pred, label)
    print("accuracy:", accuracy)

    if accuracy > best_model_val_acc:
        best_model_val_acc = accuracy
        best_model = {
            args.model: deepcopy(model.state_dict()),
            "clf": deepcopy(clf.state_dict()),
        }

    metrics.append({
        "train": {
            "loss": running_loss,
            "accuracy": correct / len(train_dataset),
        },
        "val": {
            "loss": val_loss,
            "accuracy": accuracy,
            "precision": precision_score(pred, label, average='macro'),
            "recall": recall_score(pred, label, average='macro'),
            "f1": f1_score(pred, label, average='macro'),
        }
    })

print('Finished Training')

# saving models
torch.save(best_model, os.path.join(args.output_directory, 'best_model.pkl'))

# saving metrics
with open(os.path.join(args.output_directory, 'metrics.json'), 'w') as fp:
    json.dump(metrics, fp, indent=4)

epochs_x = list(range(1, len(metrics) + 1))
plt.plot(epochs_x, [e["train"]["loss"] for e in metrics], label="train")
plt.plot(epochs_x, [e["val"]["loss"] for e in metrics], label="validation")
plt.title("Loss")
plt.legend()
plt.savefig(os.path.join(args.output_directory, 'loss.png'))
plt.close()

plt.plot(epochs_x, [e["train"]["accuracy"] for e in metrics], label="train")
plt.plot(epochs_x, [e["val"]["accuracy"] for e in metrics], label="validation")
plt.title("Accuracy")
plt.legend()
plt.savefig(os.path.join(args.output_directory, 'accuracy.png'))
plt.close()

cm = confusion_matrix(label, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=val_dataset.labels)
disp.plot()
plt.savefig(os.path.join(args.output_directory, 'cm.png'))
plt.close()
