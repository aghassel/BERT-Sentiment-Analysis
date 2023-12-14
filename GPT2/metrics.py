import os
import argparse
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

from goemotions import GoEmotionsDataset
from sentiment140 import Sentiment140Dataset
from model import Classifier


@torch.no_grad()
def pipeline(encoded_input):
    gpt_outputs = model(**encoded_input)['last_hidden_state']
    gpt_outputs = torch.flatten(gpt_outputs, 1)
    outputs = clf(gpt_outputs)
    return outputs


def eval_loop(dataloader):
    labels = []
    pred = []
    running_loss = 0.0
    for inputs, label in tqdm(dataloader):
        encoded_input = tokenizer(inputs, return_tensors='pt', padding=True).to(device)
        outputs = pipeline(encoded_input)

        pred.append(outputs.cpu())
        labels.append(label)

        with torch.no_grad():
            running_loss += criterion(outputs.cpu(), label.cpu()).item()
    
    pred = torch.cat(pred)
    labels = torch.cat(labels)
    running_loss /= len(test_dataloader)

    pred_labels = pred.argmax(1)

    return pred, pred_labels, labels, running_loss


parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--longest_sequence", type=int, default=40)
parser.add_argument("--device", choices=['cpu', 'cuda'])
parser.add_argument("--dataset", choices=['sentiment140', 'goemotions'], required=True)
parser.add_argument("--model", default='gpt2', choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
parser.add_argument("--output_directory", "-o", required=True)
parser.add_argument("--dropout", action='store_true')

args = parser.parse_args()

device = args.device
if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

torch.multiprocessing.set_sharing_strategy('file_system')
if args.dataset == 'goemotions':
    test_dataset = GoEmotionsDataset('test')
else:
    test_dataset = Sentiment140Dataset('test')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

tokenizer = GPT2Tokenizer.from_pretrained(args.model)
model = GPT2Model.from_pretrained(args.model,)
max_length = min(tokenizer.model_max_length, args.longest_sequence)
embed_dim = model.embed_dim
clf = Classifier(max_length, embed_dim, len(test_dataset.labels), args.dropout)
clf.eval()

best_model_dict = torch.load(os.path.join(args.output_directory, 'best_model.pkl'))
model.load_state_dict(best_model_dict[args.model])
clf.load_state_dict(best_model_dict['clf'])

model = model.to(device)
clf = clf.to(device)

tokenizer.pad_token = tokenizer.eos_token

criterion = nn.CrossEntropyLoss()

pred, pred_labels, label, test_loss = eval_loop(test_dataloader)

# Classic Metrics
accuracy = accuracy_score(pred_labels, label)
precision, recall, f1, _ = precision_recall_fscore_support(pred_labels, label, average='macro')

print('Test Loss: {:.4f}'.format(test_loss))
print('Accuracy: {:.4f}'.format(accuracy))
print('Precision: {:.4f}'.format(precision))
print('Recall: {:.4f}'.format(recall))
print('F1 Score: {:.4f}'.format(f1))

print("\nClassification Report:\n")
print(classification_report(label, pred_labels, target_names=test_dataset.labels))

# ROC Curve
if len(test_dataset.labels) <= 2:
    roc_auc = roc_auc_score(label, pred[:, 1])
    fpr, tpr, _ = roc_curve(label, pred[:, 1])

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.output_directory, 'roc_test.png'))
    plt.close()

    # Confustion Matrix
    cm = confusion_matrix(label, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.labels)
    disp.plot()
    plt.savefig(os.path.join(args.output_directory, 'cm_test.png'))
    plt.close()

# Radar Chart
labels=np.array(['Accuracy', 'Precision', 'Recall', 'F1'])
stats=np.array([accuracy, precision, recall, f1])

angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
stats=np.concatenate((stats,[stats[0]]))
angles+=angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, stats, color='red', alpha=0.25)
ax.plot(angles, stats, color='red', linewidth=2)
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

plt.title('Model Performance Radar Chart', size=20, color='red', y=1.1)
plt.savefig(os.path.join(args.output_directory, 'radar_test.png'))
plt.close()

# print(f"{'Emotion':<20} {'ROC-AUC':<10} {'PR-AUC':<10}")
# print("-" * 40)

# for i, label in enumerate(test_dataset.labels):
#     roc_auc = roc_auc_score(true_labels[:, i], pred[:, i])
#     pr_auc = average_precision_score(true_labels[:, i], pred[:, i])
#     print(f"{label:<20} {roc_auc:.4f}     {pr_auc:.4f}")
