import numpy as np
import pandas as pd
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json


class TrainDataset(Dataset):
    def __init__(self, sentences, scores):
        self.sentences = sentences
        self.scores = scores

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.scores[idx]


class TestDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]


def get_data(path):
    data = pd.read_csv(path, sep="\t").sample(frac=0.1, replace=True, random_state=1)
    sentences = data["GENERIC SENTENCE"].apply(lambda x: x.lower())
    scores = data["SCORE"]
    return sentences, scores


def generate_tokens(sentences):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    sentence_tokens = []
    for sentence in sentences:
        tokens = tokenizer.encode(sentence, padding="max_length", truncation=True, max_length=24, add_special_tokens=True)
        sentence_tokens.append(tokens)
    return torch.tensor(sentence_tokens, dtype=torch.float)


def generate_train_dataloader():
    sentences, scores = get_data("data/GenericsKB.tsv")
    sentences_tensor = generate_tokens(sentences)
    scores_tensor = torch.tensor(scores.to_list(), dtype=torch.float)
    dataset = TrainDataset(sentences_tensor, scores_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=3)
    return dataloader


def generate_test_dataloader(captions, labels):
    dataset = TestDataset(captions, labels)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=3)
    return dataloader


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(24, 128, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1, bias=True),
            torch.nn.Sigmoid()
        )
    
    def forward(self, X):
        return self.model(X)


def train(model, dataloader, criterion, optimizer):
    model.train()
    num_correct, total_loss = 0, 0
    for i, (sentence, label) in enumerate(dataloader):
        optimizer.zero_grad()
        scores = model.forward(sentence)
        scores = torch.reshape(scores, (scores.size(dim=0),))
        loss = criterion(scores, label)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        num_correct += int(torch.sum(torch.where(scores >= 0.5, torch.tensor(1), torch.tensor(0)) == torch.where(label >= 0.5, torch.tensor(1), torch.tensor(0))).item())
        del sentence, label, loss
    total_acc = (100 * num_correct) / (32 * len(dataloader))
    total_loss = float(total_loss / len(dataloader))
    return total_acc, total_loss


def test(model, dataloader):
    model.eval()
    for i, (caption, labels) in enumerate(dataloader):
        with torch.inference_mode():
            output = model(caption)
            output = torch.reshape(output, (output.size(dim=0),))
    return int(torch.sum(torch.where(output >= 0.5, torch.tensor(1), torch.tensor(0)) == labels))
    


if __name__=="__main__":
    train_dataloader = generate_train_dataloader()
    model = Network()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
    for epoch in range(10):
        train_acc, train_loss = train(model, train_dataloader, criterion, optimizer)
        print("\nEpoch {}/{}: \nTrain Acc: {:.04f}%\t Train Loss: {:.04f}".format(
        epoch + 1, 10, train_acc, train_loss))
        scheduler.step()
    caption_files = os.listdir("data/caption_jsons")
    results = {}
    for file in caption_files:
        path = os.path.join("data/caption_jsons", file)
        with open(path, "r") as f:
            data = json.load(f)
        total = 0
        captions, labels = [], []
        for key in data.keys():
            captions.append(data[key]["caption"])
            captions.append(data[key]["negative_caption"])
            total += 1
            labels.append(1)
            labels.append(0)
        captions_tokens = generate_tokens(captions)
        labels = torch.tensor(labels, dtype=torch.int)
        test_dataloader = generate_test_dataloader(captions_tokens, labels)
        num_correct = test(model, test_dataloader)
        results[file.split(".")[0]] = num_correct / total
        print(results[file.split(".")[0]])
    json_res = json.dumps(results)
    with open("Unimodal_text_only_baseline.json", "w") as outfile:
        outfile.write(json_res)
    