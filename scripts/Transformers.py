import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import (
    BertTokenizer,
    BertModel,
    BertConfig,
    AlbertTokenizer,
    AlbertModel,
    SqueezeBertTokenizer,
    SqueezeBertModel,
)
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from itertools import cycle
import matplotlib.pyplot as plt

from torch import cuda

device = "cuda" if cuda.is_available() else "cpu"

df = pd.read_csv("../jigsaw-toxic-comment-classification-challenge/train.csv/train.csv")
# df = df[:10]
df["list"] = df[df.columns[2:]].values.tolist()
new_df = df[["comment_text", "list"]].copy()

MAX_LEN = 200
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-05
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
tokenizer = SqueezeBertTokenizer.from_pretrained(
    "squeezebert/squeezebert-mnli-headless"
)


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment_text
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.targets[index], dtype=torch.float),
        }


train_size = 0.8
train_dataset = new_df.sample(frac=train_size, random_state=200)
test_dataset = new_df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)
training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

train_params = {"batch_size": TRAIN_BATCH_SIZE, "shuffle": True, "num_workers": 0}

test_params = {"batch_size": VALID_BATCH_SIZE, "shuffle": True, "num_workers": 0}

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 6)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


class ALBERTClass(torch.nn.Module):
    def __init__(self):
        super(ALBERTClass, self).__init__()
        self.l1 = AlbertModel.from_pretrained("albert-base-v2")
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 6)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


class SqueezeBERTClass(torch.nn.Module):
    def __init__(self):
        super(SqueezeBERTClass, self).__init__()
        self.l1 = SqueezeBertModel.from_pretrained(
            "squeezebert/squeezebert-mnli-headless"
        )
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 6)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


# model = BERTClass()
# model = ALBERTClass()
model = SqueezeBERTClass()
model.to(device)


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


def train(epoch):
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data["ids"].to(device, dtype=torch.long)
        mask = data["mask"].to(device, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
        targets = data["targets"].to(device, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _ % 5000 == 0:
            print(f"Epoch: {epoch}, Loss:  {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


for epoch in range(EPOCHS):
    train(epoch)
    torch.save(model, f"transformer_model_squeeze-{epoch}.pth")


def validation(epoch):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


for epoch in range(EPOCHS):
    outputs, targets = validation(epoch)
    outputs = np.array(outputs) >= 0.5
    outputs = np.array(outputs).astype(int)
    targets = np.array(targets).astype(int)
    accuracy = accuracy_score(targets, outputs)
    classification_report_ = classification_report(targets, outputs)
    roc_auc_score_ = roc_auc_score(targets, outputs, average=None)
    # breakpoint()
    with open("Transformers_report-squeeze.txt", "w") as f:
        f.write("acc: " + str(accuracy) + "\n")
        f.write(classification_report_ + "\n")
        f.write("roc_auc_0: " + str(roc_auc_score_) + "\n")
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(6):
        fpr[i], tpr[i], _ = roc_curve(targets[:, i], outputs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(6), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label="ROC curve of class {0} (area = {1:0.2f})" "".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of Receiver operating characteristic to multi-class")
    plt.legend(loc="lower right")
    plt.savefig("Transformers_curve-squeeze.png")
