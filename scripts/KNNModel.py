from Word2Vec import Word2VecVectorizer
from DataReader import DataReader
from typing import List, Dict
from tqdm import tqdm

from joblib import dump, load
from pathlib import Path

from sklearn import metrics
import numpy as np
import pprint
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter(indent=4)

labeled_train_data_path = "../OLIDv1.0/olid-training-v1.0.tsv"

FAST = False


class KNNModel:
    def __init__(self, max_K: int = 100):
        self.w2v = Word2VecVectorizer()
        self.embeddings = []
        self.labels = []
        self.K = 5
        self.max_K = max_K

    def train_model(self, data: List[str], labels: List[int]) -> int:
        self.embeddings = self.w2v.get_spacy_repr(data)
        self.labels = labels

        model = {"embeddings": self.embeddings, "lables": self.labels}
        dump(model, "KNN_model.joblib")
        self.load_model()

        # Ks = [0 for x in range(self.max_K)]
        # for emb, target_label in zip(tqdm(self.embeddings), self.labels):
        #     votes = {}
        #     scores = sorted(
        #         [
        #             (emb.similarity(cmp_emb), idx)
        #             for idx, cmp_emb in enumerate(self.embeddings)
        #         ]
        #     )[1:]
        #     for k in range(self.max_K):
        #         label = self.labels[scores[k][1]]
        #         if label not in votes:
        #             votes[label] = 0

        #         votes[label] += 1
        #         curr_best_label = max(zip(votes.values(), votes.keys()))[1]
        #         if curr_best_label == target_label:
        #             Ks[k] += 1

        # self.K = max(zip(Ks, range(self.max_K)))[1] + 1
        # model = {"embeddings": self.embeddings, "lables": self.labels, "K": self.K}
        # dump(model, "KNN_model.joblib")

    def load_model(self) -> None:
        model = load("KNN_model.joblib")
        self.embeddings = model["embeddings"]
        self.labels = model["lables"]
        self.K = 150

    def clasify_sample(self, sample: str):
        spacy_sample = self.w2v.get_spacy_repr([sample], use_tqdm=False)[0]
        scores = sorted(
            [
                (spacy_sample.similarity(emb), idx)
                for idx, emb in enumerate(self.embeddings)
            ]
        )
        votes = {}
        for i in range(self.K):
            label = self.labels[scores[i][1]]
            if label not in votes:
                votes[label] = 0

            votes[label] += 1

        return max(zip(votes.values(), votes.keys()))[1]

    def classify(self, data: List[str]) -> List[int]:
        print("Classifying...")
        return [self.clasify_sample(sample) for sample in tqdm(data)]

    def calc_metrics(
        self, preds: List[int], targets: List[int], stage: str, task_type: int
    ) -> Dict[str, float]:
        preds_np = np.array(preds)
        targets_np = np.array(targets)
        if max(targets) == 1:
            met = {
                "accuracy": metrics.accuracy_score(targets_np, preds_np),
                "clf_report": metrics.classification_report(targets_np, preds_np),
                "roc_auc": metrics.roc_auc_score(targets_np, preds_np),
            }
        else:
            met = {
                f"roc_auc{x}": metrics.roc_auc_score(
                    (targets_np == x).astype(np.uint8), (preds_np == x).astype(np.uint8)
                )
                for x in range(3)
            }
            met["accuracy"] = metrics.accuracy_score(targets_np, preds_np)
            met["clf_report"] = metrics.classification_report(targets_np, preds_np)

        met_path = Path("./KNNMetrics/") / stage / str(task_type)
        met_path.mkdir(parents=True, exist_ok=True)
        with open(met_path / "metrics.txt", "w") as f:
            pprint.pprint(met, f)

        return met

    def get_roc(
        self, preds: List[int], targets: List[int], stage: str, task_type: int
    ) -> None:
        preds_np = np.array(preds)
        targets_np = np.array(targets)
        roc_path = Path("./KNNMetrics/") / stage / str(task_type)
        roc_path.mkdir(parents=True, exist_ok=True)
        figure_i = i * 2 + (1 if stage == "train" else 0)
        if task_type == 2:
            fprs = []
            tprs = []
            for x in range(3):
                fpr, tpr, _ = metrics.roc_curve(
                    (targets_np == x).astype(np.uint8), (preds_np == x).astype(np.uint8)
                )
                fprs.append(fpr)
                tprs.append(tpr)

            plt.figure(figure_i)
            print(fprs)
            print(tprs)
            plt.plot(fprs[0], tprs[0], label="KNN class 0")
            plt.plot(fprs[1], tprs[1], label="KNN class 1")
            plt.plot(fprs[2], tprs[2], label="KNN class 2")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
        else:
            fpr, tpr, _ = metrics.roc_curve(targets_np, preds_np)
            print(fpr)
            print(tpr)
            plt.figure(figure_i)
            plt.plot(fpr, tpr, label="KNN")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])

        plt.title("ROC curve (zoomed in at top left)")
        plt.legend(loc="lower right")
        plt.savefig(roc_path / "roc.png")


if __name__ == "__main__":
    for i in range(3):
        dataReader = DataReader(labeled_train_data_path)
        train_data, train_labels = dataReader.get_data_and_labels(i)
        print("Reading is done!")
        model = KNNModel()
        print("Model is initialized")
        if FAST:
            train_data = train_data[:200]
            train_labels = train_labels[:200]

        model.train_model(train_data, train_labels)
        preds = model.classify(train_data)
        cmetrics = model.calc_metrics(preds, train_labels, stage="train", task_type=i)
        model.get_roc(preds, train_labels, stage="train", task_type=i)
        print("Train metrics:")
        pp.pprint(cmetrics)
        letter = "a"
        if i == 1:
            letter = "b"
        elif i == 2:
            letter = "c"
        test_data_path = f"../OLIDv1.0/testset-level{letter}.tsv"
        test_labels_path = f"../OLIDv1.0/labels-level{letter}.csv"
        test_data = dataReader.get_test_data(test_data_path)
        test_labels = dataReader.get_test_labels(test_labels_path)

        if FAST:
            test_data = test_data[:200]
            test_labels = test_labels[:200]

        preds = model.classify(test_data)
        cmetrics = model.calc_metrics(preds, test_labels, stage="test", task_type=i)
        model.get_roc(preds, test_labels, stage="test", task_type=i)
        print("Test metrics:")
        pp.pprint(cmetrics)
