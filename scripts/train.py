from DataReader import DataReader
from Model import Model
from TextProcessor import TextProcessor
from Vectorizer import Vectorizer
import numpy as np
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import time
from Word2Vec import Word2VecVectorizer

timestamp = time.time()
value = datetime.datetime.fromtimestamp(timestamp)
USE_W2V = True
FAST = False
PREFIX = f"{value.strftime('%Y-%m-%d %H:%M:%S')}_{'FAST' if FAST else 'SLOW'}_{'LSTM' if not USE_W2V else 'FFCN'}"
EPOCHS = 100


word2vec = Word2VecVectorizer()
text_process = TextProcessor()


def write_results(
    model, best_model, test_data, test_labels, text_process, vectorizer, i, stage="test"
):
    encoded_test_labels = (
        model.one_hot_encoding(test_labels)
        if max(test_labels) > 1
        else np.array(test_labels).reshape(-1, 1)
    )
    if not USE_W2V:
        processed_test_data = text_process.process_text(test_data)
        vect_data_test = vectorizer.vectorize(processed_test_data)
    else:
        vectorizer = word2vec
        vect_data_test = vectorizer.vectorize(test_data)
        vectorizer.vocab_len = 50000
        vectorizer.max_len = 300

    filepath = Path(f"./models/metrics/{PREFIX}/{stage}/models-task-{i}")
    filepath.mkdir(parents=True, exist_ok=True)

    figure_i = i * 2 + (1 if stage == "train" else 0)

    res_file_path = filepath / "metricsFile.txt"
    if i == 2:
        pred_labels = best_model.predict(vect_data_test)
        pred_labels_noh = np.argmax(pred_labels, axis=1)
        encoded_test_labels_noh = np.argmax(encoded_test_labels, axis=1)
        accuracy = accuracy_score(encoded_test_labels_noh, pred_labels_noh)
        classification_report_ = classification_report(
            encoded_test_labels_noh, pred_labels_noh
        )
        roc_auc_list = []
        fprs = []
        tprs = []
        for cnt in range(3):
            roc_auc_list.append(
                roc_auc_score(
                    (encoded_test_labels[:, cnt] > 0.5).astype(np.int),
                    (pred_labels[:, cnt] > 0.5).astype(np.int),
                )
            )
            fpr, tpr, _ = roc_curve(
                (encoded_test_labels[:, cnt] > 0.5).astype(np.int),
                (pred_labels[:, cnt] > 0.5).astype(np.int),
            )
            fprs.append(fpr)
            tprs.append(tpr)

        with open(res_file_path, "w") as f:
            f.write("acc: " + str(accuracy) + "\n")
            f.write(classification_report_ + "\n")
            f.write("roc_auc_0: " + str(roc_auc_list[0]) + "\n")
            f.write("roc_auc_1: " + str(roc_auc_list[1]) + "\n")
            f.write("roc_auc_2: " + str(roc_auc_list[2]) + "\n")

        plt.figure(figure_i)
        plt.plot(fprs[0], tprs[0], label="LSTM class 0" if not USE_W2V else 'FFCN class 0' )
        plt.plot(fprs[1], tprs[1], label="LSTM class 1" if not USE_W2V else 'FFCN class 1')
        plt.plot(fprs[2], tprs[2], label="LSTM class 2" if not USE_W2V else 'FFCN class 2')
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
    else:
        pred_labels = (best_model.predict(vect_data_test) > 0.5).astype(np.int)
        accuracy = accuracy_score(encoded_test_labels, pred_labels)
        classification_report_ = classification_report(encoded_test_labels, pred_labels)
        roc_auc = roc_auc_score(encoded_test_labels, pred_labels)
        with open(res_file_path, "w") as f:
            f.write("acc: " + str(accuracy) + "\n")
            f.write(classification_report_ + "\n")
            f.write("roc_auc: " + str(roc_auc))

        fpr, tpr, _ = roc_curve(encoded_test_labels, pred_labels,)
        plt.figure(figure_i)
        plt.plot(fpr, tpr, label="LSTM" if not USE_W2V else 'FFCN')
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

    plt.title("ROC curve (zoomed in at top left)")
    plt.legend(loc="lower right")
    parent_path = Path(f"./models/plot-AUC/{PREFIX}/{stage}/model-task-{i}")
    parent_path.mkdir(parents=True, exist_ok=True)
    plot_path = parent_path / "roc-curve.png"
    plt.savefig(plot_path)


def main():
    labeled_train_data_path = "../OLIDv1.0/olid-training-v1.0.tsv"
    test_data_paths = [
        "../OLIDv1.0/testset-levela.tsv",
        "../OLIDv1.0/testset-levelb.tsv",
        "../OLIDv1.0/testset-levelc.tsv",
    ]

    test_data_labels = [
        "../OLIDv1.0/labels-levela.csv",
        "../OLIDv1.0/labels-levelb.csv",
        "../OLIDv1.0/labels-levelc.csv",
    ]
    for i in range(3):
        dataReader = DataReader(labeled_train_data_path)
        train_data, labels = dataReader.get_data_and_labels(i)

        if FAST:
            train_data = train_data[:50]
            labels = labels[:50]

        if not USE_W2V:
            tr_data_processed = text_process.process_text(train_data)
            vectorizer = Vectorizer()
            vect_data = vectorizer.vectorize(tr_data_processed)
        else:
            vectorizer = word2vec
            vect_data = vectorizer.vectorize(train_data)
            vectorizer.vocab_len = 50000
            vectorizer.max_len = 300

        model = Model(vect_data, labels, vectorizer.vocab_len, vectorizer.max_len)
        model.train_model(0.001, EPOCHS, 80, str(i), PREFIX, "LSTM" if not USE_W2V else "FFCN")

        best_model = model.load_model(str(i), PREFIX)
        test_data = dataReader.get_test_data(test_data_paths[i])
        test_labels = dataReader.get_test_labels(test_data_labels[i])

        write_results(
            model,
            best_model,
            test_data,
            test_labels,
            text_process,
            vectorizer,
            i,
            stage="test",
        )

        write_results(
            model,
            best_model,
            train_data,
            labels,
            text_process,
            vectorizer,
            i,
            stage="train",
        )

if __name__ == "__main__":
    main()
