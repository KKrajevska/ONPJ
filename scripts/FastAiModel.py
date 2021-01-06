import pandas as pd
from fastai.text.all import *

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import numpy as np


def data_read(path):
    df = pd.read_csv(path)
    return df


def ULMFIT_encoder_training(train_df, test_df):
    label_cols = list(train_df.columns[2:])
    lm_df = train_df[["id", "comment_text"]].append(test_df)

    dls_lm = TextDataLoaders.from_df(
        lm_df, text_col="comment_text", valid_pct=0.2, is_lm=True, seq_len=72, bs=64
    )

    learn = language_model_learner(
        dls_lm, AWD_LSTM, drop_mult=0.3, metrics=[accuracy, Perplexity()]
    )

    learn.fit_one_cycle(1, 2e-2)
    learn.recorder.plot_loss()
    plt.savefig("metric_first_encoder.png")
    learn.save_encoder("ULMFIT_lm_encoder_first")
    learn.unfreeze()
    learn.fit_one_cycle(4, 1e-3)
    learn.recorder.plot_loss()
    plt.savefig("metrics_final_encoder.png")
    learn.save_encoder("ULMFIT_lm_encoder_final")


def get_label(row, labels_list):
    idxs = np.where(row == 1)[0]
    if len(idxs) == 0:
        return "okay"
    return ";".join(labels_list[val] for val in idxs)


def ULMFIT_classifier(train_df):
    labels_list = list(train_df.columns[2:])
    labels = train_df[labels_list].apply(
        lambda row: get_label(row, labels_list), axis=1
    )
    train_df["Label"] = labels

    data_blocks = DataBlock(
        blocks=(
            TextBlock.from_df(text_cols="comment_text", seq_len=128),
            MultiCategoryBlock,
        ),
        get_x=ColReader(cols="text"),
        get_y=ColReader(cols="Label", label_delim=";"),
        splitter=TrainTestSplitter(test_size=0.2, random_state=21),
    )

    data_classifier = data_blocks.dataloaders(train_df, bs=64, seed=20)
    learn_classifier = text_classifier_learner(
        data_classifier, AWD_LSTM, drop_mult=0.5, metrics=accuracy_multi
    )
    learn_classifier = learn_classifier.load_encoder("ULMFIT_lm_encoder_final")
    learn_classifier.fit_one_cycle(1, 2e-2)
    learn_classifier.freeze_to(-2)
    learn_classifier.fit_one_cycle(1, slice(1e-2 / (2.6 ** 4), 1e-2))
    learn_classifier.freeze_to(-3)
    learn_classifier.fit_one_cycle(1, slice(5e-3 / (2.6 ** 4), 5e-3))
    learn_classifier.unfreeze()
    learn_classifier.fit_one_cycle(1, slice(1e-3 / (2.6 ** 4), 1e-3))
    learn_classifier.recorder.plot_loss()
    plt.savefig("metrics_classifier.png")
    learn_classifier.export("ULMFIT_classifier.pkl")


def evaluate(test_df, labels_df):
    learn = load_learner("ULMFIT_classifier.pkl", cpu=False)
    tokenized_df = tokenize_df(test_df, "comment_text")
    test_data_classifier = learn.dls.test_dl(tokenized_df[0])
    all_predictions = learn.get_preds(dl=test_data_classifier, reorder=False)
    probs = all_predictions[0].numpy()

    probs = np.where(probs > 0.5, 1, 0)

    # print(len(probs[0]))
    # print(len(labels_df[0]))

    # labels_df_flat = labels_df.flatten()
    # probs_flat = probs.flatten()
    # print(labels_df[0])
    accuracy = accuracy_score(labels_df, probs)
    classification_report_ = classification_report(labels_df, probs)
    roc_auc_score_ = roc_auc_score(labels_df, probs, average=None)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(6):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
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
    plt.savefig("ULMFIT_ROC_CURVE.png")

    # roc_auc_list = []
    # fprs = []
    # tprs = []
    # for cnt in range(7):
    #     roc_auc_list.append(
    #         roc_auc_score(
    #             labels_df[:, cnt],
    #             probs[:, cnt],
    #         )
    #     )
    #     fpr, tpr, _ = roc_curve(
    #         labels_df[:, cnt],
    #         probs[:, cnt],
    #     )
    #     fprs.append(fpr)
    #     tprs.append(tpr)

    with open("ULMFIT_report.txt", "w") as f:
        f.write("acc: " + str(accuracy) + "\n")
        f.write(classification_report_ + "\n")
        f.write("roc_auc_0: " + str(roc_auc_score_) + "\n")
        # f.write("roc_auc_1: " + str(roc_auc_list[1]) + "\n")
        # f.write("roc_auc_2: " + str(roc_auc_list[2]) + "\n")
        # f.write("roc_auc_3: " + str(roc_auc_list[3]) + "\n")
        # f.write("roc_auc_4: " + str(roc_auc_list[4]) + "\n")
        # f.write("roc_auc_5: " + str(roc_auc_list[5]) + "\n")
        # f.write("roc_auc_6: " + str(roc_auc_list[6]) + "\n")

    # plt.figure()
    # plt.plot(fprs[0], tprs[0], label="ULMFIT class 0")
    # plt.plot(fprs[1], tprs[1], label="ULMFIT class 1")
    # plt.plot(fprs[2], tprs[2], label="ULMFIT class 2")
    # plt.plot(fprs[3], tprs[3], label="ULMFIT class 3")

    # plt.plot(fprs[4], tprs[4], label="ULMFIT class 4")
    # plt.plot(fprs[5], tprs[5], label="ULMFIT class 5")
    # plt.plot(fprs[6], tprs[6], label="ULMFIT class 6")

    # plt.plot([0, 1], [0, 1], "k--")
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.title("ROC curve (zoomed in at top left)")
    # plt.legend(loc="lower right")
    # plt.savefig("ULMFIT_ROC_CURVE.png")


def append_okay(row, labels_list):
    if (
        row[labels_list[0]]
        == 0 & row[labels_list[1]]
        == 0 & row[labels_list[2]]
        == 0 & row[labels_list[3]]
        == 0 & row[labels_list[4]]
        == 0 & row[labels_list[5]]
        == 0
    ):
        return 1
    return 0


if __name__ == "__main__":
    train_df = data_read(
        "../jigsaw-toxic-comment-classification-challenge/train.csv/train.csv"
    )
    test_df = data_read(
        "../jigsaw-toxic-comment-classification-challenge/test.csv/test.csv"
    )

    test_labels_df = data_read(
        "../jigsaw-toxic-comment-classification-challenge/test_labels.csv/test_labels.csv"
    )

    labels_list = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    ids_not_to_use = list(test_labels_df[test_labels_df["toxic"] == -1]["id"])
    test_df_clean = test_df[~test_df["id"].isin(ids_not_to_use)]
    test_labels_clean = test_labels_df[~test_labels_df["id"].isin(ids_not_to_use)]
    test_labels_clean["okay"] = test_labels_clean.apply(
        lambda row: append_okay(row, labels_list), axis=1
    )
    labels_list.append("okay")

    df = pd.DataFrame(test_labels_clean, columns=labels_list)
    test_labels = df.to_numpy()

    # print(test_df_clean)
    # ULMFIT_encoder_training(train_df, test_df)

    # ULMFIT_classifier(train_df)
    evaluate(test_df_clean, test_labels)
