import pandas as pd
from fastai.text.all import *


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
    learn_classifier = learn_classifier.load_encoder("lm_encoder_final")
    learn_classifier.fit_one_cycle(1, 2e-2)
    learn_classifier.show_results(max_n=6, figsize=(7, 8), return_fig=True)
    learn_classifier.save("ULMFIT_classifier")


if __name__ == "__main__":
    train_df = data_read(
        "../jigsaw-toxic-comment-classification-challenge/train.csv/train.csv"
    )
    test_df = data_read(
        "../jigsaw-toxic-comment-classification-challenge/test.csv/test.csv"
    )

    ULMFIT_encoder_training(train_df, test_df)
    # ULMFIT_classifier(train_df)
