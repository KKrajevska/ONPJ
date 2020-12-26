from fastai import *
from fastai.text import *
from fastai.text.data import *
from fastai.text.learner import *
from fastai.text.models import *
from fastai.metrics import *
import pandas as pd
from pathlib import Path

path = Path("./")
train_df = pd.read_csv(
    "../jigsaw-toxic-comment-classification-challenge/train.csv/train.csv"
)[:100]

dls = TextDataLoaders.from_df(
    train_df, text_col="comment_text", valid_pct=0.2, is_lm=True, seq_len=100, bs=80
)

# print(dls)
learn = language_model_learner(
    dls, AWD_LSTM, path=path, drop_mult=0.3, metrics=[accuracy]
)

# breakpoint()
# learn.lr_find()
# breakpoint()
# print(learn.recorder)
# learn.recorder.plot(skip_end=15)
learn.fit_one_cycle(1, 1e-2)