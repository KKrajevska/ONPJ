from sklearn.preprocessing import OneHotEncoder
import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.metrics import Accuracy, Precision, Recall, AUC
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
import numpy as np


class Model:
    def __init__(self, data, labels, vocab_len):
        self.data = data
        self.labels = labels
        self.vocab_len = vocab_len
        self.encoder = None
        self.input_len = self.find_max_len()
        self.encoded_labels = self.one_hot_encoding(self.labels)
        self.train_data = self.encode_data(self.data)

    def find_max_len(self):
        max_value = 0
        for i in self.data:
            if len(i) > max_value:
                max_value = len(i)

        return max_value

    def one_hot_encoding(self, labels):
        encoder = OneHotEncoder()
        return encoder.fit_transform(np.array(labels).reshape(-1, 1)).toarray()

    def encode_data(self, data):
        return pad_sequences(data, maxlen=self.input_len, padding="post")

    def construct_LSTM_Model(self, lr):
        model = Sequential()
        model.add(Embedding(self.vocab_len, 64, input_length=self.input_len))
        model.add(LSTM(200))
        model.add(Dense(self.encoded_labels.shape[1], activation="softmax"))
        adam = Adam(lr=lr)
        model.compile(
            loss="binary_crossentropy"
            if self.encoded_labels.shape[1] == 2
            else "categorical_crossntropy",
            optimizer=adam,
            metrics=[Accuracy(), Precision(), Recall(), AUC()],
        )
        return model

    def train_model(self, lr, epochs, batch_size, task_type):
        my_callbacks = [
            EarlyStopping(patience=10),
            ModelCheckpoint(
                filepath="./models-task-"
                + task_type
                + "/model.{epoch:02d}-{val_loss:.2f}.h5"
            ),
            TensorBoard(log_dir="./logs"),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=10e-5),
        ]
        model = self.construct_LSTM_Model(lr)
        model.fit(
            self.train_data,
            self.encoded_labels,
            batch_size=batch_size,
            validation_split=0.3,
            epochs=epochs,
            callbacks=my_callbacks,
        )
