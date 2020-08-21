from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout,
    Input,
    SpatialDropout1D,
    Bidirectional,
    # Flatten,
)
from tensorflow.keras.metrics import (
    CategoricalAccuracy,
    BinaryAccuracy,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from pathlib import Path
import numpy as np
import os


class Model:
    def __init__(self, data, labels, vocab_len, max_len):
        self.data = data
        self.labels = labels
        self.vocab_len = vocab_len
        self.max_len = max_len
        self.encoder = None
        # self.input_len = self.find_max_len()
        self.encoded_labels = (
            self.one_hot_encoding(self.labels)
            if max(self.labels) > 1
            else np.array(self.labels).reshape(-1, 1)
        )
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
        return data
        # return pad_sequences(data, maxlen=1500, padding="post", dtype="float32")

    def construct_LSTM_Model(self, lr):
        # model = Sequential()
        # model.add(Input(shape=(self.vocab_len,)))
        # model.add(Dense(512, activation="relu"))
        # model.add(Dense(256, activation="relu"))
        # model.add(Dense(128, activation="relu"))
        # model.add(Dense(self.encoded_labels.shape[1], activation="softmax"))
        # adam = Adam(lr=lr)
        # model.compile(loss=CategoricalCrossentropy(from_logits=False), optimizer=adam, metrics=[CategoricalAccuracy(), Precision(), Recall(), AUC()])
        # return model

        model = Sequential()
        model.add(Input(shape=(self.max_len,)))
        model.add(Embedding(self.vocab_len, 128, input_length=self.max_len))
        model.add(SpatialDropout1D(0.2))
        model.add(Bidirectional(LSTM(128, activation="tanh", return_sequences=True)))
        model.add(Bidirectional(LSTM(128, activation="tanh", return_sequences=False)))
        # if max(self.labels) > 1:
        #     model.add(Flatten())
        model.add(Dense(64, activation="elu"))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation="elu"))
        model.add(Dropout(0.5))
        model.add(Dense(16, activation="elu"))
        model.add(Dropout(0.5))
        model.add(
            Dense(
                self.encoded_labels.shape[1] if max(self.labels) > 1 else 1,
                activation="softmax" if max(self.labels) > 1 else "sigmoid",
            )
        )
        print("##########################", max(self.labels) > 1)
        # model.add(Dense(self.encoded_labels.shape[1]))
        adam = Adam(lr=lr)
        # model.compile(
        #     loss="binary_crossentropy"
        #     if self.encoded_labels.shape[1] == 2
        #     else "categorical_crossentropy",
        #     optimizer=adam,
        #     metrics=[CategoricalAccuracy(), Precision(), Recall(), AUC()],
        # )
        loss = (
            CategoricalCrossentropy() if max(self.labels) > 1 else BinaryCrossentropy()
        )
        accuracy = (
            CategoricalAccuracy(name="acc")
            if max(self.labels) > 1
            else BinaryAccuracy(name="acc")
        )
        model.compile(
            loss=loss,
            optimizer=adam,
            metrics=[accuracy]
            # metrics=[BinaryAccuracy(), Precision(thresholds=0.5), Recall(thresholds=0.5), AUC()],
        )
        return model

    def train_model(self, lr, epochs, batch_size, task_type, prefix):
        filepath = Path(f"./models/{prefix}/models-task-{task_type}")
        filepath.mkdir(parents=True, exist_ok=True)
        my_callbacks = [
            ModelCheckpoint(
                filepath="./models/"
                + prefix
                + "/models-task-"
                + task_type
                + "/model.{epoch:02d}-{val_acc:.5f}-{val_loss:.5f}.h5",
                verbose=10,
            ),
            EarlyStopping(patience=20),
            TensorBoard(log_dir="./logs/" + prefix + "/" + task_type),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=10e-5, verbose=1
            ),
        ]
        self.model = self.construct_LSTM_Model(lr)
        self.model.fit(
            self.train_data,
            self.encoded_labels,
            batch_size=batch_size,
            validation_split=0.2,
            epochs=epochs,
            callbacks=my_callbacks,
            # class_weight={0: 0.3, 1: 1.0},
        )

    def load_model(self, i, prefix):
        path = f"./models/{prefix}/models-task-{i}"
        best_model = ""
        min_val_los = 1000.0
        max_val_acc = 0.0
        for model_file in os.listdir(path):
            [acc, loss] = map(float, model_file.replace(".h5", "").split("-")[1:])
            # start = model_file.find("-") + 1
            # end = model_file.find(".h5")

            # val_los = float(model_file[start:end])
            if acc > max_val_acc or (acc == max_val_acc and loss < min_val_los):
                min_val_los = loss
                max_val_acc = acc
                best_model = model_file
            # if val_los > min_val_los:
            #     best_model = model_file
            #     min_val_los = val_los

        print(min_val_los, best_model)
        loaded = load_model(path + "/" + best_model)
        return loaded
