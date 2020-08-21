from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class Vectorizer:
    def __init__(self):
        self.vectorizer = None
        self.vocab_len = None

    def vectorize(self, data):
        # vectorizer = TfidfVectorizer(max_features=5000)
        # vectorizer = CountVectorizer(max_features=50000, binary=True)
        self.vocab_len = 50000
        self.max_len = 1500
        merged_data = [" ".join(tweet) for tweet in data]
        if not self.vectorizer:
            self.vectorizer = Tokenizer(num_words=self.vocab_len)
            self.vectorizer.fit_on_texts(merged_data)

        sequences = self.vectorizer.texts_to_sequences(merged_data)
        sequences_matrix = sequence.pad_sequences(sequences, maxlen=self.max_len)
        return sequences_matrix
        # if not self.vectorizer:
        #     self.vectorizer = vectorizer.fit(merged_data)

        # vectors = self.vectorizer.transform(merged_data).toarray()
        # self.vocab_len = len(self.vectorizer.vocabulary_.keys())
        # return vectors
