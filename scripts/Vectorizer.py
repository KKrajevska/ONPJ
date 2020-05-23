from sklearn.feature_extraction.text import TfidfVectorizer

class Vectorizer:
    def __init__(self):
        self.vectorizer=None
        self.vocab_len=None
    
    def tf_idf_vectorize(self, data):
        vectorizer = TfidfVectorizer()
        merged_data = [' '.join(tweet) for tweet in data]
        if not self.vectorizer:
            self.vectorizer = vectorizer.fit(merged_data)

        vectors = self.vectorizer.transform(merged_data).toarray()
        self.vocab_len = len(self.vectorizer.vocabulary_.keys())
        return vectors