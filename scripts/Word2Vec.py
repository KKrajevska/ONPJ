import spacy
import numpy as np


class Word2VecVectorizer:

    def __init__(self):
        self.nlp=spacy.load("en_core_web_sm")

    def word2vec_vectorize(self,data):
        emmbedings=[]
        for sample in data:
            tokens=nlp(sample)
            doc_emb=None
            for token in tokens:
                if doc_emb is None:
                   doc_emb = np.zeros_like(token.vector)
                doc_emb+=token.vector/len(tokens)

            emmbedings.append(doc_emb)

        return emmbedings
            



       


