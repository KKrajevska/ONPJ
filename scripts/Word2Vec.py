import spacy
import numpy as np

# from joblib import Parallel, delayed

from typing import List, Any
from tqdm import tqdm


class Word2VecVectorizer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")

    def get_doc_embs(self, data):
        samples = self.get_spacy_repr(data)
        emmbedings = []
        for sample in samples:
            doc_emb = None
            for token in sample:
                if doc_emb is None:
                    doc_emb = np.zeros_like(token.vector)
                doc_emb += token.vector / len(sample)

            emmbedings.append(doc_emb)

        return emmbedings

    def get_spacy_repr(self, data: List[str], use_tqdm: bool = True) -> List[Any]:
        # return Parallel(n_jobs=2)(delayed(self.nlp)(sample) for sample in data)
        if use_tqdm:
            return [self.nlp(sample) for sample in tqdm(data)]
        else:
            return [self.nlp(sample) for sample in data]
