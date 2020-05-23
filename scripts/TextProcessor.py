from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
import string


class TextProcessor:
    def __init__(self):
        pass

    def tokenize(self, data):
        tokenized_data = []
        for tweet in data:
            tokenized_data.append(word_tokenize(tweet.lower()))

        return tokenized_data

    def get_wordnet_pos(self, word):
        tag = pos_tag(word)[0][1]
        if tag.startswith("J"):
            return wordnet.ADJ
        elif tag.startswith("V"):
            return wordnet.VERB
        elif tag.startswith("N"):
            return wordnet.NOUN
        else:
            return wordnet.ADV

    def lemmatize(self, clean_data):
        wnl = WordNetLemmatizer()
        new_data = []
        for i, tweet in enumerate(clean_data):
            new_data.append([])
            for word in tweet:
                new_data[i].append(wnl.lemmatize(word, pos=self.get_wordnet_pos(word)))

        return new_data

    def remove_stopwords(self, tokenized_data):
        stop = stopwords.words("english") + list(string.punctuation) + ["user"]
        clean_data = []
        for tweet in tokenized_data:
            clean_data.append([w for w in tweet if w not in stop])

        return clean_data

    def process_text(self, data, stopwords=True):
        tokenized_data = self.tokenize(data)
        if stopwords == True:
            clean_data = self.remove_stopwords(tokenized_data)
            processed_data = self.lemmatize(clean_data)
        else:
            processed_data = self.lemmatize(tokenized_data)

        return processed_data

