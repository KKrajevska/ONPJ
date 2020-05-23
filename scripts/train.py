from DataReader import DataReader
from Model import Model
from TextProcessor import TextProcessor
from Vectorizer import Vectorizer

def main():
    labeled_train_data_path = "../OLIDv1.0/olid-training-v1.0.tsv"
    for i in range(3):
        dataReader=DataReader(labeled_train_data_path)
        train_data, labels=dataReader.get_train_data_and_labels(i)
        text_process=TextProcessor()
        tr_data_processed=text_process.process_text(train_data)
        vectorizer=Vectorizer()
        vect_data=vectorizer.tf_idf_vectorize(tr_data_processed)
        model=Model(vect_data, labels, vectorizer.vocab_len)
        model.train_model(0.001, 1000, 64)

if __name__ == "__main__":
    main()