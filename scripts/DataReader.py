import csv


class DataReader:
    def __init__(self, path):
        self.train_data_labels = path
        self.map_labels = {
            0: {0: [0], 1: [1, 2, 3, 4]},
            1: {0: [1], 1: [2, 3, 4]},
            2: {0: [2], 1: [3], 2: [4]},
        }

        for problem_id, labels in self.map_labels.items():
            for target_label, origin_labels in labels.items():
                labels[target_label] = set(origin_labels)

    def get_train_data_and_labels(self, problem_id):
        data = []
        labels = []
        with open(self.train_data_labels, encoding="utf8") as tf:
            reader = csv.reader(tf, delimiter="\t")
            for i, line in enumerate(reader):
                if i == 0:
                    continue

                label = line[-3:]
                label_encoded = self.label_encoder(label)
                for target_label, origin_labels in self.map_labels[problem_id].items():
                    if label_encoded in origin_labels:
                        labels.append(target_label)
                        data.append(line[1])
                        break

        return data, labels

    def label_encoder(self, labels):
        label = 0
        if labels[0] == "OFF":
            if labels[1] == "UNT":
                label = 1
            elif labels[1] == "TIN":
                if labels[2] == "IND":
                    label = 2
                elif labels[2] == "GRP":
                    label = 3
                elif labels[2] == "OTH":
                    label = 4

        return label

    def get_test_data(self, path):
        data = []
        with open(path, encoding="utf8") as tf:
            reader = csv.reader(tf, delimiter="\t")
            for i, line in enumerate(reader):
                if i == 0:
                    continue

                data.append(line[1])

        return data

    def get_test_labels(self, path):
        labels = []
        with open(path, encoding="utf8") as tf:
            reader = csv.reader(tf, delimiter=",")
            for line in reader:
                labels.append(line[1])

        return labels
