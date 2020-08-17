import csv
import pprint
from typing import Dict, List, Tuple

pp = pprint.PrettyPrinter(indent=4)


class DataReader:
    def __init__(self, path: str) -> None:
        self.data_path: str = path
        self.target_to_og_labels: Dict[int, Dict[int, List[int]]] = {
            0: {0: [0], 1: [1, 2, 3, 4]},
            1: {0: [1], 1: [2, 3, 4]},
            2: {0: [2], 1: [3], 2: [4]},
        }

        self.og_to_target_labels = {
            problem_id: {
                target: og for og, targets in mapping.items() for target in targets
            }
            for problem_id, mapping in self.target_to_og_labels.items()
        }

    def get_data_and_labels(self, problem_id: int) -> Tuple[List[str], List[int]]:
        data = []
        labels = []
        with open(self.data_path, encoding="utf8") as tf:
            reader = csv.reader(tf, delimiter="\t")
            for i, line in enumerate(reader):
                if i == 0:
                    continue

                label_list = line[-3:]
                label_encoded = self.label_encoder(label_list)
                if label_encoded in self.og_to_target_labels[problem_id]:
                    labels.append(self.og_to_target_labels[problem_id][label_encoded])
                    data.append(line[1])

                # for target_label, origin_labels in self.target_to_og_labels[
                #     problem_id
                # ].items():
                #     if label_encoded in origin_labels:
                #         labels.append(target_label)
                #         data.append(line[1])
                #         break

        return data, labels

    def label_encoder(self, labels: List[str]) -> int:
        label: int = 0
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


if __name__ == "__main__":
    dr = DataReader("../OLIDv1.0/olid-training-v1.0.tsv")
    data = dr.get_test_data("../OLIDv1.0/testset-levela.tsv")
    labels = dr.get_test_labels("../OLIDv1.0/labels-levela.csv")
    breakpoint()
