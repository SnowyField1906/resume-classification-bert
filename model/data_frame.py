import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from model.text_preprocessor import TextPreprocessor


class DataFrame:
    def __init__(self, path: str, x_col: str, y_col: str):
        data = pd.read_csv(path)
        data.rename(columns={x_col: "x", y_col: "y"}, inplace=True)

        data["resume_len"] = data.x.apply(len)

        plt.rcParams['figure.figsize'] = (12,8)
        sns.countplot(data.y)
        plt.ylabel("labels")
        plt.tight_layout()
        plt.savefig("./model/assets/label_distribution.png")

        labels_dict = {}
        for idx, label in enumerate(data.y.unique()):
            labels_dict[label] = idx

        data.y = data.y.apply(func=lambda x: labels_dict[x])
        data.y = data.y.astype(np.int64)

        self.data = data
        self.labels_dict = labels_dict


    def preprocess(self):
        text_preprocessor = TextPreprocessor()

        for func in text_preprocessor:
            self.data.x = self.data.x.apply(func=func)

    def labels(self, output: list[int]) -> dict[str, str]:
        labels = {}
        keys = list(self.labels_dict.keys())

        for idx, val in enumerate(output):
            labels[keys[idx]] = str(round(val * 100, 2))

        return labels