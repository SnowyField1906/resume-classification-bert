import numpy as np
import pandas as pd

from text_preprocessor import TextPreprocessor


class DataFrame:
    def __init__(self, path: str, x_col: str, y_col: str):
        data = pd.read_csv(path)
        data.rename(columns={x_col: "x", y_col: "y"}, inplace=True)

        data["resume_len"] = data.x.apply(len)

        labels_dict = {}
        for idx, label in enumerate(data.y.unique()):
            labels_dict[label] = idx
        labels_dict

        data.y = data.y.apply(func=lambda x: labels_dict[x])
        data.y = data.y.astype(np.int64)

        self.data = data

    def preprocess(self):
        text_preprocessor = TextPreprocessor()

        for func in text_preprocessor:
            self.data.x = self.data.x.apply(func=func)
