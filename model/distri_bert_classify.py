import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.layers import Dense, Dropout, Input, Flatten, BatchNormalization
from keras.models import load_model, Model
from keras.utils import plot_model
from transformers import TFDistilBertForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


class DistriBertClassify:
    def __init__(
        self,
        name,
        from_pt=True,
        model_path=None,
        max_length=None,
    ):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(name)

        if model_path:
            self.bert_model = load_model(model_path, {"TFDistilBertForSequenceClassification": TFDistilBertForSequenceClassification})
        else:
            self.bert_model = TFDistilBertForSequenceClassification.from_pretrained(
                name, from_pt=from_pt
            )

            input_ids = Input(shape=(self.max_length,), dtype=tf.int32, name="input_ids")
            attention_masks = Input(shape=(self.max_length,), dtype=tf.int32, name="attention_mask")

            embeddings = self.bert_model(input_ids, attention_mask=attention_masks)[0]

            output = Flatten()(embeddings)
            output = Dense(units=1024, activation="relu")(output)
            output = BatchNormalization()(output)
            output = Dropout(0.25)(output)
            output = Dense(units=512, activation="relu")(output)
            output = Dropout(0.25)(output)
            output = Dense(units=256, activation="relu")(output)
            output = BatchNormalization()(output)
            output = Dropout(0.25)(output)
            output = Dense(units=128, activation="relu")(output)
            output = Dropout(0.25)(output)
            output = Dense(units=64, activation="relu")(output)
            output = Dense(units=25, activation="softmax")(output)

            self.bert_model = Model(inputs=[input_ids, attention_masks], outputs=output)
            self.bert_model.layers[2].trainable = True

            plot_model(self.bert_model, to_file="./model/assets/model.png", show_shapes=True)

    def update_data_frame(self, df_train, df_test):
        self.df_train = df_train
        self.df_test = df_test

    def compile(self, optimizer, loss, metrics):
        self.bert_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, callbacks) -> tuple[float, float]:
        x_train = self.tokenizer(
            text=self.df_train.x.tolist(),
            max_length=self.max_length,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors="tf",
            return_attention_mask=True,
            return_token_type_ids=False,
            verbose=1,
        )
        x_test = self.tokenizer(
            text=self.df_test.x.tolist(),
            max_length=self.max_length,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors="tf",
            return_attention_mask=True,
            return_token_type_ids=False,
            verbose=1,
        )

        train_data = {
            "input_ids": x_train["input_ids"],
            "attention_mask": x_train["attention_mask"],
        }
        validation_data = {
            "input_ids": x_test["input_ids"],
            "attention_mask": x_test["attention_mask"],
        }

        t = self.bert_model.fit(
            x=train_data,
            y=self.df_train.y,
            validation_data=(validation_data, self.df_test.y),
            callbacks=callbacks,
            epochs=200,
            batch_size=32,
        )
        self.bert_model.save("./model/assets/resume_parser.h5", save_format="tf")

        loss, acc = self.bert_model.evaluate(
            validation_data,
            self.df_test.y,
        )
        print("Test Sparse Categorical Crossentropy Loss:", loss)
        print("Test Balanced Categorical Accuracy:", acc)

        test_predictions = self.bert_model.predict(validation_data)
        test_predictions = np.argmax(test_predictions, axis=1)

        # NOTE: Turn off when running on server
        # cm = confusion_matrix(self.df_test.y, test_predictions)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        # disp.plot()
        # plt.savefig("./model/assets/confusion_matrix.png")


        # NOTE: Turn off when running on server
        # _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        # ax1.plot(t.history['loss'],'r',label='train loss')
        # ax1.plot(t.history['val_loss'],'b',label='test loss')
        # ax1.set_xlabel('No. of Epochs')
        # ax1.set_ylabel('Categorical Crossentropy Loss')
        # ax1.set_title('Loss Graph')
        # ax1.legend();
        # ax1.grid(True)
        # ax2.plot(t.history['balanced_accuracy'],'r',label='train accuracy')
        # ax2.plot(t.history['val_balanced_accuracy'],'b',label='test accuracy')
        # ax2.set_xlabel('No. of Epochs')
        # ax2.set_ylabel('Balanced Categorical Accuracy')
        # ax2.set_title('Accuracy Graph')
        # ax2.legend();
        # ax2.grid(True)
        # plt.tight_layout()
        # plt.savefig("./model/assets/loss_accuracy_graph.png")

        with open("./model/assets/classification_report.json", "w") as f:
            json.dump(classification_report(self.df_test.y, test_predictions), f)

        return loss, acc

    def predict(self, text):
        x = self.tokenizer(
            text=text,
            max_length=self.max_length,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors="tf",
            return_attention_mask=True,
            return_token_type_ids=False,
            verbose=1,
        )

        data = {
            "input_ids": x["input_ids"],
            "attention_mask": x["attention_mask"],
        }

        return self.bert_model.predict(data)
