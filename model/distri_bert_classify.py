import numpy as np
import tensorflow as tf

from keras.layers import Dense, Dropout, Input, Flatten, BatchNormalization
from keras.models import load_model, Model
from transformers import TFDistilBertForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix


class DistriBertClassify:
    def __init__(
        self,
        name,
        from_pt=True,
        model_path=None,
        max_length=None,
        input_ids_key=None,
        attention_mask_key=None,
    ):
        self.max_length = max_length
        self.input_ids_key = input_ids_key
        self.attention_mask_key = attention_mask_key
        
        self.tokenizer = AutoTokenizer.from_pretrained(name)

        if model_path:
            self.bert_model = load_model(model_path)
        else:
            self.bert_model = TFDistilBertForSequenceClassification.from_pretrained(
                name, from_pt=from_pt
            )

            input_ids = Input(shape=(self.max_length,), dtype=tf.int32, name=self.input_ids_key)
            attention_masks = Input(shape=(self.max_length,), dtype=tf.int32, name=self.attention_mask_key)

            embeddings = self.bert_model(input_ids=input_ids, attention_mask=attention_masks)[0]

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
            self.input_ids_key: x_train[self.input_ids_key],
            self.attention_mask_key: x_train[self.attention_mask_key],
        }
        validation_data = {
            self.input_ids_key: x_test[self.input_ids_key],
            self.attention_mask_key: x_test[self.attention_mask_key],
        }

        self.bert_model.fit(
            x=train_data,
            y=self.df_train.y,
            validation_data=(validation_data, self.df_test.y),
            callbacks=callbacks,
            epochs=500,
            batch_size=32,
        )

        self.bert_model.save("resume_parser.h5", save_format="tf")

        loss, acc = self.bert_model.evaluate(
            validation_data,
            self.df_test.y,
        )
        print("Test Sparse Categorical Crossentropy Loss:", loss)
        print("Test Balanced Categorical Accuracy:", acc)

        test_predictions = self.bert_model.predict(validation_data)
        test_predictions = np.argmax(test_predictions, axis=1)

        print("Confusion Matrix:")
        print(confusion_matrix(self.df_test.y, test_predictions))
        print("Classification Report:")
        print(classification_report(self.df_test.y, test_predictions))

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
            self.input_ids_key: x[self.input_ids_key],
            self.attention_mask_key: x[self.attention_mask_key],
        }

        return self.bert_model.predict(data)
