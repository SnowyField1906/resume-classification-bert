import numpy as np
from keras.models import load_model

from keras.models import Model
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
        self.tokenizer = AutoTokenizer.from_pretrained(name)

        if model_path:
            self.bert_model = load_model(model_path)
        else:
            self.bert_model = TFDistilBertForSequenceClassification.from_pretrained(
                name, from_pt=from_pt
            )

        self.max_length = max_length
        self.input_ids_key = input_ids_key
        self.attention_mask_key = attention_mask_key

    def update_data_frame(self, df_train, df_test):
        self.df_train = df_train
        self.df_test = df_test

    def compile(self, inputs, outputs, optimizer, loss, metrics):
        self.bert_model = Model(inputs=inputs, outputs=outputs)
        self.bert_model.layers[2].trainable = True

        self.bert_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, callbacks):
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
