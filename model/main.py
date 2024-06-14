import tensorflow as tf

from keras.layers import Dense, Dropout, Input, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from keras.metrics import SparseCategoricalAccuracy
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split

from data_frame import DataFrame
from distri_bert_classify import DistriBertClassify


def train():
    df = DataFrame("./dataset.csv", "Resume", "Category")
    df.preprocess()

    df_train, df_test = train_test_split(
        df.data, test_size=0.3, shuffle=True, random_state=101
    )

    max_length = 300
    input_ids_key = "input_ids"
    attention_mask_key = "attention_mask"

    model = DistriBertClassify(
        "manishiitg/distilbert-resume-parts-classify",
        from_pt=True,
        max_length=max_length,
        input_ids_key=input_ids_key,
        attention_mask_key=attention_mask_key,
    )
    model.update_data_frame(
        df_train,
        df_test,
    )

    input_ids = Input(shape=(max_length,), dtype=tf.int32, name=input_ids_key)
    attention_masks = Input(shape=(max_length,), dtype=tf.int32, name=attention_mask_key)

    output = Flatten()(input_ids)
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

    model.compile(
        inputs=[input_ids, attention_masks],
        outputs=output,
        optimizer=Adam(learning_rate=5e-5, epsilon=2e-8, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=SparseCategoricalAccuracy("balanced_accuracy"),
    )

    model.train(
        [
            EarlyStopping(
                monitor="val_balanced_accuracy",
                patience=250,
                verbose=1,
                mode="max",
                restore_best_weights=True,
            ),
            ModelCheckpoint(
                "resume_parser.h5",
                monitor="val_balanced_accuracy",
                verbose=1,
                mode="max",
                save_best_only=True,
            ),
        ]
    )


def load():
    model = DistriBertClassify(
        "manishiitg/distilbert-resume-parts-classify",
        model_path="resume_parser.h5",
        max_length=300,
    )

    model.predict(["I am a software engineer", "I am a data scientist"])

load()
