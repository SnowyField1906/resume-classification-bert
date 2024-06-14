import tensorflow as tf

from keras.layers import Dense, Dropout, Input, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from keras.metrics import SparseCategoricalAccuracy
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split

from model.data_frame import DataFrame
from model.distri_bert_classify import DistriBertClassify
from model.text_preprocessor import TextPreprocessor

max_length = 300
input_ids_key = "input_ids"
attention_mask_key = "attention_mask"

def train() -> tuple[float, float]:
    df = DataFrame("./model/dataset.csv", "Resume", "Category")
    df.preprocess()

    df_train, df_test = train_test_split(
        df.data, test_size=0.3, shuffle=True, random_state=101
    )

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

    loss, acc = model.train(
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

    return loss, acc


def load(content):
    # from tika import parser
    # content = str(parser.from_file("./resume.pdf")["content"])
    
    text_preprocessor = TextPreprocessor()
    data_frame = DataFrame("./model/dataset.csv", "Resume", "Category")

    for func in text_preprocessor:
        content = func(content)

    model = DistriBertClassify(
        "manishiitg/distilbert-resume-parts-classify",
        model_path="./model/resume_parser.h5",
        max_length=300,
        input_ids_key="input_ids",
        attention_mask_key="attention_mask",
    )
    res = model.predict([content])
    return data_frame.labels(res[0])