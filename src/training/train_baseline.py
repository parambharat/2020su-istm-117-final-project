import os
import json
import pandas as pd

from src.training import utils
from src.models import baseline_model
from src.pipeline import input_pipeline as ipt


class Config:
    TRAIN_FNAME = "data/train_data.txt"
    VAL_FNAME = "data/val_data.txt"
    TEST_FNAME = "data/test_data.txt"
    TOKENIZER_FNAME = "models/tokenizers/baseline_tokenizer.json"
    EMBEDDING_DIM = 300
    MAX_SEQ_LEN = 1024
    BATCH_SIZE = 6
    EPOCH_LEN = 1000
    VAL_LEN = 20
    N_EPOCHS = 10

    GLOVE_DIR = "data/glove.6B"
    GLOVE_PATH = os.path.join(GLOVE_DIR, f"glove.6B.{EMBEDDING_DIM}d.txt")

    N_SEG_SPLITS = 5
    MAX_SEG_LEN = 600
    TRAIN_EMBED = True
    N_LSTM = 2
    LSTM_DIM = 512
    HIDDEN_DIM = 512
    DROPOUT_RATE = 0.5
    EXP_NAME = f"baseline_em-{EMBEDDING_DIM}-{int(TRAIN_EMBED)}_LS-{N_LSTM}-{LSTM_DIM}_HD-{HIDDEN_DIM}-rev"
    MODEL_LOGS_PATH = f"models/baseline_models/{EXP_NAME}/logs"
    MODEL_WEIGHTS_PATH = f"models/baseline_models/{EXP_NAME}/{EXP_NAME}.hdf5"

    CALLBACK_PARAMS = {
        "lr_scheduler_params": {
            "monitor": "val_loss",
            "factor": 0.1,
            "patience": 2,
            "verbose": 1,
            "mode": "auto",
            "min_delta": 1e-4,
            "cooldown": 0,
            "min_lr": 0,
        },
        "checkpoint_params": {
            "filepath": MODEL_WEIGHTS_PATH,
            "monitor": "val_loss",
            "save_best": True,
            "save_weights": True,
            "verbose": 1,
        },
        "early_stopping_params": {
            "monitor": "val_loss",
            "min_delta": 0.001,
            "patience": 5,
            "verbose": 1,
            "mode": "auto",
            "restore_best_weights": True,
        },
        "tensorboard_params": {"log_dir": MODEL_LOGS_PATH},
        "logger_params": {"f_name": f"{MODEL_LOGS_PATH}/metrics.json"},
    }


def train(config):
    train_data = open(config.TRAIN_FNAME).readlines()
    val_data = open(config.VAL_FNAME).readlines()
    test_data = open(config.TEST_FNAME).readlines()

    tokenizer = ipt.load_tokenizer(
        train_data, f_name=config.TOKENIZER_FNAME, force=False
    )
    word_index = tokenizer.word_index
    embedding_matrix = ipt.load_embedding_matrix(
        word_index, config.GLOVE_PATH, config.EMBEDDING_DIM
    )

    vocab_size, embedding_dim = embedding_matrix.shape

    train_data = tokenizer.texts_to_sequences(train_data)
    val_data = tokenizer.texts_to_sequences(val_data)
    test_data = tokenizer.texts_to_sequences(test_data)

    train_dataloader = utils.DataGenerator(
        train_data,
        batch_size=config.BATCH_SIZE,
        max_len=config.MAX_SEQ_LEN,
        n_seg_splits=config.N_SEG_SPLITS,
        max_seg_len=config.MAX_SEG_LEN,
    )

    val_dataloader = utils.DataGenerator(
        val_data,
        batch_size=config.BATCH_SIZE,
        max_len=config.MAX_SEQ_LEN,
        n_seg_splits=config.N_SEG_SPLITS,
        max_seg_len=config.MAX_SEG_LEN,
    )

    test_dataloader = utils.DataGenerator(
        test_data,
        batch_size=config.BATCH_SIZE,
        max_len=config.MAX_SEQ_LEN,
        n_seg_splits=config.N_SEG_SPLITS,
        max_seg_len=config.MAX_SEG_LEN,
    )

    model = baseline_model.create_model(
        max_len=config.MAX_SEQ_LEN,
        embedding_matrix=embedding_matrix,
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
        n_lstm=config.N_LSTM,
        lstm_dim=config.LSTM_DIM,
        hidden_dim=config.HIDDEN_DIM,
        dropout_rate=config.DROPOUT_RATE,
        train_embeddings=config.TRAIN_EMBED,
    )
    model.summary()
    model = utils.compile_model(model)
    model = utils.load_weights(model, config.MODEL_WEIGHTS_PATH)

    callbacks = utils.load_callbacks(**config.CALLBACK_PARAMS)
    history = model.fit(
        train_dataloader,
        validation_data=val_dataloader,
        steps_per_epoch=config.EPOCH_LEN,
        validation_steps=config.VAL_LEN,
        epochs=config.N_EPOCHS,
        callbacks=[callbacks],
    )
    hist_df = pd.DataFrame(history.history)
    hist_df.to_json(f"{config.MODEL_LOGS_PATH}/history.json")

    test_eval_results = utils.eval_model(model, test_dataloader, thresh=0.5, steps=None)
    eval_results_path = f"{config.MODEL_LOGS_PATH}/eval_results.json"

    json.dump(test_eval_results, open(eval_results_path, "w+"))


def main():
    config = Config()
    train(config)


if __name__ == "__main__":
    main()
