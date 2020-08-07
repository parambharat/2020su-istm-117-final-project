import json
import pandas as pd

from src.training import utils
from src.models import transformer_model
from transformers import TransfoXLTokenizer
import functools


class Config:
    TRAIN_FNAME = "data/train_data.txt"
    VAL_FNAME = "data/val_data.txt"
    TEST_FNAME = "data/test_data.txt"
    TOKENIZER_FNAME = "transfo-xl-wt103"
    MAX_SEQ_LEN = 1024
    BATCH_SIZE = 6
    EPOCH_LEN = 100
    VAL_LEN = 20
    N_EPOCHS = 100

    N_SEG_SPLITS = 5
    MAX_SEG_LEN = 600
    TRAIN_EMBED = False
    LSTM_DIM = 512
    HIDDEN_DIM = 512
    DROPOUT_RATE = 0.5
    EXP_NAME = f"transformer_xl-{int(TRAIN_EMBED)}_LS-{LSTM_DIM}_HD-{HIDDEN_DIM}"
    MODEL_LOGS_PATH = f"models/transformer_xl/{EXP_NAME}/logs"
    MODEL_WEIGHTS_PATH = f"models/transformer_xl/{EXP_NAME}/{EXP_NAME}.hdf5"

    CALLBACK_PARAMS = {
        "lr_scheduler_params": {
            "monitor": "val_loss",
            "factor": 0.1,
            "patience": 5,
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
            "save_weights_only": True,
            "verbose": 1,
        },
        "early_stopping_params": {
            "monitor": "val_loss",
            "min_delta": 0.001,
            "patience": 10,
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

    tokenizer = TransfoXLTokenizer.from_pretrained(config.TOKENIZER_FNAME)
    tokenize = functools.partial(tokenizer.encode, add_space_before_punct_symbol=True)

    train_data = list(map(tokenize, train_data))
    val_data = list(map(tokenize, val_data))
    test_data = list(map(tokenize, test_data))

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

    model = transformer_model.create_model(
        max_len=config.MAX_SEQ_LEN,
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
