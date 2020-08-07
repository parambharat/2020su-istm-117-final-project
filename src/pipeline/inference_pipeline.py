from transformers import TransfoXLTokenizer
import functools
import tensorflow as tf

import src.pipeline.input_pipeline as ipt
from src.models import transformer_model
from src.training import utils


class Config:
    TOKENIZER_FNAME = "transfo-xl-wt103"
    MAX_SEQ_LEN = 1024

    TRAIN_EMBED = False
    LSTM_DIM = 512
    HIDDEN_DIM = 512
    DROPOUT_RATE = 0.5
    EXP_NAME = f"transformer_xl-{int(TRAIN_EMBED)}_LS-{LSTM_DIM}_HD-{HIDDEN_DIM}"
    MODEL_LOGS_PATH = f"models/transformer_xl/{EXP_NAME}/logs"
    MODEL_WEIGHTS_PATH = f"models/transformer_xl/{EXP_NAME}/{EXP_NAME}.hdf5"
    THRESHOLD = 0.5


def load_model(config):
    model = transformer_model.create_model(
        max_len=config.MAX_SEQ_LEN,
        lstm_dim=config.LSTM_DIM,
        hidden_dim=config.HIDDEN_DIM,
        dropout_rate=config.DROPOUT_RATE,
        train_embeddings=config.TRAIN_EMBED,
    )
    model.summary()
    model = utils.load_weights(model, config.MODEL_WEIGHTS_PATH)
    return model


def load_tokenizer(config):
    tokenizer = TransfoXLTokenizer.from_pretrained(config.TOKENIZER_FNAME)
    return tokenizer


class InferencePipeline:
    def __init__(self, config):
        self.config = config
        self.model = load_model(config)
        self.tokenizer = load_tokenizer(config)
        self.encode = self.encoder(self.tokenizer)
        self.decode = self.decoder(self.tokenizer)

    def encoder(self, tokenizer):
        tokenize = functools.partial(
            tokenizer.encode, add_space_before_punct_symbol=True
        )
        funcs = [ipt.preprocess_input, tokenize]

        def _inner(x):
            for func in funcs:
                x = func(x)
            x = tf.keras.preprocessing.sequence.pad_sequences(
                [x],
                maxlen=self.config.MAX_SEQ_LEN,
                dtype="int32",
                padding="post",
                truncating="post",
                value=0,
            )
            return x

        return _inner

    def decoder(self, tokenizer):
        def _inner(x):
            decoded = tokenizer.decode(x)
            return decoded

        return _inner

    def segment_text(self, text):
        text = self.encode(text)
        preds = self.model.predict(text)
        decoded = self.decode(text[0])
        segments = preds[0].tolist()
        return segments, decoded


def main():
    inference_config = Config()
    inference_pipe = InferencePipeline(inference_config)
    sample_text = open("data/microsoft_transcript.txt").read()
    result = inference_pipe.segment_text(sample_text)
    print(result)


if __name__ == "__main__":
    main()
