import os
import json

import numpy as np
import tensorflow as tf
import bisect

import tqdm
from sklearn.metrics import precision_recall_fscore_support


def sample_segment(n_seg_splits=5, max_seg_len=600):
    """Sample a topic segment from an input sequence.

    :param n_seg_splits: Number of splits to sample from the sequence
    :param max_seg_len: the maximum length of a segment from the sequence
    :return:
    """
    sorted_keys = np.linspace(0, 1, n_seg_splits)
    order = bisect.bisect_left(sorted_keys, np.random.random())
    return range(100, max_seg_len, 100)[order - 1]


def break_sequence(sequence, n_seg_splits=5, max_seg_len=600):
    """

    :param sequence:
    :param n_seg_splits:
    :param max_seg_len:
    :return:
    """
    # the original length of the sequence
    seq_length = len(sequence)

    # with 0.2 probability sample the original sequence
    if np.random.random() > 0.8:
        return sequence
    # with 0.8 probability sample a truncated sequence
    trunc_length = sample_segment(n_seg_splits=n_seg_splits, max_seg_len=max_seg_len)
    # return the original sequence if truncation exceeds the sequence size
    if trunc_length > seq_length:
        return sequence
    # sample the last truncated_len tokens from the sequence witj 05. probability
    elif np.random.random() > 0.5:
        return sequence[seq_length - trunc_length :]
    # sample the first truncated_len tokens from the sequence with 05. probability
    else:
        return sequence[:trunc_length]


def sample_record(sequences, max_len=2048, n_seg_splits=5, max_seg_len=600):
    """

    :param sequences:
    :param max_len:
    :param splits:
    :param max_seg_len:
    :return:
    """
    idxs = np.arange(len(sequences))
    # sample about 10 sequences for creating a sample input
    choices = np.random.choice(idxs, 10)
    out_seq, out_target = [], []
    for choice in choices:
        if len(out_seq) < max_len:
            # get a segment of topic for the output
            broken_sequence = break_sequence(
                sequences[choice], n_seg_splits=n_seg_splits, max_seg_len=max_seg_len
            )
            out_seq.extend(broken_sequence)
            # track the start of topic for each segment
            out_target.extend([0] + ([1] * (len(broken_sequence) - 1)))
        else:
            out_seq = np.array(out_seq[:max_len], dtype=np.int32)
            out_target = np.array(out_target[:max_len], dtype=np.int32)
            return out_seq, out_target
    out_seq = np.array(out_seq[:max_len], dtype=np.int32)
    out_target = np.array(out_target[:max_len], dtype=np.int32)
    return out_seq, out_target


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, batch_size=32, max_len=2048, n_seg_splits=5, max_seg_len=600):
        self.x = x
        self.batch_size = batch_size
        self.max_len = max_len
        self.n_seg_splits = n_seg_splits
        self.max_seg_len = max_seg_len

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_x, batch_y = [], []
        for i in range(self.batch_size):
            x, y = sample_record(
                self.x,
                max_len=self.max_len,
                n_seg_splits=self.n_seg_splits,
                max_seg_len=self.max_seg_len,
            )
            batch_x.append(x), batch_y.append(y)
        batch_x = tf.keras.preprocessing.sequence.pad_sequences(
            batch_x,
            maxlen=self.max_len,
            dtype="int32",
            padding="post",
            truncating="post",
            value=0,
        )
        batch_y = tf.keras.preprocessing.sequence.pad_sequences(
            batch_y,
            maxlen=self.max_len,
            dtype="int32",
            padding="post",
            truncating="post",
            value=1,
        )
        return batch_x, batch_y


class MetricsLogCallback(tf.keras.callbacks.Callback):
    def __init__(self, f_name):
        super(MetricsLogCallback, self).__init__()
        self.f_name = f_name
        if os.path.isfile(f_name):
            mode = "a+"
        else:
            mode = "w+"
        self.json_log = open(f_name, mode=mode, buffering=1)

    def on_epoch_end(self, epoch, logs=None):
        # convert ndarrays to list
        converted_log = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in logs.items()
        }
        converted_log = {
            k: (v.item() if isinstance(v, np.float32) else v) for k, v in logs.items()
        }
        self.json_log.write(json.dumps({"epoch": epoch, "logs": converted_log}) + "\n"),

    def on_train_end(self, logs=None):
        self.json_log.close()


def load_callbacks(
    lr_scheduler_params,
    checkpoint_params,
    early_stopping_params,
    tensorboard_params,
    logger_params,
):
    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(**lr_scheduler_params)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(**checkpoint_params)
    early_stopping = tf.keras.callbacks.EarlyStopping(**early_stopping_params)
    tensorboard = tf.keras.callbacks.TensorBoard(**tensorboard_params)
    logger = MetricsLogCallback(**logger_params)

    callback_list = [
        lr_schedule,
        checkpoint,
        early_stopping,
        tensorboard,
        logger,
    ]
    return callback_list


def load_weights(model, weights_fname):
    if os.path.isfile(weights_fname):
        model.load_weights(weights_fname)
    return model


def compile_model(model):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0003,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam",
    )
    loss = tf.keras.losses.binary_crossentropy
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def pk(seg, k=None, boundary=1):
    ref, hyp = seg
    if k is None:
        k = int(round(len(ref) / (ref.count(boundary) * 2.0)))

    err = 0
    for i in range(len(ref) - k + 1):
        r = ref[i : i + k].count(boundary) > 0
        h = hyp[i : i + k].count(boundary) > 0
        if r != h:
            err += 1
    return err / (len(ref) - k + 1.0)


def score_results(y_true, y_pred):
    y_true, y_pred = y_true.tolist(), y_pred.tolist()
    y_segs = zip(y_true, y_pred)
    scores = map(pk, y_segs)
    return np.mean(list(scores))


def eval_model(model, dataloader, steps=100, thresh=0.4):
    results = {}
    datagen = iter(dataloader)
    y_true, y_pred = [], []
    steps = len(dataloader) - 1 if not steps else steps
    for i in tqdm.trange(steps):
        batch_x, batch_y = next(datagen)
        preds = model.predict(batch_x)
        y_true.append(batch_y)
        y_pred.append(preds)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    print(y_pred)
    y_pred = y_pred > thresh
    results["pk_score"] = score_results(y_true, y_pred)
    precision, recall, f_score, _ = precision_recall_fscore_support(
        y_true=np.array(y_true).astype(np.bool).flatten(),
        y_pred=np.array(y_pred).flatten(),
        average="macro",
    )
    results["precision"] = precision
    results["recall"] = recall
    results["f1_score"] = f_score
    return results
