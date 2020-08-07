import tensorflow as tf
from transformers import TFTransfoXLModel


def create_model(
    max_len, lstm_dim=512, hidden_dim=512, dropout_rate=0.5, train_embeddings=False
):
    inputs = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32)
    model = TFTransfoXLModel.from_pretrained("transfo-xl-wt103")
    model.trainable = train_embeddings

    dropout_layer = tf.keras.layers.Dropout(dropout_rate)
    lstm_layer = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_dim, return_sequences=True), merge_mode=None
    )
    layer_normalization = tf.keras.layers.LayerNormalization()
    reprojection_layer = tf.keras.layers.Dense(hidden_dim, activation="relu")

    outputs = model(inputs)
    last_hidden_states = outputs[0]
    last_hidden_states = dropout_layer(last_hidden_states)
    lstm_outputs = lstm_layer(last_hidden_states)

    fwd_outputs = layer_normalization(lstm_outputs[0])
    bkd_outputs = layer_normalization(lstm_outputs[1])

    fwd_outputs = reprojection_layer(fwd_outputs)
    bkd_outputs = reprojection_layer(bkd_outputs)

    fwd_outputs = dropout_layer(fwd_outputs)
    bkd_outputs = dropout_layer(bkd_outputs)

    sims = tf.matmul(fwd_outputs, tf.transpose(bkd_outputs, perm=[0, 2, 1]))
    sims = tf.linalg.diag_part(sims)

    sims = tf.nn.sigmoid(sims)
    model = tf.keras.models.Model([inputs], [sims])
    model.summary()
    return model
