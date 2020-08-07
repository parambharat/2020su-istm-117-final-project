import tensorflow as tf


def create_model(
    max_len,
    embedding_matrix,
    embedding_dim,
    vocab_size,
    n_lstm=2,
    lstm_dim=512,
    hidden_dim=512,
    dropout_rate=0.5,
    train_embeddings=False,
):

    inputs = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32)

    embedding_layer = tf.keras.layers.Embedding(
        vocab_size,
        embedding_dim,
        weights=[embedding_matrix],
        trainable=train_embeddings,
    )

    dropout_layer = tf.keras.layers.Dropout(dropout_rate)
    init_lstm_layers = []
    if n_lstm:
        for i in range(n_lstm):
            layer = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(lstm_dim, return_sequences=True)
            )
            init_lstm_layers.append(layer)
    lstm_layer = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_dim, return_sequences=True), merge_mode=None
    )
    layer_normalization = tf.keras.layers.LayerNormalization()
    reprojection_layer = tf.keras.layers.Dense(hidden_dim, activation="relu")

    last_hidden_states = embedding_layer(inputs)
    if n_lstm:
        for layer in init_lstm_layers:
            last_hidden_states = layer(last_hidden_states)

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
