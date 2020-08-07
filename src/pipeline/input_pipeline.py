import os
import json

import numpy as np
import tensorflow as tf

from gensim.utils import simple_preprocess


def preprocess_input(text):
    text = simple_preprocess(text, deacc=True, min_len=2, max_len=15)
    text = " ".join(text)
    return text


def create_tokenizer(texts):
    texts = map(preprocess_input, texts)
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(texts)
    return tokenizer


def save_tokenizer(tokenizer, f_name):
    tokenizer_conf = tokenizer.to_json()
    open(f_name, "w+").write(tokenizer_conf)
    return True


def load_tokenizer(texts, f_name=None, cache=True, force=False):
    if f_name and os.path.isfile(f_name) and not force:
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(open(f_name).read())
    else:
        tokenizer = create_tokenizer(texts)
        if cache:
            save_tokenizer(tokenizer, f_name)
    return tokenizer


def load_glove_vectors(glove_path):
    embeddings_index = {}
    f = open(glove_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def load_embedding_matrix(word_index, glove_path, embedding_dim):
    embeddings_index = load_glove_vectors(glove_path)
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
