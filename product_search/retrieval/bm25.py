import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from product_search import config
from product_search.text import tokenize


class BM25:
    def __init__(self, k1: float = config.BM25_K1, b: float = config.BM25_B):
        self.k1, self.b = k1, b

    def fit(self, texts):
        self.vectorizer = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        tf = self.vectorizer.fit_transform(texts).tocoo()
        doc_len = np.bincount(tf.row, weights=tf.data, minlength=tf.shape[0])
        df = np.bincount(tf.col, minlength=tf.shape[1])
        idf = np.log(1 + (tf.shape[0] - df + 0.5) / (df + 0.5))
        norm = self.k1 * (1 - self.b + self.b * doc_len / doc_len.mean())
        w = tf.data * (self.k1 + 1) / (tf.data + norm[tf.row]) * idf[tf.col]
        self.weights = sparse.csc_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)
        return self

    def scores(self, query: str) -> np.ndarray:
        vocab = self.vectorizer.vocabulary_
        cols = [vocab[t] for t in tokenize(query) if t in vocab]
        if not cols:
            return np.zeros(self.weights.shape[0], dtype=np.float32)
        return np.asarray(self.weights[:, cols].sum(axis=1)).ravel()
