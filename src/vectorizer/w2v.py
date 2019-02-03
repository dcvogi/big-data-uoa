from gensim.models import Word2Vec
import numpy as np


class EmbeddingsVectorizer:
    def __init__(self, max_features):
        self.model = None
        self.words = None
        self.word2idx = None
        self.max_features = max_features

    def fit(self, docs):
        tokenized_docs = [d.split() for d in docs]
        self.model = Word2Vec(tokenized_docs, window=5, min_count=0, workers=4, size=self.max_features)
        self.words = self.model.wv.vocab
        self.word2idx = dict(zip(self.words, range(0, len(self.words))))

        return self

    def transform(self, doc):
        tokenized_doc = [word for word in doc.lower().split() if word in self.word2idx]

        if not tokenized_doc:
            return np.zeros(self.model.vector_size)

        doc_vector = [self.model[word] for word in tokenized_doc]

        mean_vector = np.mean(doc_vector, axis=0)

        return mean_vector

    def fit_transform(self, docs):
        self.fit(docs)

        vectors = [self.transform(doc) for doc in docs]

        return vectors
