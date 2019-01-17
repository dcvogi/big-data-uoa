from gensim.models import Word2Vec
import numpy as np


class EmbeddingsVectorizer:
    def __init__(self):
        self.model = None
        self.words = None
        self.word2idx = None

    def fit(self, docs):
        tokenized_docs = [d.split() for d in docs]
        self.model = Word2Vec(tokenized_docs, window=5, min_count=10, workers=4, size=100)
        self.words = self.model.wv.vocab
        self.word2idx = dict(zip(self.words, range(0, len(self.words))))

        return self

    def transform1(self, doc):
        tokenized_doc = [word for word in doc.lower().split() if word in self.word2idx]

        if not tokenized_doc:
            return np.zeros(self.model.vector_size)

        doc_vector = [np.zeros(self.model.vector_size) for i in range(0, len(self.words))]

        for word in tokenized_doc:
            word_idx = self.word2idx[word]
            doc_vector[word_idx] = self.model[word]

        mean_vector = np.mean(doc_vector, axis=0)

        return mean_vector


    def transform(self, doc):
        return self.transform2(doc)

    def fit_transform(self, docs):
        self.fit(docs)

        vectors = [self.transform(doc) for doc in docs]

        return vectors
