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

    def transform(self, docs):
        mean_vectors = []
        for doc in docs:
            tokenized_doc = [word for word in doc.lower().split() if word in self.word2idx]

            if not tokenized_doc:
                doc_vector = [np.zeros(self.model.vector_size)]
            else:
                doc_vector = [self.model[word] for word in tokenized_doc]

            mean_vector = np.mean(doc_vector, axis=0)

            mean_vectors.append(mean_vector)

        result = np.array(mean_vectors)
        return result

    def fit_transform(self, docs):
        self.fit(docs)

        vectors = self.transform(docs)

        return vectors
