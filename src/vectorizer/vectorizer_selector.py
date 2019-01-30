from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

from vectorizer import EmbeddingsVectorizer


class VectorizerSelector:
    def __init__(self, vectorizer, stop_words):
        # Initialize vectorizer
        # TODO: Add Word2Vec vectorizer
        if vectorizer is 'bow':
            self.vectorizer = CountVectorizer(stop_words=stop_words)
        elif vectorizer is 'tfidf':
            self.vectorizer = TfidfVectorizer(stop_words=stop_words)
        elif vectorizer is 'hash':
            self.vectorizer = HashingVectorizer(stop_words=stop_words, n_features=2**18)
        elif vectorizer is 'w2v':
            self.vectorizer = EmbeddingsVectorizer()
        else:
            raise Exception("{} is not a valid vectorizer".format(vectorizer))