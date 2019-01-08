from .task import Task

# Third-party libs
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DuplicatesTask(Task):
    def __init__(self, train, test, vectorizer, threshold):
        Task.__init__(self, train, test)

        self.train_raw_docs = self.train['Content'].values
        self.threshold = threshold

        # Initialize vectorizer
        # TODO: Add Word2Vec vectorizer
        if vectorizer is 'bow':
            self.vectorizer = CountVectorizer(stop_words='english')
        elif vectorizer is 'tfidf':
            self.vectorizer = TfidfVectorizer(stop_words='english')
        elif vectorizer is 'w2v':
            raise NotImplementedError("W2V vectorizer is not implemented")
        else:
            raise Exception("{} is not a valid vectorizer".format(vectorizer))
        
        # Generate document vectors from raw documents
        self.doc_vectors = self.vectorizer.fit_transform(self.train_raw_docs)
        

    def run_task(self):
        # Main Functionality goes here
        similarity_matrix = np.matrix(cosine_similarity(self.doc_vectors))
        
        