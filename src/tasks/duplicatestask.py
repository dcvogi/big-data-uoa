from .task import Task

# Third-party libs
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DuplicatesTask(Task):
    def __init__(self, train, test, vectorizer, threshold, stop_words):
        Task.__init__(self, train, test)

        self.train_raw_docs = self.train['Content'].values
        self.stop_words = stop_words
        self.threshold = threshold

        # Initialize vectorizer
        # TODO: Add Word2Vec vectorizer
        if vectorizer is 'bow':
            self.vectorizer = CountVectorizer(stop_words=self.stop_words)
        elif vectorizer is 'tfidf':
            self.vectorizer = TfidfVectorizer(stop_words=self.stop_words)
        elif vectorizer is 'hash':
            self.vectorizer = HashingVectorizer(stop_words=self.stop_words, n_features=2**4)
        elif vectorizer is 'w2v':
            raise NotImplementedError("W2V vectorizer is not implemented")
        else:
            raise Exception("{} is not a valid vectorizer".format(vectorizer))
        
        # Generate document vectors from raw documents
        self.doc_vectors = self.vectorizer.fit_transform(self.train_raw_docs)
        

    def run_task(self):
        # Main Functionality goes here
        similarity_matrix = np.matrix(cosine_similarity(self.doc_vectors))
        sim_docs = []
        for i, val in enumerate(similarity_matrix):
            for j, val_two in enumerate(similarity_matrix):
                if i != j and j > i:
                    if similarity_matrix[i, j] > self.threshold:
                        sim_docs.append([i, j, similarity_matrix[i, j]])

        for index, similar in enumerate(sim_docs):
            sim_docs[index][0] = self.train['Id'].values[similar[0]]
            sim_docs[index][1] = self.train['Id'].values[similar[1]]

        df = pd.DataFrame(sim_docs)
        df.to_csv("output.csv", header=False, index=False)
        