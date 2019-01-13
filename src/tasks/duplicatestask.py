from .task import Task

# Third-party libs
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .vectorizer_selector import VectorizerSelector

class DuplicatesTask(Task):
    def __init__(self, train, test, vectorizer, threshold, stop_words):
        Task.__init__(self, train, test)

        self.train_raw_docs = self.train['Content'].values
        self.stop_words = stop_words
        self.threshold = threshold
        self.vectorizer = VectorizerSelector(vectorizer, stop_words).vectorizer

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
        df.to_csv("../results/duplicatePairs.csv", header=False, index=False)
        