from .task import Task

# Third-party libs
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from vectorizer import VectorizerSelector

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
        # Get the cosine similarity matrix
        similarity_matrix = np.matrix(cosine_similarity(self.doc_vectors))
        # Get the upper triangular
        similarity_matrix_triu = np.triu(similarity_matrix, 1)
        # Replace the index with the Ids
        similarities = pd.DataFrame(similarity_matrix_triu,
                                    columns=self.train['Id'],
                                    index=self.train['Id'])
        # Return a reshaped DataFrame or Series having a multi-level index                                    
        similarities = similarities.stack()
        # Keep the data wich are above the threshold
        similarities = pd.DataFrame(similarities[similarities > self.threshold])
        # Renaming
        similarities.index.rename(['Document_ID1','Document_ID2'], inplace=True)
        similarities.reset_index(inplace=True)
        similarities.rename(columns={0: 'Similarity'}, inplace=True)
        
        similarities.to_csv('../results/duplicatePairs.csv', index=False)
