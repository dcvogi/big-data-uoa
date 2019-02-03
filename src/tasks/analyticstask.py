from .task import Task
from .evaluation_report import EvaluationReport
from vectorizer import VectorizerSelector

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD


class AnalyticsTask(Task):
    def __init__(self, train, test, classifier, vectorizer, test_size, max_features=100, svd=False,
                 components_percentage=0.3):
        Task.__init__(self, train, test)

        self.label_encoder = LabelEncoder()
        self.train_raw_docs = self.train['Content'].values
        self.train_raw_labels = self.train['Category'].values
        self.test_raw_docs = self.test['Content'].values
        self.classifier = classifier
        self.test_size = test_size
        self.svd = svd
        self.vectorizer = VectorizerSelector(vectorizer=vectorizer, stop_words='english',
                                             max_features=max_features).vectorizer

        # Generate document vectors from raw documents
        self.doc_vectors = self.vectorizer.fit_transform(self.train_raw_docs)
        self.num_dimensions = max(self.doc_vectors[0].shape)

        if svd:
            number_of_components = int(components_percentage*self.num_dimensions)
            svd_model = TruncatedSVD(n_components=number_of_components)
            self.doc_vectors = svd_model.fit_transform(self.doc_vectors)

        self.doc_labels = self.label_encoder.fit_transform(self.train_raw_labels)

    def support_vector_machine(self):
        train_x, test_x, train_y, test_y = train_test_split(self.doc_vectors, self.doc_labels,
                                                            test_size=self.test_size)

        svc = SVC(kernel='linear')
        svc.fit(train_x, train_y)

        return svc

    def random_forest(self):
        train_x, test_x, train_y, test_y = train_test_split(self.doc_vectors, self.doc_labels,
                                                            test_size=self.test_size)

        rf = RandomForestClassifier()
        rf.fit(train_x, train_y)

        return rf

    def multilayer_perceptron(self):
        train_x, test_x, train_y, test_y = train_test_split(self.doc_vectors, self.doc_labels,
                                                            test_size=self.test_size)

        mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(15,), random_state=1)
        mlp.fit(train_x, train_y)

        return mlp

    def test_model(self):
        print "Classifier={}, Vectorizer={}, SVD={}".format(self.classifier, type(self.vectorizer).__name__, self.svd)

        if self.classifier is "svm":
            clf = self.support_vector_machine()
        elif self.classifier is "rf":
            clf = self.random_forest()
        elif self.classifier is "mlp":
            clf = self.multilayer_perceptron()

        train_x, test_x, train_y, test_y = train_test_split(self.doc_vectors, self.doc_labels,
                                                            test_size=self.test_size)

        predicted = clf.predict(test_x)

        return self.evaluate(test_y, predicted).__dict__

    def evaluate(self, test_y, predicted):
        prec = precision_score(test_y, predicted, average='macro')
        rec = recall_score(test_y, predicted, average='macro')
        acc = accuracy_score(test_y, predicted)
        f1 = f1_score(test_y, predicted, average='macro')
        # TODO: Set an AUC score
        auc_score = None#roc_auc_score(test_y, predicted)

        report = EvaluationReport(accuracy=acc, precision=prec, recall=rec, f1=f1, auc=None)

        return report

    def run_task(self):
        return self.test_model()
