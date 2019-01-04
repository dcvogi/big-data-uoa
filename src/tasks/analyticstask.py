from .task import Task
from .evaluation_report import EvaluationReport

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class AnalyticsTask(Task):
    def __init__(self, train, test, classifier, vectorizer, test_size):
        Task.__init__(self, train, test)

        self.label_encoder = LabelEncoder()
        self.train_raw_docs = self.train['Content'].values
        self.train_raw_labels = self.train['Category'].values
        self.test_raw_docs = self.test['Content'].values
        self.classifier = classifier
        self.test_size = test_size

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

    def test_model(self):
        if self.classifier is "svm":
            print "Running Support Vector Machine"
            clf = self.support_vector_machine()
        elif self.classifier is "rf":
            print "Running Random Forest"
            clf = self.random_forest()

        train_x, test_x, train_y, test_y = train_test_split(self.doc_vectors, self.doc_labels,
                                                            test_size=self.test_size)

        predicted = clf.predict(test_x)

        print self.evaluate(test_y, predicted).__dict__

        return predicted

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
        self.test_model()
