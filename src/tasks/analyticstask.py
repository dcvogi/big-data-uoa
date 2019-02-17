# Local imports
from .task import Task
from .evaluation_report import EvaluationReport
from vectorizer import VectorizerSelector

# Third party imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD

# Python imports
import pickle
import os


class AnalyticsTask(Task):
    def __init__(self, train, test, classifier, vectorizer, test_size, max_features=100, svd=False):
        Task.__init__(self, train, test)

        self.random_state = 512
        self.number_of_components = 550
        self.label_encoder = LabelEncoder()
        self.train_raw_docs = self.train['Content'].values
        self.train_raw_labels = self.train['Category'].values
        self.test_raw_docs = self.test['Content'].values
        self.test_raw_docs_ids = self.test['Id'].values
        self.classifier = classifier
        self.test_size = test_size
        self.svd = svd
        self.vectorizer = VectorizerSelector(vectorizer=vectorizer, stop_words='english',
                                             max_features=max_features).vectorizer

        # Generate document vectors from raw documents
        self.doc_vectors = self.vectorizer.fit_transform(self.train_raw_docs)
        self.test_vectors = self.vectorizer.transform(self.test_raw_docs)
        self.num_dimensions = max(self.doc_vectors[0].shape)

        if svd:
            svd_model = TruncatedSVD(n_components=self.number_of_components)    
            self.doc_vectors = svd_model.fit_transform(self.doc_vectors)
        
        self.doc_labels = self.label_encoder.fit_transform(self.train_raw_labels)

    def support_vector_machine(self):
        """
        :return: A SVC instance
        """
        svc = SVC(kernel='linear', random_state=self.random_state)

        return svc

    def random_forest(self):
        rf = RandomForestClassifier(random_state=self.random_state)
        return rf

    def multilayer_perceptron(self):
        """
        :return: A Multi-Layer Perceptron model instance
        """
        mlp = MLPClassifier(solver='lbfgs', alpha=0.7, random_state=self.random_state)

        return mlp

    def get_model(self):
        """
        Returns the appropriate model according to the input keyword (svm, rf, mlp)
        :return: The model
        """
        if self.classifier is "svm":
            clf = self.support_vector_machine()
        elif self.classifier is "rf":
            clf = self.random_forest()
        elif self.classifier is "mlp":
            clf = self.multilayer_perceptron()

        return clf

    def persist_model(self):
        """
        Serializes a model instance and persists it to disk
        :return: None
        """
        import pickle
        import os

        models_path = "../models"
        if not os.path.exists(models_path):
            os.mkdir(models_path)

        clf = self.get_model()
        clf.fit(self.doc_vectors, self.doc_labels)

        model_output_path = "{}/{}.pickle".format(models_path, self.classifier)

        with open(model_output_path, "w+") as out:
            pickle.dump(clf, out)

    def make_predictions(self):
        """
        Loads a serialized model and produces the testSet_categories.csv file with the predictions
        :return: None
        """
        model_path = "../models/mlp.pickle"
        if not os.path.exists(model_path):
            return

        with open(model_path) as model_file:
            model = pickle.loads(model_file.read())
            predictions = model.predict(self.test_vectors)
            labels = self.label_encoder.inverse_transform(predictions)
            result = reduce(lambda x, y: "{}\n{}".format(x, y),
                            map(lambda x: "{}\t{}".format(x[0], x[1]), zip(self.test_raw_docs_ids, labels)))

            with open("../testSet_categories.csv", "w+") as testset_out:
                testset_out.write(result)

    def test_model(self):
        """
        Evaluates the model using k-fold validation
        :return: The k-fold evaluation report
        """
        print "Classifier={}, Vectorizer={}, SVD={}".format(self.classifier, type(self.vectorizer).__name__, self.svd)

        clf = self.get_model()

        return self.k_fold_validation(clf)

    def compute_roc_auc_score(self, test_y, predicted):
        """
        Computes the ROC AUC score for multi-class classification, using the LabelBinarizer
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html
        :param test_y: The test data labels
        :param predicted: The predicted labels
        :return: The ROC AUC score value
        """
        from sklearn.preprocessing import LabelBinarizer
        lb = LabelBinarizer()
        lb.fit(test_y)

        y1 = lb.transform(test_y)
        y2 = lb.transform(predicted)

        return roc_auc_score(y1, y2)

    def k_fold_validation(self, model, k=10):
        """
        Performs a k-fold validation for all available evaluation metrics
        :param model: The model to evaluate
        :param k: The k (number of folds) value
        :return: The evaluation report
        """
        from sklearn.model_selection import KFold

        kfold = KFold(n_splits=k, shuffle=True)

        folds = kfold.split(X=self.doc_vectors, y=self.doc_labels)

        acc = 0.0
        prec = 0.0
        rec = 0.0
        f1 = 0.0
        auc_score = 0.0

        for train_idx, test_idx in folds:
            train_x = self.doc_vectors[train_idx]
            train_y = self.doc_labels[train_idx]
            test_x = self.doc_vectors[test_idx]
            test_y = self.doc_labels[test_idx]

            model.fit(train_x, train_y)

            predicted = model.predict(test_x)

            prec += precision_score(test_y, predicted, average='macro')
            rec += recall_score(test_y, predicted, average='macro')
            acc += accuracy_score(test_y, predicted)
            f1 += f1_score(test_y, predicted, average='macro')
            auc_score += self.compute_roc_auc_score(test_y, predicted)

        acc = acc / k
        prec = prec / k
        rec = rec / k
        f1 = f1 / k
        auc_score = auc_score / k

        report = EvaluationReport(accuracy=acc, precision=prec, recall=rec, f1=f1, auc=auc_score)

        return report

    def evaluate(self, test_y, predicted):
        prec = precision_score(test_y, predicted, average='macro')
        rec = recall_score(test_y, predicted, average='macro')
        acc = accuracy_score(test_y, predicted)
        f1 = f1_score(test_y, predicted, average='macro')
        auc_score = self.compute_roc_auc_score(test_y, predicted)

        report = EvaluationReport(accuracy=acc, precision=prec, recall=rec, f1=f1, auc=auc_score)

        return report

    def run_task(self):
        evaluation_report_json = self.test_model().__dict__

        print evaluation_report_json

        return evaluation_report_json
