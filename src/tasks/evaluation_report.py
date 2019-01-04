class EvaluationReport:
    def __init__(self, accuracy, precision, recall, f1, auc):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.auc = auc