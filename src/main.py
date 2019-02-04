import pandas as pd
import os

from tasks import WordCloudTask, WordCloudTask2, AnalyticsTask, DuplicatesTask


class Main:
  def __init__(self):
    results_path = "../results"
    if not os.path.exists(results_path):
        os.mkdir("../results")

  test_set = pd.read_csv("../datasets/test_set.csv", sep="\t")
  train_set = pd.read_csv("../datasets/train_set.csv", sep="\t")

  def wordcloud(self):
    wc = WordCloudTask2(self.train_set, self.test_set)
    wc.run()

  def duplicates(self, vectorizer, threshold, stop_words):
    dp = DuplicatesTask(self.train_set, self.test_set, vectorizer=vectorizer, threshold=threshold, stop_words=stop_words)
    dp.run()

  def classification(self, clf, vectorizer, mf=100, svd=False, components_perc=None):
    analytics = AnalyticsTask(self.train_set, self.test_set, classifier=clf, vectorizer=vectorizer, test_size=0.2,
                              svd=svd, max_features=mf)

    return analytics.run()

  def export_evaluation_report(self):
    max_features = 1000

    svm_bow = self.classification("svm", "bow", mf=max_features)
    rf_bow = self.classification("rf", "bow", mf=max_features)
    svm_bow_svd = self.classification("svm", "bow", mf=max_features, svd=True)
    rf_bow_svd = self.classification("rf", "bow", mf=max_features)
    svm_w2v = self.classification("svm", "w2v", mf=max_features)
    rf_w2v = self.classification("rf", "w2v", mf=max_features)
    mlp_hash = self.classification("mlp", "hash", mf=max_features)

    reports = [svm_bow, rf_bow, svm_bow_svd, rf_bow_svd, svm_w2v, rf_w2v, mlp_hash]
    evaluation_10fold_csv = "Statistic Measure, SVM(BoW), Random Forest(BoW), SVM(SVD), Random Forest(SVD), SVM(W2v), " \
                            "Random Forest(W2V), Our Method\n" \
                            "Accuracy,{}\n" \
                            "Precision,{}\n" \
                            "Recall,{}\n" \
                            "F-Measure,{}\n" \
                            "AUC,{}".format(','.join(map(lambda x: str(x['accuracy']), reports)),
                                            ','.join(map(lambda x: str(x['precision']), reports)),
                                            ','.join(map(lambda x: str(x['recall']), reports)),
                                            ','.join(map(lambda x: str(x['f1']), reports)),
                                            ','.join(map(lambda x: str(x['auc']), reports)))


    with open("../results/Evaluation_10fold.csv", "w+") as eval_out:
        eval_out.write(evaluation_10fold_csv)




main = Main()
main.export_evaluation_report()
main.wordcloud()
main.duplicates("tfidf", 0.7, "english")