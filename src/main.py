import pandas as pd

from tasks import WordCloudTask, AnalyticsTask, DuplicatesTask
from vectorizer import EmbeddingsVectorizer


class Main:
  test_set = pd.read_csv("../datasets/test_set.csv", sep="\t")
  train_set = pd.read_csv("../datasets/train_set.csv", sep="\t").head(1000)

  def wordcloud(self):
    wc = WordCloudTask(self.train_set, self.test_set)
    wc.run()

  def duplicates(self, vectorizer, threshold, stop_words):
    dp = DuplicatesTask(self.train_set, self.test_set, vectorizer=vectorizer, threshold=threshold, stop_words=stop_words)
    dp.run()

  def classification(self, clf, vectorizer):
    analytics = AnalyticsTask(self.train_set, self.test_set, classifier=clf, vectorizer=vectorizer, test_size=0.3)

    analytics.run()


main = Main()
main.wordcloud()
main.classification("svm", "bow")
main.classification("svm", "tfidf")
main.classification("rf", "bow")
main.classification("rf", "tfidf")
main.classification("mlp", "tfidf")
main.duplicates("tfidf", 0.7, "english")