import pandas as pd

from tasks import WordCloudTask


class Main:
  test_set = pd.read_csv("../datasets/test_set.csv", sep="\t")
  train_set = pd.read_csv("../datasets/train_set.csv", sep="\t")

  @staticmethod
  def wordcloud(train, test):
    wc = WordCloudTask(train, test)
    wc.run()

  @staticmethod
  def duplicates(train, test):
    pass

  @staticmethod
  def classification(train, test):
    pass