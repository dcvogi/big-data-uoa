import pandas as pd

from tasks import WordCloudTask

test_set = pd.read_csv("../datasets/test_set.csv", sep="\t")
train_set = pd.read_csv("../datasets/train_set.csv", sep="\t")


def wordcloud(train, test):
  wc = WordCloudTask(train, test)
  wc.run()


def duplicates(train, test):
  pass


def classification(train, test):
  pass