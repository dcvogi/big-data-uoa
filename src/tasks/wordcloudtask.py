from task import Task

# Third-party libs
from wordcloud import WordCloud, STOPWORDS
from matplotlib import pyplot as plt


class WordCloudTask(Task):
    def __init__(self, train, test):
        Task.__init__(self, train, test)

    def run_task(self):
        text = ' '.join(self.train['Content'].values)
        stop_words = set(STOPWORDS)

        # WordCloud generation
        wc = WordCloud(max_words=1000, margin=10, stopwords=stop_words, random_state=1).generate(text)
        plt.imshow(wc)
        plt.show()
