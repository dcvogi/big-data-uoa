from task import Task
import numpy as np
from PIL import Image
# Third-party libs
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from matplotlib import pyplot as plt


class WordCloudTask2(Task):
    def __init__(self, train, test):
        Task.__init__(self, train, test)

    def run_task(self):

        pivot = self.train.pivot(index='Id', columns='Category', values='Content').apply(
            lambda x: x.str.cat(sep=' '))

        for index in pivot.index:
            # WordCloud generation
            mask = np.array(Image.open('../datasets/' + str(index) + '.png'))
            wc = WordCloud(background_color='white', mask=mask, stopwords=STOPWORDS)
            wc.generate(pivot[index])

            # create coloring from image
            image_colors = ImageColorGenerator(mask)
            fig, axes = plt.subplots(figsize=(12,8))
            plt.imshow(wc.recolor(), interpolation='bilinear')
            plt.axis("off")
            plt.savefig('../results/Wordcloud_'  + str(index) + '.png')