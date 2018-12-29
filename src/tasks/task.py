from abc import abstractmethod


class Task:
    def __init__(self, train, test):
        self.train = train
        self.test = test

    @abstractmethod
    def run_task(self):
        pass

    def run(self):
        self.run_task()
