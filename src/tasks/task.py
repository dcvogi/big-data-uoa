from abc import abstractmethod


class Task:
    def __init__(self, train, test):
        """
        Task constructor
        :param train: The training dataset
        :param test: The test dataset
        """
        self.train = train
        self.test = test

    @abstractmethod
    def run_task(self):
        """
        The main run_task method. This abstract method should be implemented in
        any Task sub-class implementation.
        :return: The result (if any)
        """
        pass

    def run(self):
        """
        This is a generic method that is invoked in order to trigger
        the Task execution
        :return: Whatever run_task returns
        """
        return self.run_task()
