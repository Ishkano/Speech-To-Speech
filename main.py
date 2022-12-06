from src import Parameters
from src import Preprocessing
from src import Model
from src import Run


class Controller(Parameters):

    def __init__(self):
        # Preprocessing pipeline
        self.data = self.prepare_data()

        # Initialize the model
        self.model = Model(Parameters)

        # Training - Evaluation pipeline
        Run().train(self.model, self.data, Parameters)

    @staticmethod
    def prepare_data():
        # Preprocessing pipeline
        pr = Preprocessing()
        return pr.get_preprocd()


if __name__ == '__main__':
    controller = Controller()
