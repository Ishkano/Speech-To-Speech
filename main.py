from src import Parameters
from src import Preprocessing
from src import Model
from src import Run


class Controller(Parameters):

    def __init__(self):
        # Preprocessing pipeline
        self.data_loaders = Preprocessing(Parameters)

        # Initialize the model
        self.model = Model(Parameters)

        # Training - Evaluation pipeline
        Run().train(self.model, self.data_loaders, Parameters)


if __name__ == '__main__':
    controller = Controller()
