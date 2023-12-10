import argparse
import logging
import signal
import time

import pandas as pd
from sklearn import metrics

from coreml.stochasticoptimization import models


class SignalHandler:
    shutdown = False

    def __init__(self):
        # Interrupt from keyboard (CTRL + C).
        signal.signal(signal.SIGINT, self.request_shutdown)
        # Termination signal.
        signal.signal(signal.SIGTERM, self.request_shutdown)

    def request_shutdown(self, *args):
        print('Shutdown was requested')
        self.shutdown = True

    def can_run(self):
        return not self.shutdown


signal_handler = SignalHandler()

urls = [
    'https://raw.githubusercontent.com/esokolov/'
    'ml-course-hse/master/2022-fall/homeworks-practice/homework-practice-05-trees/students.csv'
]

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '-lr',
    '--learning-rate',
    help='Learning rate for a model optimization',
    type=float
)
arg_parser.add_argument(
    '-d',
    '--degree',
    help='Polynomial\'s degree for a non-linear model optimization',
    type=int
)
arg_parser.add_argument(
    '-c',
    '--regularization',
    help='Regularization coefficient',
    type=float
)

logging.basicConfig(
    level=logging.INFO,
    filename='./stochasticoptimization_daemon.log',
    filemode='w',
    format='%(asctime)s %(message)s.'
)

if __name__ == '__main__':
    # Simulation of model training and inference
    parsed_args = arg_parser.parse_args()
    while signal_handler.can_run():
        for url in urls:
            df = pd.read_csv(url)
            sgd_svm = models.StochasticSVM(
                lr=parsed_args.learning_rate,
                degree=parsed_args.degree,
                C=parsed_args.regularization
            )
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            sgd_svm.fit(X, y)
            logging.info(f'Fitted by {url} dataset')

            predictions = []
            ground_truths = []
            for i in range(X.shape[0]):
                prediction = sgd_svm.predict(X[i].reshape(1, -1))
                ground_truth = y[i]
                logging.info(f'Predicted: {prediction} | Ground truth: {ground_truth}')
                predictions.append(prediction)
                ground_truths.append(ground_truth)
                time.sleep(0.1)
            total_accuracy = metrics.accuracy_score(ground_truths, predictions)
            logging.info(f'TOTAL ACCURACY: {total_accuracy}')
