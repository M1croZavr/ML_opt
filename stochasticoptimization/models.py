from sklearn import pipeline, preprocessing
import numpy as np
from matplotlib import pyplot as plt, patches, lines
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.inspection import DecisionBoundaryDisplay


class StochasticSVM(BaseEstimator, ClassifierMixin):
    """
    Support vector classifier optimized by Stochastic Gradient Descent

    StochasticSVM fits a model with coefficients w = (w1, ..., wp) and b
    to minimize the hinge loss between the observed targets in the dataset.

    Parameters
    ----------
    lr -- float, default=0.05
        Learning rate for SGD
    C -- float, default=0.1
        Tradeoff between gap width and number of gap violations
    degree -- int, default=None
        Defines maximum degree of generating polynomial features
    eps -- float, default=1e-6
        Convergence tolerance of optimization problem
    plot -- bool, default=False
        Draw result plot or not

    Attributes
    ----------
    w -- np.ndarray
        Learnable parameters of the model
    ...
    """
    def __init__(self, lr=0.05, C=1, degree=None, eps=1e-6, plot=False):
        self.pipeline = pipeline.Pipeline([('Polynomial features', preprocessing.PolynomialFeatures(1 if degree is None else degree, include_bias=False)),
                                           ('Standart Scaler', preprocessing.StandardScaler())])
        self.lr = lr
        self.C = C
        self.eps = eps
        self.plot = plot
        self.w = None
        self.X_ = None
        self.y_ = None

    def fit(self, X, y):
        """
        Optimize binary support vector classifier by SGD.

        Parameters
        ----------
        X -- numpy ndarray, features data 2d array
        y -- numpy ndarray, targets data

        Returns
        -------
        self -- object, fitted model
        """
        self.X_ = X
        self.y_ = y
        X = self.pipeline.fit_transform(X)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        indices = np.arange(X.shape[0])
        self.w = np.random.randn(*(X.shape[1], ))
        w_old = None
        while (w_old is None) or (np.sum(np.sqrt((self.w - w_old) ** 2)) > self.eps):
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            w_old = self.w
            for i in range(X.shape[0]):
                t_i = -1 if y_shuffled[i] == 0 else 1
                if t_i * (self.w.dot(X_shuffled[i])) >= 1:
                    self.w -= self.lr * self.w
                else:
                    self.w -= self.lr * np.array([self.w[j] - self.C * X_shuffled[i, j] * t_i
                                                  for j in range(len(self.w))])
        if self.plot:
            self.make_plot()
        return self

    def make_plot(self):
        """
        Make plot for binary classification.

        Parameters
        ----------
        X -- numpy ndarray, features data 2d array
        y -- numpy ndarray, targets data
        grid_step -- float, what step to use in arange points to use in drawing contour

        Returns
        -------
        None
        """
        fig, ax = plt.subplots(figsize=(18, 9))
        display = DecisionBoundaryDisplay.from_estimator(self,
                                                         self.X_,
                                                         response_method='predict',
                                                         xlabel='Feature1',
                                                         ylabel='Feature2',
                                                         alpha=0.55,
                                                         ax=ax)
        display.ax_.scatter(self.X_[self.y_ == 1, 0], self.X_[self.y_ == 1, 1], c='green', marker='+', label='class 1')
        display.ax_.scatter(self.X_[self.y_ == 0, 0], self.X_[self.y_ == 0, 1], c='red', marker='D', label='class 0')
        display.ax_.set_title('Classification decision plane')
        legend_elements = [patches.Patch(facecolor='green', edgecolor='green', label='Class 1'),
                           patches.Patch(facecolor='red', edgecolor='red', label='Class 2'),
                           patches.Patch(facecolor='black', edgecolor='black', label='Separating plane')]
        display.ax_.legend(handles=legend_elements)
        plt.show()

    def predict(self, X, to_draw=False):
        """
        Makes predictions for X features array.

        Parameters
        ----------
        X -- array, features data 2d array
        threshold -- float

        Returns
        -------
        y -- array, array of predictions of regression model
        """
        X = self.pipeline.transform(X)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        if to_draw:
            return np.array([self.w.dot(X[i]) for i in range(X.shape[0])])
        else:
            return np.array([1 if self.w.dot(X[i]) >= 0 else 0 for i in range(X.shape[0])])

    def __str__(self):
        """
        String representation of model. Weights of all the features.
        """
        return 'y = ' + ' + '.join([f'{c}x{i}' if i != 0 else f'{c}'
                                    for i, c in enumerate(self.w)])
