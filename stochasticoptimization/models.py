from sklearn import pipeline, preprocessing
import numpy as np
from matplotlib import pyplot as plt, patches, lines


class StochasticSVM:
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
            self.make_plot(X[:, 1:3], y)
        return self

    def make_plot(self, X, y, grid_step=.01):
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
        plt.figure(figsize=(14, 9))
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green', marker='+', label='class 1')
        plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='D', label='class 0')
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title('Classification decision plane')
        legend_elements = [patches.Patch(facecolor='green', edgecolor='green', label='Class 1'),
                           patches.Patch(facecolor='red', edgecolor='red', label='Class 2'),
                           patches.Patch(facecolor='black', edgecolor='black', label='Separating plane')]
        plt.legend(handles=legend_elements)
        x1_min, x1_max = X[:, 0].min() - .1, X[:, 0].max() + .1
        x2_min, x2_max = X[:, 1].min() - .1, X[:, 1].max() + .1
        xx, yy = np.meshgrid(np.arange(x1_min, x1_max, grid_step),
                             np.arange(x2_min, x2_max, grid_step))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()], to_draw=True)
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, levels=[0.1], colors='black')
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
