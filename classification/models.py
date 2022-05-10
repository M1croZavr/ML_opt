import numpy as np
from onedimensionaloptimization import utils_onedimensionaloptimization as uo
from twovarextremas import plotting_3d
from sklearn import preprocessing, svm, pipeline
from matplotlib import pyplot as plt


def sigmoid(wx):
    return 1 / (1 + np.exp(-1 * wx))


class LogisticRegressionRidge:
    """
    Logistic regression with Ridge regularization by coefficient C(inverse of lambda)

    LogisticRegression fits a linear model with sigmoid function coefficients w = (w1, ..., wp)
    to minimize the log loss between the observed targets in the dataset.

    Parameters
    ----------
    lr -- float, default=0.001
        Learning rate in SGD
    eps -- float, default=0.00005
        Convergence condition coefficient in SGD
    C -- float, default=1
        Inverse of regularization coefficient
    plot -- bool, default=False
        Draw result plot or not

    Attributes
    ----------
    w -- array, weights of features found by SGD
    ...
    """
    def __init__(self, lr=0.001, eps=0.00005, C=1, plot=False):
        self.lr = lr
        self.eps = eps
        self.C = C
        self.plot = plot
        self.w = None

    def fit(self, X, y):
        """
        Fits logistic regression model by stochastic gradient descent.

        Parameters
        ----------
        X -- numpy ndarray, features data 2d array
        y -- numpy ndarray, targets data

        Returns
        -------
        self -- object, fitted model
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        indices = np.arange(X.shape[0])
        self.w = np.ones((X.shape[1], ))
        w_old = None
        while (w_old is None) or (np.sum(np.sqrt((self.w - w_old) ** 2)) > self.eps):
            np.random.shuffle(indices)
            X_sh = X[indices]
            y_sh = y[indices]
            w_old = self.w
            for i in range(X.shape[0]):
                self.w -= self.lr * np.array([(sigmoid(self.w.dot(X_sh[i])) - y_sh[i]) * X_sh[i, j] + (1 / self.C) * self.w[j] if j != 0
                                              else (sigmoid(self.w.dot(X_sh[i])) - y_sh[i]) * X_sh[i, j]
                                              for j in range(len(self.w))])
        if self.plot:
            self.make_plot(X[:, 1:3], y)
        return self

    def make_plot(self, X, y, grid_step=.01):
        plt.figure(figsize=(12, 8))
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green', marker='+', label='class 1')
        plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='D', label='class 0')
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title('Classification decision plane')
        plt.legend()
        x1_min, x1_max = X[:, 0].min() - .1, X[:, 0].max() + .1
        x2_min, x2_max = X[:, 1].min() - .1, X[:, 1].max() + .1
        xx, yy = np.meshgrid(np.arange(x1_min, x1_max, grid_step),
                             np.arange(x2_min, x2_max, grid_step))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, colors='black')
        plt.show()

    def predict_proba(self, X):
        """
        Makes probability predictions for X features array.

        Parameters
        ----------
        X -- array, features data 2d array

        Returns
        -------
        y -- array, array of predictions of regression model
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return sigmoid(np.sum(self.w * X, axis=1))

    def predict(self, X, threshold=0.5):
        """
        Makes predictions for X features array.

        Parameters
        ----------
        X -- array, features data 2d array
        threshold -- float, threshold for probability(default=0.5)

        Returns
        -------
        y -- array, array of predictions of regression model
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return (np.sum(self.w * X, axis=1) >= threshold).astype(np.float32)

    def __str__(self):
        if not(self.w is None):
            return 'y = ' + 'sigmoid(' + ' + '.join([f'{self.w[i]}x{i}' if i != 0 else f'{self.w[i]}' for i in range(len(self.w))]) + ')'
        else:
            return 'Fit model before calling __str__'


class LogisticRegressionLasso:
    """
    Logistic regression with Lasso regularization by coefficient C(inverse of lambda)

    LogisticRegression fits a linear model with sigmoid function coefficients w = (w1, ..., wp)
    to minimize the log loss between the observed targets in the dataset.

    Parameters
    ----------
    lr -- float, default=0.001
        Learning rate in SGD
    eps -- float, default=0.00005
        Convergence condition coefficient in SGD
    C -- float, default=1
        Inverse of regularization coefficient
    plot -- bool, default=False
        Draw result plot or not

    Attributes
    ----------
    w -- array, weights of features found by SGD
    ...
    """
    def __init__(self, lr=0.001, eps=0.00005, C=1, plot=False):
        self.lr = lr
        self.eps = eps
        self.C = C
        self.plot = plot
        self.w = None

    def fit(self, X, y):
        """
        Fits logistic regression model by stochastic gradient descent.

        Parameters
        ----------
        X -- numpy ndarray, features data 2d array
        y -- numpy ndarray, targets data

        Returns
        -------
        self -- object, fitted model
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        indices = np.arange(X.shape[0])
        self.w = np.ones((X.shape[1], ))
        w_old = None
        while (w_old is None) or (np.sum(np.sqrt((self.w - w_old) ** 2)) > self.eps):
            np.random.shuffle(indices)
            X_sh = X[indices]
            y_sh = y[indices]
            w_old = self.w
            for i in range(X.shape[0]):
                self.w -= self.lr * np.array([(sigmoid(self.w.dot(X_sh[i])) - y_sh[i]) * X_sh[i, j] + (1 / self.C) * uo.sign(self.w[j]) if j != 0
                                              else (sigmoid(self.w.dot(X_sh[i])) - y_sh[i]) * X_sh[i, j]
                                              for j in range(len(self.w))])
        if self.plot:
            self.make_plot(X[:, 1:3], y)
        return self

    def make_plot(self, X, y, grid_step=.01):
        plt.figure(figsize=(12, 8))
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green', marker='+', label='class 1')
        plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='D', label='class 0')
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title('Classification decision plane')
        plt.legend()
        x1_min, x1_max = X[:, 0].min() - .1, X[:, 0].max() + .1
        x2_min, x2_max = X[:, 1].min() - .1, X[:, 1].max() + .1
        xx, yy = np.meshgrid(np.arange(x1_min, x1_max, grid_step),
                             np.arange(x2_min, x2_max, grid_step))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, colors='black')
        plt.show()

    def predict_proba(self, X):
        """
        Makes probability predictions for X features array.

        Parameters
        ----------
        X -- array, features data 2d array

        Returns
        -------
        y -- array, array of predictions of regression model
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return sigmoid(np.sum(self.w * X, axis=1))

    def predict(self, X, threshold=0.5):
        """
        Makes predictions for X features array.

        Parameters
        ----------
        X -- array, features data 2d array
        threshold -- float, threshold for probability(default=0.5)

        Returns
        -------
        y -- array, array of predictions of regression model
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return (np.sum(self.w * X, axis=1) >= threshold).astype(np.float32)

    def __str__(self):
        if not(self.w is None):
            return 'y = ' + 'sigmoid(' + ' + '.join([f'{self.w[i]}x{i}' if i != 0 else f'{self.w[i]}' for i in range(len(self.w))]) + ')'
        else:
            return 'Fit model before calling __str__'


class SupportVectorClassifier:
    """
    Linear support vector classifier with Ridge and Lasso regularization by coefficient C(inverse of lambda)

    SupportVectorClassifier fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the hinge loss between the observed targets in the dataset without using dual optimization problem.

    Parameters
    ----------
    reg -- str, default=Ridge
        Which regularization to use (ridge or lasso)
    C -- float, default=1
        Inverse of regularization coefficient
    eps -- float, default=0.0005
        Convergence tolerance of optimization problem
    plot -- bool, default=False
        Draw result plot or not

    Attributes
    ----------
    pipeline -- sklearn.Pipeline
        Pipeline with support vector classifier model and scaler
    ...
    """
    def __init__(self, reg='Ridge', C=1, eps=0.0005, plot=False):
        if reg.lower() == 'ridge':
            self.reg = 'l2'
        elif reg.lower() == 'lasso':
            self.reg = 'l1'
        else:
            raise ValueError('Regularization must be one of {\'Ridge\', \'Lasso\'}')
        self.C = C
        self.eps = eps
        self.pipeline = pipeline.Pipeline([('Scaler', preprocessing.StandardScaler()),
                                           ('Estimator', svm.LinearSVC(penalty=self.reg,
                                                                       tol=self.eps,
                                                                       C=self.C,
                                                                       dual=False))])
        self.plot = plot

    def fit(self, X, y):
        """
        Fits linear support vector classifier model.

        Parameters
        ----------
        X -- numpy ndarray, features data 2d array
        y -- numpy ndarray, targets data

        Returns
        -------
        self -- object, fitted model
        """
        self.pipeline.fit(X, y)
        if self.plot:
            self.make_plot(X[:, :2], y)
        return self.pipeline

    def make_plot(self, X, y, grid_step=.01):
        plt.figure(figsize=(12, 8))
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green', marker='+', label='class 1')
        plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='D', label='class 0')
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title('Classification decision plane')
        plt.legend()
        x1_min, x1_max = X[:, 0].min() - .1, X[:, 0].max() + .1
        x2_min, x2_max = X[:, 1].min() - .1, X[:, 1].max() + .1
        xx, yy = np.meshgrid(np.arange(x1_min, x1_max, grid_step),
                             np.arange(x2_min, x2_max, grid_step))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, colors='black')
        plt.show()

    def predict(self, X):
        """
        Makes predictions for X features array.

        Parameters
        ----------
        X -- array, features data 2d array
        threshold -- float, threshold for probability(default=0.5)

        Returns
        -------
        y -- array, array of predictions of regression model
        """
        return self.pipeline.predict(X)

    def __str__(self):
        return 'y = ' + ' + '.join([f'{c}x{i}' if i != 0 else f'{c}'
                                    for i, c in enumerate(self.pipeline.named_steps['Estimator'].intercept_
                                                          + self.pipeline.named_steps['Estimator'].coef_[0])])


class SupportVectorClassifierDual:
    """
    Linear support vector classifier with Ridge and Lasso regularization by coefficient C(inverse of lambda)

    SupportVectorClassifier fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the hinge loss between the observed targets in the dataset using dual optimization problem.

    Parameters
    ----------
    reg -- str, default=Ridge
        Which regularization to use (ridge or lasso)
    C -- float, default=1
        Inverse of regularization coefficient
    eps -- float, default=0.0005
        Convergence tolerance of optimization problem
    plot -- bool, default=False
        Draw result plot or not

    Attributes
    ----------
    pipeline -- sklearn.Pipeline
        Pipeline with support vector classifier model and scaler
    ...
    """
    def __init__(self, reg='Ridge', C=1, eps=0.0005, plot=False):
        if reg.lower() == 'ridge':
            self.reg = 'l2'
        elif reg.lower() == 'lasso':
            self.reg = 'l1'
        else:
            raise ValueError('Regularization must be one of {\'Ridge\', \'Lasso\'}')
        self.C = C
        self.eps = eps
        self.pipeline = pipeline.Pipeline([('Scaler', preprocessing.StandardScaler()),
                                           ('Estimator', svm.LinearSVC(penalty=self.reg,
                                                                       tol=self.eps,
                                                                       C=self.C,
                                                                       dual=True))])
        self.plot = plot

    def fit(self, X, y):
        """
        Fits support vector classifier with dual.

        Parameters
        ----------
        X -- numpy ndarray, features data 2d array
        y -- numpy ndarray, targets data

        Returns
        -------
        self -- object, fitted model
        """
        self.pipeline.fit(X, y)
        if self.plot:
            self.make_plot(X[:, :2], y)
        return self.pipeline

    def make_plot(self, X, y, grid_step=.01):
        plt.figure(figsize=(12, 8))
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green', marker='+', label='class 1')
        plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='D', label='class 0')
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title('Classification decision plane')
        plt.legend()
        x1_min, x1_max = X[:, 0].min() - .1, X[:, 0].max() + .1
        x2_min, x2_max = X[:, 1].min() - .1, X[:, 1].max() + .1
        xx, yy = np.meshgrid(np.arange(x1_min, x1_max, grid_step),
                             np.arange(x2_min, x2_max, grid_step))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, colors='black')
        plt.show()

    def predict(self, X):
        """
        Makes predictions for X features array.

        Parameters
        ----------
        X -- array, features data 2d array
        threshold -- float, threshold for probability(default=0.5)

        Returns
        -------
        y -- array, array of predictions of regression model
        """
        return self.pipeline.predict(X)

    def __str__(self):
        return 'y = ' + ' + '.join([f'{c}x{i}' if i != 0 else f'{c}'
                                    for i, c in enumerate(self.pipeline.named_steps['Estimator'].intercept_
                                                          + self.pipeline.named_steps['Estimator'].coef_[0])])
