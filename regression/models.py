import numpy as np
from onedimensionaloptimization import utils_onedimensionaloptimization as uo
from twovarextremas import plotting_3d
from sklearn import preprocessing


class LinearRegression:
    """
    Least squares linear regression with l1, l2, student regularization

    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Parameters
    ----------
    reg -- str, default=None
        Regularization l1, l2, student or None to make regression without regularization
    lr -- float, default=0.01
        Learning rate in SGD
    eps -- float, default=0.05
        Convergence condition coefficient in SGD
    c -- float, default=0.01
        Regularization coefficient if regularization is provided
    plot -- bool, default=False
        Draw result plot or no

    Attributes
    ----------
    w -- array, weights of features found by SGD
    ...
    """
    def __init__(self, reg=None, lr=0.01, eps=0.005, c=0.01, plot=False):
        self.reg = reg
        self.eps = eps
        self.c = c
        self.lr = lr
        self.plot = plot
        self.w = None

    def fit(self, X, y):
        """
        Fits linear regression model with specified regularization by stochastic gradient descent.

        Parameters
        ----------
        X -- array, features data 2d array
        y -- array, targets data

        Returns
        -------
        self -- object, fitted model
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y = y.copy()
        indices = np.arange(X.shape[0])
        self.w = np.ones((X.shape[1], ))
        w_old = None
        while (w_old is None) or (np.sum(np.sqrt((self.w - w_old) ** 2)) > self.eps):
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            w_old = self.w
            for i in range(X.shape[0]):
                if self.reg is None:
                    self.w -= self.lr * np.array([(self.w.dot(X[i]) - y[i]) * X[i, j] for j in range(len(self.w))])
                elif self.reg == 'l2':
                    self.w -= self.lr * np.array([(self.w.dot(X[i]) - y[i]) * X[i, j] + self.c * self.w[j]
                                                  for j in range(len(self.w))])
                elif self.reg == 'l1':
                    self.w -= self.lr * np.array([(self.w.dot(X[i]) - y[i]) * X[i, j] + self.c * uo.sign(self.w[j])
                                                  for j in range(len(self.w))])
        if self.plot:
            if X.shape[1] > 3:
                print('Cannot draw a plot with dimension > 2')
            elif X.shape[1] == 3:
                x1_mesh, x2_mesh = np.meshgrid(np.linspace(min(X[:, 1]), max(X[:, 1]), 25),
                                               np.linspace(min(X[:, 2]), max(X[:, 2]), 25))
                y_mesh = np.array([[self.predict(np.array([[xx1, xx2]]))[0] for xx1, xx2 in zip(x1_i, x2_i)]
                                   for x1_i, x2_i in zip(x1_mesh, x2_mesh)])
                plotting_3d.regression_3d(X[:, 1:], y, x1_mesh, x2_mesh, y_mesh)
            elif X.shape[1] == 2:
                lin = np.linspace(min(X[:, 1]), max(X[:, 1]), 50)
                y_lin = np.array([self.predict(np.array([[x]]))[0] for x in lin])
                plotting_3d.regression_2d(X[:, 1:], y, lin, y_lin)

        return self

    def predict(self, X):
        """
        Makes predictions for X features array.

        Parameters
        ----------
        X -- array, features data 2d array

        Returns
        -------
        y -- array, array of predictions of regression model
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return np.sum(self.w * X, axis=1)

    def __str__(self):
        if not(self.w is None):
            return 'y = ' + ' + '.join([f'{self.w[i]}x{i}' if i != 0 else f'{self.w[i]}' for i in range(len(self.w))])
        else:
            return 'Fit model before calling __str__'


class PolynomialRegression:
    """
    Least squares polynomial regression with l1, l2, student regularization

    PolynomialRegression fits a polynomial model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Parameters
    ----------
    reg -- str, default=None
        Regularization l1, l2, student or None to make regression without regularization
    degree -- int, default=1
        Defines maximum degree of generating polynomial features
    lr -- float, default=0.01
        Learning rate in SGD
    eps -- float, default=0.05
        Convergence condition coefficient in SGD
    c -- float, default=0.01
        Regularization coefficient if regularization provided
    plot -- bool, default=False
        Draw result plot or no

    Attributes
    ----------
    w -- array, weights of features found by SGD
    poly_transformer -- sklearn.preprocessing._polynomial.PolynomialFeatures, Polynomial features' generator
    normalizer -- sklearn.preprocessing._polynomial.StandardScaler, Scaling features
    """
    def __init__(self, reg=None, degree=1, lr=0.01, eps=0.005, c=0.01, plot=False):
        self.reg = reg
        self.degree = degree
        self.lr = lr
        self.eps = eps
        self.c = c
        self.plot = plot
        self.w = None
        self.poly_transformer = preprocessing.PolynomialFeatures(degree=self.degree)
        self.normalizer = preprocessing.StandardScaler()

    def fit(self, X, y):
        """
        Fits polynomial regression model with specified regularization by stochastic gradient descent.

        Parameters
        ----------
        X -- array, features data 2d array
        y -- array, targets data

        Returns
        -------
        self -- object, fitted model
        """
        X_poly = self.poly_transformer.fit_transform(X)
        X_poly = np.hstack((X_poly[:, 0].reshape(-1, 1), self.normalizer.fit_transform(X_poly[:, 1:])))
        y_poly = y.copy()
        indices = np.arange(X_poly.shape[0])
        self.w = np.ones((X_poly.shape[1], ))
        w_old = None
        while (w_old is None) or (np.sum(np.sqrt((self.w - w_old) ** 2)) > self.eps):
            np.random.shuffle(indices)
            X_poly = X_poly[indices]
            y_poly = y_poly[indices]
            w_old = self.w
            for i in range(X_poly.shape[0]):
                if self.reg is None:
                    self.w -= self.lr * np.array([(self.w.dot(X_poly[i]) - y_poly[i]) * X_poly[i, j]
                                                  for j in range(len(self.w))])
                elif self.reg == 'l2':
                    self.w -= self.lr * np.array([(self.w.dot(X_poly[i]) - y_poly[i]) * X_poly[i, j] + self.c * self.w[j]
                                                  for j in range(len(self.w))])
                elif self.reg == 'l1':
                    self.w -= self.lr * np.array([(self.w.dot(X_poly[i]) - y_poly[i]) * X_poly[i, j] + self.c * uo.sign(self.w[j])
                                                  for j in range(len(self.w))])

        if self.plot:
            if X.shape[1] > 2:
                print('Cannot draw a plot with dimension > 2')
            elif X.shape[1] == 2:
                x1_mesh, x2_mesh = np.meshgrid(np.linspace(min(X[:, 0]), max(X[:, 0]), 25),
                                               np.linspace(min(X[:, 1]), max(X[:, 1]), 25))
                y_mesh = np.array([[self.predict(np.array([[xx1, xx2]]))[0] for xx1, xx2 in zip(x1_i, x2_i)]
                                   for x1_i, x2_i in zip(x1_mesh, x2_mesh)])
                plotting_3d.regression_3d(X, y, x1_mesh, x2_mesh, y_mesh)
            elif X.shape[1] == 1:
                lin = np.linspace(min(X[:, 0]), max(X[:, 0]), 50)
                y_lin = np.array([self.predict(np.array([[x]]))[0] for x in lin])
                plotting_3d.regression_2d(X, y, lin, y_lin)

        return self

    def predict(self, X):
        """
        Makes predictions for X features array.

        Parameters
        ----------
        X -- array, features data 2d array

        Returns
        -------
        y -- array, array of predictions of regression model
        """
        X = self.poly_transformer.transform(X)
        X = np.hstack((X[:, 0].reshape(-1, 1), self.normalizer.transform(X[:, 1:])))
        return np.sum(self.w * X, axis=1)

    def __str__(self):
        if not(self.w is None):
            return 'y = ' + ' + '.join([f'{self.w[i]}x{i}' if i != 0 else f'{self.w[i]}' for i in range(len(self.w))])
        else:
            return 'Fit model before calling __str__'


class ExpRegression:
    """
    Least squares exponential regression with l1, l2, student regularization

    ExpRegression fits a polynomial model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Parameters
    ----------
    reg -- str, default=None
        Regularization l1, l2, student or None to make regression without regularization
    lr -- float, default=0.01
        Learning rate in SGD
    eps -- float, default=0.05
        Convergence condition coefficient in SGD
    c -- float, default=0.01
        Regularization coefficient if regularization provided
    plot -- bool, default=False
        Draw result plot or no

    Attributes
    ----------
    w -- array, weights of features found by SGD
    poly_transformer -- sklearn.preprocessing._polynomial.PolynomialFeatures, Polynomial features generator
    """
    def __init__(self, reg=None, lr=0.01, eps=0.05, c=0.01, plot=False):
        self.reg = reg
        self.eps = eps
        self.c = c
        self.lr = lr
        self.plot = plot
        self.w = None

    def fit(self, X, y):
        """
        Fits linear regression model with specified regularization by stochastic gradient descent.

        Parameters
        ----------
        X -- array, features data 2d array
        y -- array, targets data

        Returns
        -------
        self -- object, fitted model
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y = np.log(y.copy())
        if self.reg is None:
            self.w = (np.linalg.inv(X.T @ X) @ X.T @ y.reshape(-1, 1)).flatten()
        elif self.reg in ('l2', 'l1'):
            reg_matrix = self.c * np.diag(np.array([0] + [1 for _ in range(X.shape[1] - 1)]))
            self.w = (np.linalg.inv(X.T @ X + reg_matrix) @ X.T @ y.reshape(-1, 1)).flatten()
        else:
            print('No such regularization')
            return None
        if self.plot:
            if X.shape[1] > 3:
                print('Cannot draw a plot with dimension > 2')
            elif X.shape[1] == 3:
                x1_mesh, x2_mesh = np.meshgrid(np.linspace(min(X[:, 1]), max(X[:, 1]), 25),
                                               np.linspace(min(X[:, 2]), max(X[:, 2]), 25))
                y_mesh = np.array([[self.predict(np.array([[xx1, xx2]]))[0] for xx1, xx2 in zip(x1_i, x2_i)]
                                   for x1_i, x2_i in zip(x1_mesh, x2_mesh)])
                plotting_3d.regression_3d(X[:, 1:], np.exp(y), x1_mesh, x2_mesh, y_mesh)
            elif X.shape[1] == 2:
                lin = np.linspace(min(X[:, 1]), max(X[:, 1]), 50)
                y_lin = np.array([self.predict(np.array([[x]]))[0] for x in lin])
                plotting_3d.regression_2d(X[:, 1:], np.exp(y), lin, y_lin)

        return self

    def predict(self, X):
        """
        Makes predictions for X features array.

        Parameters
        ----------
        X -- array, features data 2d array

        Returns
        -------
        y -- array, array of predictions of regression model
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return np.exp(np.sum(self.w * X, axis=1))

    def __str__(self):
        if not(self.w is None):
            return 'y = ' + ' * '.join([f'{np.exp(self.w[i])} ** x{i}' if i != 0 else f'{np.exp(self.w[i])}' for i in range(len(self.w))])
        else:
            return 'Fit model before calling __str__'
