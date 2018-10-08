
import numpy as np
import scipy.stats as ss
from numpy.polynomial.polynomial import polyvander2d, polyval2d
from collections import namedtuple

from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split
from sklearn.utils import resample


fitstats = namedtuple('stats', ['mse', 'r2', 'beta_variance', 'zscore', 'beta_low', 'beta_up'])


class Ridge2d(object):
    """
    Ridge regression for 2D polynoms of given degree

    zi = f(xi, yi) + espilon_i

    f(x, y) = sum beta_ij * x**i * y**j for i=0,1,...deg[0], j=0,1,..., deg[1]

    X_ij = x**i * y**j


    """

    def __init__(self, deg=(2, 2), lam=0, alpha=0.05, fulloutput=False):
        self.deg = deg
        self.lam = lam
        self.alpha = alpha
        self.coefficients = None
        self.fulloutput=fulloutput
        self._mean_X = ()
        self._mean_y = 0


    def fit(self, X, y):
        # ndim = len(self.deg)
        orders = [n + 1 for n in self.deg]
        order = np.prod(orders)
        self._mean_X = np.mean(X, axis=0, keepdims=True)
        self._mean_y = np.mean(y)
        X_ = X - self._mean_X
        y_ = y - self._mean_y

        x0 = X_[:,0].reshape(-1, 1)
        x1 = X_[:,1].reshape(-1, 1)
        xb_all = polyvander2d(x0.ravel(), x1.ravel(), deg=self.deg).reshape(-1, order)
        # xb has shape (n, order) where order = (deg[0] +1 ) * (deg[1] +1)
        # x.reshape(-1, 1)  # shape = (nx, ny)  => Change into shape (n, 1) where n = nx*ny
        # so that xb is:
        # xb_all = [np.ones(n, 1), x**1  ,   x**2, ...  , x**deg[0], x**1 * y**1, x**2 * y**1,..., x**deg[0] * y**1,... , x**deg[0]*y**deg[1]]
        xb = xb_all[:, 1:]  # drop the constant term
        xtx_inv = np.linalg.pinv(xb.T.dot(xb) + self.lam * np.eye(order-1))
        # beta = [beta_00, beta_10,  beta_20, ..., beta_01       beta_11,  beta_21, ..., beta_20, beta_21,  beta_22, ...]
        beta = np.vstack((0.,
                          xtx_inv.dot(xb.T).dot(y_.reshape(-1, 1)))).reshape(orders)
        self.coefficients = beta

        if self.fulloutput:
            # beta has shape (deg[0] +1, deg[1] +1)
            yhat = polyval2d(x0, x1, beta) + self._mean_y
            # This equal to evaluating the following sum:  sum beta_ij * x**i * y**j for i=0,...deg[0], j=0, ..., deg[1]

            mean_sqared_error = mse(y, yhat)  #
            n = y.size
            sigma =  n * mean_sqared_error / (n-order) # Eq. after Eq. 3.9 on pp 47 in Hastie etal.
            beta_covariance = xtx_inv * sigma # Eq. 3.10 on pp 47 in Hastie etal.  # Is it valid for ridge?? Maybe missing a correction term when lam>0
            beta_variance = np.diag(beta_covariance) # .reshape(orders)

            std_error = np.sqrt(beta_variance)
            z_score = beta / std_error
            # 1-alpha confidence interval for beta. Eq 3.14 in Hastie
            z_alpha = -ss.norm.ppf(self.alpha/2)  # inverse of the gaussian cdf function (ss.norm.cdf(-z_alpha)==alpha/2), cdf = cumulative density function
            beta_low = beta - z_alpha * std_error
            beta_up = beta + z_alpha * std_error

            self.stats = fitstats(mse=mean_sqared_error,
                                  r2=r_squared(y, yhat),
                                  beta_variance=beta_variance,
                                  zscore=z_score,
                                  beta_low=beta_low,
                                  beta_up=beta_up)
        return self

    def predict(self, X):
        X_  = X - self._mean_X
        return self._mean_y + polyval2d(X_[:,0], X_[:,1], self.coefficients).reshape(-1, 1)


class OLS2d(Ridge2d):
    """
    Ordinary Least Squares for 2D polynoms of degree 'deg'

    zi = f(xi, yi) + espilon_i

    f(x, y) = sum beta_ij * x**i * y**j for i=0,1,...deg[0], j=0,1,..., deg[1]

    X_ij = x**i * y**j


    """

    def __init__(self, deg=(2, 2), alpha=0.05, fulloutput=False):
        super(OLS2d, self).__init__(deg, 0, alpha, fulloutput)