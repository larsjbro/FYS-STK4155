
'''
Created on 27. sep. 2018


@author: ljb


The best way to present these confidence intervals is to nail down say the best model.
If you compute the MSE for the different approximations (polynomials), you may find a
behavior like that of fig 5 of Mehta et al, see https://arxiv.org/pdf/1803.08823.pdf,
that is a model where the MSE is at its minimum. If not, I would pick the model with
lowest MSE (and possibly best R2 score) and tell the reader that this is my recommended
model. Then I would present the parameters beta  for the best model (limit yourself to that)
and include the confidence intervals for that model only. Else you'll end up swamping the
report with tons of data. Based on your calculations it is you who recommends the best model,
with its pros and cons.
'''
import pandas as pd
import numpy as np
import scipy.stats as ss
# from random import random, seed
from numpy.polynomial.polynomial import polyvander2d, polyval2d
from collections import namedtuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split
from sklearn.utils import resample


fitstats = namedtuple('stats', ['mse', 'r2', 'beta_variance', 'zscore', 'beta_low', 'beta_up', 'data'])
fitstat2 = namedtuple('stats2', ['prediction_r2_score',
                                 'prediction_error',
                                 'average_bias_squared',
                                 'average_variance'])



def franke_function(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2)**2) - 0.25 * ((9 * y - 2)**2))
    term2 = 0.75 * np.exp(-((9 * x + 1)**2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7)**2 / 4.0 - 0.25 * ((9 * y - 3)**2))
    term4 = -0.2 * np.exp(-(9 * x - 4)**2 - (9 * y - 7)**2)
    return term1 + term2 + term3 + term4


def r_squared(y, yhat):
    mean_y = np.mean(y)
    return 1.0 - mse(y, yhat)/mse(y, mean_y)


def mse(y, yhat):
    return np.mean((y-yhat)**2)


def cross_validate_prediction_error(X, y, fitter=None, k=5):
    """ K fold cross validation of prediction error
    """
    Xi, yi = resample(X, y, replace=False)  # Shuffle the dataset randomly.
    n = len(yi)

    split = n // k  # size

    indices = np.arange(split * k).reshape(k, split)
    prediction_error = []
    for i, test_indices in enumerate(indices):
        train_indices = np.hstack((indices[:i].ravel(),
                                   indices[:i+1].ravel()))
        fitter.fit(Xi[train_indices,: ], yi[train_indices,:])
        yhat = fitter.predict(Xi[test_indices,:])
        prediction_error.append(mse(y[test_indices, :], yhat))
    return np.mean(prediction_error)


def bootstrap_bias_variance(x, y, model, n_bootstraps=200, test_size=0.2, plot=True):
    # Hold out some test data that is never used in training.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    # The following (m x n_bootstraps) matrix holds the column vectors y_pred
    # for each bootstrap iteration.
    y_pred = np.empty((y_test.shape[0], n_bootstraps))
    for i in range(n_bootstraps):
        x_, y_ = resample(x_train, y_train)

        # Evaluate the new model on the same test data each time.
        y_pred[:, i] = model.fit(x_, y_).predict(x_test).ravel()

    # Note: Expectations and variances taken w.r.t. different training
    # data sets, hence the axis=1. Subsequent means are taken across the test data
    # set in order to obtain a total value, but before this we have average_error/average_bias_squared/average_variance
    # calculated per data point in the test set.
    # Note 2: The use of keepdims=True is important in the calculation of average_bias_squared as this
    # maintains the column vector form. Dropping this yields very unexpected results.


    mean_y = np.mean(y_test)

    errors = np.mean((y_test - y_pred) ** 2, axis=1, keepdims=True)
    bias_sqared = (y_test - np.mean(y_pred, axis=1, keepdims=True)) ** 2
    variances = np.var(y_pred, axis=1, keepdims=True)

    average_r2 =  1.0 - np.mean(errors)/mse(y_test, mean_y)
    average_error = np.mean( errors )
    average_bias_squared = np.mean( bias_sqared )
    average_variance = np.mean( variances )

#     print(f'R2: {average_r2:0.2g}')
#     print(f'Error: {average_error:0.2g}')
#     print(f'Bias^2: {average_bias_squared:0.2g}', )
#     print(f'Var: {average_variance:0.2g}', )
#     print('{:0.2g} >= {:0.2g} + {:0.2g} = {:0.2g}'.format(average_error, average_bias_squared,
#                                       average_variance, average_bias_squared+average_variance))
    if plot:
        plt.plot(x[::5, :], y[::5, :], label='f(x)')
        plt.scatter(x_test, y_test, label='Data points')
        plt.scatter(x_test, np.mean(y_pred, axis=1), label='Pred')
        plt.legend()

    stats = fitstat2(prediction_r2_score=average_r2,
                     prediction_error=average_error,
                     average_bias_squared=average_bias_squared,
                     average_variance=average_variance)
    return stats


class Lasso2d(object):
    def __init__(self, deg=(2, 2), lam=0, alpha=0.05, fulloutput=False):
        self.deg = deg
        self.lam = lam
        self.alpha = alpha
        self.coefficients = None
        self.fulloutput=fulloutput
        self.model = make_pipeline(PolynomialFeatures(self.deg[0]),
                                   Lasso(alpha=self.lam, normalize=True, max_iter=8000))
    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


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
            beta_0 = beta.ravel()[1:]
            z_score = beta_0 / std_error
            # 1-alpha confidence interval for beta. Eq 3.14 in Hastie
            low_alpha2 = self.alpha/2
            hi_alpha2 = 1-low_alpha2
            z_alpha = -ss.norm.ppf(low_alpha2)  # inverse of the gaussian cdf function (ss.norm.cdf(-z_alpha)==alpha/2), cdf = cumulative density function
            beta_low = beta_0 - z_alpha * std_error
            beta_up = beta_0 + z_alpha * std_error

            data = pd.DataFrame(np.vstack((beta_0,
                                    np.sqrt(beta_variance.ravel()),
                                    z_score.ravel(),
                                    beta_low.ravel(),
                                    beta_up.ravel())).T, columns=['coef', 'std', 'z_score', f'{low_alpha2:.3f}', f'{hi_alpha2:.3f}'])
            self.stats = fitstats(mse=mean_sqared_error,
                                  r2=r_squared(y, yhat),
                                  beta_variance=beta_variance,
                                  zscore=z_score,
                                  beta_low=beta_low,
                                  beta_up=beta_up,
                                  data=data)
        return self

    def predict(self, X):
        X_  = X - self._mean_X
        return self._mean_y + polyval2d(X_[:,0], X_[:,1], self.coefficients).reshape(-1, 1)

    def summary(self):
        data = self.stats.data
        print('Dep. Variable:                      y   R-squared:                       {:0.3f}'.format(self.stats.r2))
        print('                                        MSE:                       {:0.3f}'.format(self.stats.mse))

        # print(data)
        print(data.to_string())
        print(data.to_latex())


class OLS2d(Ridge2d):
    """
    Ordinary Least Squares for 2D polynoms of degree 'deg'

    zi = f(xi, yi) + espilon_i

    f(x, y) = sum beta_ij * x**i * y**j for i=0,1,...deg[0], j=0,1,..., deg[1]

    X_ij = x**i * y**j


    """

    def __init__(self, deg=(2, 2), alpha=0.05, fulloutput=False):
        super(OLS2d, self).__init__(deg, 0, alpha, fulloutput)


def fit_lasso2d(X_, z_, z0_, deg, lam):
    fitter = make_pipeline(PolynomialFeatures(deg[0]), Lasso(alpha=lam,
                                                             normalize=True,
                                                             max_iter=8000))
    return _fit_2d(X_, z_, z0_, fitter)


def fit_ridge2d(X_, z_, z0_, deg, lam):
    # fitter = make_pipeline(PolynomialFeatures(deg[0]), Ridge(alpha=lam))
    fitter = Ridge2d(deg=deg, lam=lam, alpha=0.05, fulloutput=False)
    return _fit_2d(X_, z0_, z_, fitter)


def _fit_2d(X_, z_, z0_, fitter):
    stats = bootstrap_bias_variance(X_, z_, fitter, n_bootstraps=200, test_size=0.2, plot=False)

    return stats


    shape = z0_.shape

    split = shape[0] // 5
    # prepare bootstrap sample
    (x_train, x_test,
     z_train, z_test,
     z0_train, z0_test)  = train_test_split(X_, z_, z0_, test_size=split)

    test_data = X_[:split],  z_[:split]
    train_data = X_[split:],  z_[split:]

    prediction_error_from_cross_validation = cross_validate_prediction_error(x_train,
                                                                             z_train, fitter, k=5)

    fitter.fit(x_train, z_train)

    # fit_fun = partial(ridge2d, deg=deg, lam=lam)
    # fit_fun = partial(lasso2d, deg=deg, lam=lam)

    # fit_fun = partial(ols2d, deg=deg)  # lam=0
    # beta_avg, beta_var = bootstrap(*train_data, fun=fit_fun)

    zpredict = fitter.predict(x_test)
    prediction_error = mse(z_test, zpredict)
    prediction_r2_score = r_squared(z_test, zpredict)
    # print('R2_cv:{:0.2g}'.format(prediction_r2_score))

    average_bias_squared = mse(z0_test, zpredict)
    stats = fitstat2(prediction_r2_score=r_squared(test_data[1], zpredict),
                     prediction_error=prediction_error,
                     average_bias_squared=average_bias_squared,
                     average_variance=prediction_error - average_bias_squared)

#     print('mean_squared_error: {:.2g}'.format(prediction_error))
#     print('R2: {:.2f}'.format(prediction_r2_score))
#     print('Sigma^2: {:.2g}'.format(sigma ** 2))
#     print('Average Bias**2: {:.2g}'.format(average_bias_squared))
#     print('Average Variance: {:.2g}'.format(average_variance))

    return stats


# def fit_ridge2d(X_, z0_, z_, deg, lam, sigma):
#     shape = z0_.shape
#
#     split = shape[0] // 5
#     # prepare bootstrap sample
#     test_data = X_[:split],  z_[:split]
#     train_data = X_[split:],  z_[split:]
#
#     fitter = make_pipeline(PolynomialFeatures(deg[0]), Ridge(alpha=lam))
#     # fitter = Ridge2d(deg=deg, lam=lam, alpha=0.05, fulloutput=False)
#     prediction_error_from_cross_validation = cross_validate_prediction_error(train_data[0],
#                                                                              train_data[1], fitter, k=5)
#
#     fitter.fit(train_data[0], train_data[1])
#
#     # fit_fun = partial(ridge2d, deg=deg, lam=lam)
#     # fit_fun = partial(lasso2d, deg=deg, lam=lam)
#
#     # fit_fun = partial(ols2d, deg=deg)  # lam=0
#     # beta_avg, beta_var = bootstrap(*train_data, fun=fit_fun)
#
#     zpredict = fitter.predict(test_data[0])
#     prediction_error = mse(test_data[1], zpredict)
#     average_bias_squared = mse(z0_[:split], zpredict)
#     stats = fitstat2(prediction_r2_score=r_squared(test_data[1], zpredict),
#                      prediction_error_from_cross_validation=prediction_error_from_cross_validation,
#                      prediction_error=prediction_error,
#                      average_bias_squared=average_bias_squared,
#                      average_variance=prediction_error - average_bias_squared)
#
# #     print('mean_squared_error: {:.2g}'.format(prediction_error))
# #     print('R2: {:.2f}'.format(prediction_r2_score))
# #     print('Sigma^2: {:.2g}'.format(sigma ** 2))
# #     print('Average Bias**2: {:.2g}'.format(average_bias_squared))
# #     print('Average Variance: {:.2g}'.format(average_variance))
#
#     return stats



def generate_dataset(m, sigma):
    """
    Return random Franke function data set

    Returns
    -------
        X, z, z0
    where
        z0 = franke_function(X[:, 0], X[:, 1])
    and
        z = z0 + noise

    """

    X = np.random.rand(m, 2)
    z0 = franke_function(X[:, 0], X[:, 1]).reshape(-1, 1)
    z = z0 + sigma * np.random.randn(m, 1)
    return X, z, z0


def plot_prediction_error_vs_degrees(X, z, z0, degrees=2, lam=None, sigma=0, method='ridge'):
    if method == 'ridge':
        fitfun = fit_ridge2d
    elif method == 'lasso':
        fitfun = fit_lasso2d

    m = len(z)
    lam_dict = lam if isinstance(lam, dict) else {}


    stats_all = []
    for degree in degrees:
        if lam_dict:
            # (m, sigma, degree)
            lam = lam_dict[(m, sigma, degree)]
        stats = fitfun(X, z, z0, deg=(degree, degree), lam=lam)
        stats_all.append(tuple(stats))

    stats_all = np.array(stats_all)
    names = list(stats._asdict())

    colors = 'mbygr'
    lines = ['-', '--', '-', '--', '-.']
    if lam==0:
        method_name = dict(ridge='ols').get(method, method)
    else:
        method_name = method
    if lam_dict:
        lam_txt = lam = 'optimum'
    else:
        lam_txt = int(-np.log10(lam+1e-300))

    for i, name in enumerate(names):
        sym = colors[i]+lines[i]
        plt.plot(degrees, stats_all[:, i], sym, label=name)
        if i==0:

            plt.title(f'm={m}, sigma={sigma:0.2f}, lam={lam}')
            plt.xlabel('degrees')
            plt.axis([0, 5, 0, 1])
            plt.legend()
            plt.savefig('{}_r2_score_vs_degrees_m{}_l{}_s{}.png'.format(method_name, m, lam, int(sigma*100)) )
            # plt.show('hold')
            plt.close()

    plt.title(f'm={m}, sigma={sigma:0.2f}, lam={lam}')
    plt.xlabel('degrees')
    # plt.axis([4, 25, 0, 1])
    plt.legend()
    plt.savefig('{}prediction_error_vs_degrees_m{}_l{}_s{}.png'.format(method_name, m, lam, int(sigma*100)) )
    # plt.show('hold')
    plt.close()


def plot_prediction_error_vs_lambdas(X, z, z0, degree=2, lambdas=None, sigma=0, method='ridge'):
    if method == 'ridge':
        fitfun = fit_ridge2d
    elif method == 'lasso':
        fitfun = fit_lasso2d

    m = len(z)

    stats_all = []
    for lam in lambdas:
        stats = fitfun(X, z, z0, deg=(degree, degree), lam=lam)
        stats_all.append(tuple(stats))

    stats_all = np.array(stats_all)
    names = list(stats._asdict())

    colors = 'mbygr'
    lines = ['-', '--', '-', '--', '-.']
    for i, name in enumerate(names):
        sym = colors[i]+lines[i]
        plt.semilogx(lambdas, stats_all[:, i], sym, label=name)
        if i==0:
            plt.title(f'm={m}, sigma={sigma:0.2f}, degree={degree}')
            plt.xlabel('lambda')

            plt.legend()
            plt.savefig('{}_r2_score_m{}_d{}_s{}.png'.format(method, m, degree, int(sigma*100)) )
            plt.close()

    plt.title(f'm={m}, sigma={sigma:0.2f}, degree={degree}')
    plt.xlabel('lambda')
    # plt.axis([4, 25, 0, 1])
    plt.legend()

    plt.savefig('{}prediction_error_m{}_d{}_s{}.png'.format(method, m, degree, int(sigma*100)) )

    # plt.show('hold')
    plt.close()
    lambdas = np.array(lambdas).reshape(-1, 1)
    df = pd.DataFrame.from_records(np.hstack((lambdas, stats_all)), columns=['lambda'] + names)
    ix = df['prediction_error'].argmin(axis=0)
    print('lam_min={}'.format(lambdas[ix]))
    filename = '{}_prediction_error_m{}_d{}_s{}.txt'.format(method, m, degree, int(sigma*100))
    df.to_csv(filename, sep='\t', encoding='utf-8')


def test_scikit1d():
    f = lambda x:  np.sin(x) * x

    # generate points used to plot
    x_plot = np.linspace(0, 10, 100)

    # generate points and keep a subset of them
    x = np.linspace(0, 10, 100)
    rng = np.random.RandomState(0)
    rng.shuffle(x)
    x = np.sort(x[:20])
    y = f(x)

    # create matrix versions of these arrays
    X = x[:, np.newaxis]
    X_plot = x_plot[:, np.newaxis]

    colors = ['teal', 'yellowgreen', 'gold']
    lw = 2
    plt.plot(x_plot, f(x_plot), color='cornflowerblue', linewidth=lw,
             label="ground truth")
    plt.scatter(x, y, color='navy', s=30, marker='o', label="training points")

    for count, degree in enumerate([3, 4, 5]):
        model = make_pipeline(PolynomialFeatures(degree), Ridge())
        model.fit(X, y)
        y_plot = model.predict(X_plot)
        plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw,
                 label="degree %d" % degree)

    plt.legend(loc='lower left')

    plt.show()


def example_fit_scikit2d(deg=(4, 4), m=10000, sigma=0.1, lam=1):
    shape = m, 1
    x = np.random.rand(*shape)
    y = np.random.rand(*shape)
    z0 = franke_function(x, y)
    z = z0 + sigma * np.random.randn(*shape)

    X = np.hstack((x, y))
#     poly = PolynomialFeatures(degree=deg[0])
#     t_X = poly3.fit_transform(X)
#     model = linear_model.Lasso(alpha=lam)

    model = make_pipeline(PolynomialFeatures(deg[0]), Lasso(alpha=lam))
    model.fit(X, z)

    xy = np.linspace(0, 1, 201)
    xnew, ynew = np.meshgrid(xy, xy)
    zpredict = model.predict(np.vstack((xnew.ravel(), ynew.ravel())).T).reshape(xnew.shape)


    plot_surface(xnew, ynew, zpredict)

    z_new = franke_function(xnew, ynew)
    print(1-mse(z_new, zpredict)/mse(z_new, z_new.mean()))

    plot_surface(xnew, ynew, z_new)
#     ix = np.argsort(x.ravel())
#     plt.plot(x[ix,:], y[ix, :], 'g-')
#     plt.axis([0, 2.0, 0, 15.0])
#     plt.xlabel(r"$x$")
#     plt.ylabel(r"$y$")
#     plt.title(r'Linear Regression')
    plt.show('hold')


def example_fit_2d(deg=(4, 4), m=10000, sigma=0.1, lam=1):
    shape = m, 1
    x = np.random.rand(*shape)
    y = np.random.rand(*shape)
    z0 = franke_function(x, y)
    z = z0 + sigma * np.random.randn(*shape)

    fitter = Ridge2d(deg=deg, lam=lam, fulloutput=True)
    fitter.fit(np.hstack((x, y)), z)
    print(fitter.stats)

    xy = np.linspace(0, 1, 201)
    xnew, ynew = np.meshgrid(xy, xy)
    zpredict = fitter.predict(np.vstack((xnew.ravel(), ynew.ravel())).T).reshape(xnew.shape)


    plot_surface(xnew, ynew, zpredict)

    z_new = franke_function(xnew, ynew)
    print(mse(z_new, zpredict)/mse(z_new, z_new.mean()))

    plot_surface(xnew, ynew, z_new)
#     ix = np.argsort(x.ravel())
#     plt.plot(x[ix,:], y[ix, :], 'g-')
#     plt.axis([0, 2.0, 0, 15.0])
#     plt.xlabel(r"$x$")
#     plt.ylabel(r"$y$")
#     plt.title(r'Linear Regression')
    plt.show('hold')


def plot_surface(x, y, z):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)


def tri_surface_plot(surface, title, surface1=None, title2=None):
    if title2 is None:
        title2 = title

    x, y, z = surface.T
    fig = plt.figure()

    if surface1 is not None:
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_trisurf(x, y, z, cmap=cm.viridis, linewidth=0, antialiased=True)
        plt.title(title)

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        x1, y1, z1 = surface1.T
        ax.plot_trisurf(x1, y1, z1, cmap=cm.viridis, linewidth=0, antialiased=True)

        plt.title(title2)
    else:
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(x, y, z, cmap=cm.viridis, linewidth=0, antialiased=True)
        plt.title(title)


def surface_plot(surface,title, surface1=None, title2=None):
    if title2 is None:
        title2 = title
    M,N = surface.shape

    ax_rows = np.arange(M)
    ax_cols = np.arange(N)

    [X,Y] = np.meshgrid(ax_cols, ax_rows)

    fig = plt.figure()
    if surface1 is not None:
        ax = fig.add_subplot(1,2,1,projection='3d')
        ax.plot_surface(X,Y,surface,cmap=cm.viridis,linewidth=0)
        plt.title(title)

        ax = fig.add_subplot(1,2,2,projection='3d')
        ax.plot_surface(X,Y,surface1,cmap=cm.viridis,linewidth=0)

        plt.title(title2)
    else:
        ax = fig.gca(projection='3d')
        ax.plot_surface(X,Y,surface,cmap=cm.viridis,linewidth=0)
        plt.title(title)


def task_a(degrees=(1,2,3,4,5),
         lambdas=(0, ),
         sample_sizes=(300, 3000, 30000),
         sigmas=(0, 0.1, 0.5),
         method='ridge'):

    for m in sample_sizes:
        print(f'm={m}')
        for sigma in sigmas:
            X, z, z0 = generate_dataset(m, sigma)
            print(f'    sigma={sigma}')
            for lam in lambdas:
                print(f'        lambda={lam}')
                plot_prediction_error_vs_degrees(X, z, z0, degrees, lam=lam, sigma=sigma, method=method)




def task_b(degrees=(1,2,3,4,5), lambdas=(1e-10, 1e-7, 1e-4, 1e-1),
           sample_sizes=(300, 3000, 30000),
           sigmas=(0, 0.1, 0.5)):

    task(degrees, lambdas, sample_sizes, sigmas, method='ridge')


def task_c(degrees=(1,2,3,4,5), lambdas=(1e-7, 1e-5, 1e-3, 1e-1),
           sample_sizes=(300, 3000, 30000),
           sigmas=(0, 0.1, 0.5)):
    task(degrees, lambdas, sample_sizes, sigmas, method='lasso')


def task(degrees=(1,2,3,4,5),
         lambdas=(1e-10, 1e-7, 1e-4, 1e-1, 10, 100),
         sample_sizes=(300, 3000, 30000),
         sigmas=(0, 0.1, 0.5),
         method='ridge'):

    for m in sample_sizes:
        print(f'm={m}')
        for sigma in sigmas:
            X, z, z0 = generate_dataset(m, sigma)
            print(f'    sigma={sigma}')
            for degree in degrees:
                print(f'        degree={degree}')
                plot_prediction_error_vs_lambdas(X, z, z0, degree, lambdas=lambdas, sigma=sigma, method=method)


# (m, sigma, degree)
RIDGE_OPTIMUM_LAMBDA = {(300,0,1): 1.00000000e-07,
                  (300,0,2): 1.00000000e-10,
                  (300,0,3): 1.00000000e-10,
                  (300,0,4): 1.00000000e-7,
                  (300,0,5): 1.00000000e-7,

                  (300,0.1,1): 1.00000000e-1,
                  (300,0.1,2): 1.00000000e-4,
                  (300,0.1,3): 1.00000000e-10,
                  (300,0.1,4): 1.00000000e-7,
                  (300,0.1,5): 1.00000000e-4,

                  (300,0.5,1): 1.00000000e-4,
                  (300,0.5,2): 1.00000000e-4,
                  (300,0.5,3): 1.00000000e-7,
                  (300,0.5,4): 1.00000000e-7,
                  (300,0.5,5): 1.00000000e-4,

                  (3000,0,1): 1.00000000e-10,
                  (3000,0,2): 1.00000000e-1,
                  (3000,0,3): 1.00000000e-1,
                  (3000,0,4): 1.00000000e-10,
                  (3000,0,5): 1.00000000e-10,

                  (3000,0.1,1): 1.00000000e-10,
                  (3000,0.1,2): 1.00000000e-7,
                  (3000,0.1,3): 1.00000000e-10,
                  (3000,0.1,4): 1.00000000e-7,
                  (3000,0.1,5): 1.00000000e-7,

                  (3000,0.5,1): 1.00000000e-10,
                  (3000,0.5,2): 1.00000000e-10,
                  (3000,0.5,3): 1.00000000e-4,
                  (3000,0.5,4): 1.00000000e-7,
                  (3000,0.5,5): 1.00000000e-7,
                  }


LASSO_OPTIMUM_LAMBDA = {(300,0,1): 1.00000000e-07,
                  (300,0,2): 1.00000000e-5,
                  (300,0,3): 1.00000000e-7,
                  (300,0,4): 1.00000000e-5,
                  (300,0,5): 1.00000000e-7,

                  (300,0.1,1): 1.00000000e-3,
                  (300,0.1,2): 1.00000000e-7,
                  (300,0.1,3): 1.00000000e-5,
                  (300,0.1,4): 1.00000000e-7,
                  (300,0.1,5): 1.00000000e-7,

                  (300,0.5,1): 1.00000000e-5,
                  (300,0.5,2): 1.00000000e-7,
                  (300,0.5,3): 1.00000000e-3,
                  (300,0.5,4): 1.00000000e-5,
                  (300,0.5,5): 1.00000000e-7,

                  (3000,0,1): 1.00000000e-5,
                  (3000,0,2): 1.00000000e-7,
                  (3000,0,3): 1.00000000e-7,
                  (3000,0,4): 1.00000000e-7,
                  (3000,0,5): 1.00000000e-7,

                  (3000,0.1,1): 1.00000000e-7,
                  (3000,0.1,2): 1.00000000e-7,
                  (3000,0.1,3): 1.00000000e-7,
                  (3000,0.1,4): 1.00000000e-7,
                  (3000,0.1,5): 1.00000000e-7,

                  (3000,0.5,1): 1.00000000e-3,
                  (3000,0.5,2): 1.00000000e-7,
                  (3000,0.5,3): 1.00000000e-7,
                  (3000,0.5,4): 1.00000000e-5,
                  (3000,0.5,5): 1.00000000e-5,
                  }


def fit_best_model(degree=4, lam=0, sample_size=300, sigma=0.5, method='ridge'):
    if method== 'ridge':
        model = Ridge2d(deg=(degree, degree), lam=lam, fulloutput=True)
    else:
        model = Lasso2d(deg=(degree, degree), lam=lam)


    X, z, z0 = generate_dataset(sample_size, sigma)
    model.fit(X, z)
    model.summary()
    zhat = model.predict(X)

    tri_surface_plot(np.hstack((X, zhat)), 'Fitted terrain surface OLS',
                     np.hstack((X, z)), 'True terrain surface')
    plt.show('hold')


if __name__ == '__main__':
    np.random.seed(4155)
    # example_fit_2d(deg=(5, 5), m=5000, sigma=.0, lam=0)
    fit_best_model(degree=4, lam=0, sample_size=300, sigma=0.5, method='ridge')
    # task_a()
#     task_a(degrees=(1, 2, 3, 4, 5),
#            lambdas=(LASSO_OPTIMUM_LAMBDA, ),
#            sample_sizes=(300, 3000),
#            sigmas=(0, 0.1, 0.5),
#            method='lasso')

#    task_b()
#     task_c()
    # test_scikit1d()
    # example_fit_scikit2d(deg=(5,5), m=10000, sigma=0.0, lam=1e-5)
