import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def generate_data(n=100,sigma=0.1):
    x = np.random.rand(n,1)
    y = 2.0+5*x*x+sigma*np.random.randn(n,1)
    return x,y


def linear_regression(n=100):
    x, y = generate_data(n)
    linreg = LinearRegression()
    linreg.fit(x,y)
    # This is our new x-array to which we test our model
    xnew = np.array([[0],[1]])
    ypredict = linreg.predict(xnew)

    plt.plot(xnew, ypredict, "r-")
    plt.plot(x, y ,'ro')
    #plt.axis([0,1.0,0, 5.0])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'Simple Linear Regression')
    plt.show()


def second_order_polynomial_regression(n=100):
    x, y = generate_data(n)
    coefficients = np.polyfit(x.ravel(), y.ravel(), 2)
    polynomial = np.poly1d(coefficients)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = polynomial(x_fit)
    plt.scatter(x, y, color='red', label='Data points')
    plt.plot(x_fit, y_fit, label='Fitted polynomial', color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Second Order Polynomial Fit')
    plt.legend()
    plt.show()


def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)


def MSE(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data - y_model) ** 2) / n


def polynomial_sci_kit(n=100, sigma=0.1):
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression())
    ])


    x, y = generate_data(n, sigma)
    model.fit(x, y)
    x2 = np.linspace(0,1,)[:, np.newaxis]
    y2 = model.predict(x2)
    y2_model = model.predict(x)
    mse = MSE(y,y2_model)
    r2 = R2(y,y2_model)
    plt.plot(x,y, '.')
    plt.plot(x2,y2)
    plt.title(f'MSE={mse:.3f},R2={r2:.3f}')
    plt.show()


def lecture_note_solution_exercise_2_with_plot(degree=2):
    x = np.random.rand(100)
    y = 2.0 + 5 * x * x + 0.1 * np.random.randn(100)

    #  The design matrix now as function of a given polynomial
    X = np.zeros((len(x), degree+1))
    X[:, 0] = 1.0
    for i in range(1,degree+1):

        X[:, i] = x ** i

    # We split the data in test and training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # matrix inversion to find beta
    beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    print(beta)
    # and then make the prediction
    ytilde = X_train @ beta
    print("Training R2")
    r2_train = R2(y_train, ytilde)
    print("Training MSE")
    mse_train=MSE(y_train, ytilde)
    ypredict = X_test @ beta
    print("Test R2")
    r2_test = R2(y_test, ypredict)
    print("Test MSE")
    mse_test = MSE(y_test, ypredict)


    index=np.argsort(X_test[:,1])
    x = X_test[index,1]
    y = y_test[index]
    y_fit = X_test[index,:] @ beta

    plt.scatter(x, y, color='red', label='Data points')
    plt.plot(x, y_fit, label='Fitted polynomial', color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'MSE_train_test=({mse_train:.3f},{mse_test:.3f}),R2_train_test=({r2_train:.3f},{r2_test:.3f}),Degree={degree}')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    #main(n=5)
    #linear_regression(n=5)
    #second_order_polynomial_regression(n=100)
    #polynomial_sci_kit(n=100, sigma=1)
    '''
    Trying with sigma =1 for big noise. I see that the polynomial has a good general fit, 
    but the data are scattered quite far from the fitted line as shown with R2 = 0.690
    With Sigma 0.1 the fit and scatter are both close to the fitted line.
    '''
    lecture_note_solution_exercise_2_with_plot(degree=5)
