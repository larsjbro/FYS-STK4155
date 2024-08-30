import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def lecture_note_solution_exercise_2():
    x = np.random.rand(100)
    y = 2.0 + 5 * x * x + 0.1 * np.random.randn(100)

    #  The design matrix now as function of a given polynomial
    X = np.zeros((len(x), 3))
    X[:, 0] = 1.0
    X[:, 1] = x
    X[:, 2] = x ** 2
    # We split the data in test and training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # matrix inversion to find beta
    beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    print(beta)
    # and then make the prediction
    ytilde = X_train @ beta
    print("Training R2")
    print(R2(y_train, ytilde))
    print("Training MSE")
    print(MSE(y_train, ytilde))
    ypredict = X_test @ beta
    print("Test R2")
    print(R2(y_test, ypredict))
    print("Test MSE")
    print(MSE(y_test, ypredict))

    index=np.argsort(X_test[:,1])
    x = X_test[index,1]
    y = y_test[index]
    y_fit = X_test[index,:] @ beta

    plt.scatter(x, y, color='red', label='Data points')
    plt.plot(x, y_fit, label='Fitted polynomial', color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Second Order Polynomial Fit')
    plt.legend()
    plt.show()

if __name__ == '__main__':