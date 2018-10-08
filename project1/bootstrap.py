def bootstrap_bias_variance(x, y, model, n_bootstraps=200, test_size=0.2):
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
    bias_sqares = (y_test - np.mean(y_pred, axis=1, keepdims=True)) ** 2
    variances = np.var(y_pred, axis=1, keepdims=True)

    r2 =  1.0 - np.mean(errors)/mse(y_test, mean_y)
    error = np.mean( errors )
    bias_squared = np.mean(bias_sqares)
    variance = np.mean( variances )

	return error, bias_squared, variance, r2