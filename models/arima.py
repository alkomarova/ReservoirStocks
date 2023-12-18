from pmdarima.arima import auto_arima, arima
import math
import numpy as np

class ArimaPredictions:

    def __init__(self, data, forecast_size = 10, test_size = 0.3) -> None:
        self.data = data
        self.forecast_size = forecast_size
        self.test_size= test_size

        self.data_train_len = round(self.data.shape[0]*(1-self.test_size))
        self.data_test_len = self.data.shape[0] - self.data_train_len
        self.X_train = data[:self.data_train_len][:, 0]
        self.Y_train = data[1:self.data_train_len+1][:, 0]
        self.X_test = data[self.data_train_len:-1][:, 0]
        self.Y_test = data[self.data_train_len+1:][:, 0]

        #self.feature_len = self.X_test.shape[1]

    def get_predictions(self, opt=True):

        if opt == True:
            model = auto_arima(self.X_train, start_p=1, start_q=1,
                           test='adf',
                           max_p=5, max_q=5,
                           m=1,
                           d=1,
                           seasonal=False,
                           start_P=0,
                           D=None,
                           trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
        else:
            model = arima.ARIMA(order = (1, 0, 2))
            model.fit(self.X_train)

        test_blocks = math.floor(self.data_test_len / self.forecast_size)
        Y_pred = []
        X_train_temp = self.X_train
        X_test_temp = self.X_test
        Y_pred_test = []

        for i in range(test_blocks):
            x = model.predict(n_periods = self.forecast_size)
            Y_pred_test = np.append(Y_pred_test, x)

            X_train_temp = np.append(X_train_temp, X_test_temp[:][:self.forecast_size])
            X_test_temp = X_test_temp[:][self.forecast_size:]
            model.fit(X_train_temp)

        return Y_pred_test
