import hyperopt
from hyperopt import fmin, tpe, hp
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models.esn import ESNPredictions
from utils.data_preparation import get_window_data, get_longterm_predictions, transform_y_for_longterm

class LSTMPredictions():

    def __init__(self, data, window_size = 100, forecast_size = 1, test_size = 0.3) -> None:
        self.data = data
        self.window_size = window_size
        self.forecast_size = forecast_size
        self.test_size= test_size

        self.data_train_len = round(self.data.shape[0]*(1-self.test_size))
        self.data_test_len = self.data.shape[0] - self.data_train_len
        self.X_train, self.Y_train = get_window_data(self.data[:self.data_train_len], self.window_size)
        self.X_test, self.Y_test = get_window_data(self.data[self.data_train_len - self.window_size:-1], self.window_size)
        self.Y_test_one_dim = self.Y_test
        self.Y_train = transform_y_for_longterm(self.Y_train[:, 0], self.forecast_size)
        self.Y_test = transform_y_for_longterm(self.Y_test[:, 0], self.forecast_size)
        self.Y_test_one_dim = get_longterm_predictions(self.Y_test)

        self.X_train = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], self.X_train.shape[2]))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], self.X_test.shape[2]))


    def create_lstm_model(self, params):
        model = Sequential()
        model.add(LSTM(units=int(params['units']), input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(Dense(self.forecast_size))
        optimizer = Adam(learning_rate=params['lr'])
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    def objective(self, params):
        model = self.create_lstm_model(params)
        model.fit(self.X_train[:len(self.Y_train)], self.Y_train, epochs=10, batch_size=32, verbose=0)
        y_pred = model.predict(self.X_test[:len(self.Y_test)])
        mae = mean_absolute_error(self.Y_test, y_pred)
        return mae

    def get_optimal_parameters(self):
        space = {
            'units': hp.uniform('units', 50, 150),
            'lr': hp.loguniform('lr', -5, -2),
        }
        best = fmin(fn=self.objective, space=space, algo=tpe.suggest, max_evals=10)
        return best

    def get_predictions(self, opt = True):
        if opt == True:
            params = self.get_optimal_parameters()
        else:
            params = {'lr': 0.075, 'units': 100}
        model = self.create_lstm_model(params)
        model.fit(self.X_train[:len(self.Y_train)], self.Y_train, epochs=10, batch_size=32, verbose=0)
        y_pred = model.predict(self.X_test[:len(self.Y_test)])
        y_pred = get_longterm_predictions(y_pred)

        return y_pred


