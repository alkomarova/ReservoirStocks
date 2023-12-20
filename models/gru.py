import hyperopt
from hyperopt import fmin, tpe, hp
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models.lstm import LSTMPredictions
from utils.data_preparation import get_window_data, get_longterm_predictions, transform_y_for_longterm

class GRUPredictions(LSTMPredictions):
    def create_gru_model(self, params):
        model = Sequential()
        model.add(GRU(units=int(params['units']), input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(Dense(self.forecast_size))
        optimizer = Adam(learning_rate=params['lr'])
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    def objective(self, params):
        model = self.create_gru_model(params)
        model.fit(self.X_train[:len(self.Y_train)], self.Y_train, epochs=10, batch_size=32, verbose=0)
        y_pred = model.predict(self.X_test[:len(self.Y_test)])
        mae = mean_absolute_error(self.Y_test, y_pred)
        return mae

    def get_predictions(self, opt = True):
        if opt == True:
            params = self.get_optimal_parameters()
        else:
            params = {'lr': 0.075, 'units': 100}
        model = self.create_gru_model(params)
        model.fit(self.X_train[:len(self.Y_train)], self.Y_train, epochs=10, batch_size=32, verbose=0)
        y_pred = model.predict(self.X_test[:len(self.Y_test)])
        y_pred = get_longterm_predictions(y_pred)

        return y_pred
