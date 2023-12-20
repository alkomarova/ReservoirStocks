import numpy as np
import pandas as pd
import ta


def get_processed_data(path,
                       ta_list=['volume_adi', 'trend_macd', 'volatility_kcp', 'momentum_stoch_rsi', 'trend_sma_slow']):
    # print(os.getcwd())
    print(type(path))
    data = pd.read_csv(path).rename(columns={"date": "Date", "open": "Open", "high": "High",
                                             "low": "Low", "close": "Close",
                                             "volume": "Volume"})
    name = data['Name'][0]
    data = data.drop(['Name'], axis=1)
    data['LogRet'] = np.log(data['Close']).diff()
    data = ta.add_all_ta_features(
        data, open="Open", high="High", low="Low", close="LogRet", volume="Volume")
    ta_list = ['LogRet'] + ta_list
    data_array = data[ta_list].dropna().to_numpy()
    return data_array, name


def get_window_data(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


def transform_y_for_longterm(arr, forecast_size):
    return np.array([list(arr[i:i + forecast_size]) for i in range(len(arr) - forecast_size + 1)])


def get_longterm_predictions(lt_predictions):
    predictions = []
    predictions += list(lt_predictions[0])
    for i in range(1, len(lt_predictions)):
        predictions.append(lt_predictions[i][-1])
    return predictions
