import os

import numpy as np
from sklearn import metrics
from tqdm import tqdm
import time
import tracemalloc

from models.arima import ArimaPredictions
from models.esn import ESNPredictions
from models.gru import GRUPredictions
from models.lstm import LSTMPredictions
from utils.data_preparation import get_processed_data


def get_mae_metrics(path, opt=False):
    mae = []
    data_array, name = get_processed_data(path)
    esn_simple = ESNPredictions(data_array, window_size=50, forecast_size=10, test_size=0.1, model_type='simple')
    Y_pred_simple = esn_simple.get_predictions(opt=opt)
    Y_test_simple = esn_simple.Y_test[:, 0][:len(Y_pred_simple)]
    mae.append(metrics.mean_absolute_error(Y_test_simple, Y_pred_simple[:len(Y_test_simple)]))

    esn_deep = ESNPredictions(data_array, window_size=50, forecast_size=10, test_size=0.1, model_type='deep')
    Y_pred_deep = esn_deep.get_predictions(opt=opt)
    Y_test_deep = esn_deep.Y_test[:, 0][:len(Y_pred_deep)]
    mae.append(metrics.mean_absolute_error(Y_test_deep, Y_pred_deep[:len(Y_test_deep)]))

    esn_hier = ESNPredictions(data_array, window_size=50, forecast_size=10, test_size=0.1, model_type='hierarchical')
    Y_pred_hier = esn_hier.get_predictions(opt=opt)
    Y_test_hier = esn_hier.Y_test[:, 0][:len(Y_pred_hier)]
    mae.append(metrics.mean_absolute_error(Y_test_hier, Y_pred_hier[:len(Y_test_hier)]))

    arima = ArimaPredictions(data_array, forecast_size=10, test_size=0.2)
    Y_pred_arima = arima.get_predictions(opt=opt)
    Y_test_arima = arima.Y_test[:len(Y_pred_arima)]
    mae.append(metrics.mean_absolute_error(Y_test_arima, Y_pred_arima[:len(Y_test_arima)]))

    lstm = LSTMPredictions(data_array, window_size=50, forecast_size=10, test_size=0.1, )
    Y_pred_lstm = lstm.get_predictions(opt=opt)
    Y_test_lstm = lstm.Y_test_one_dim[:len(Y_pred_lstm)]
    mae.append(metrics.mean_absolute_error(Y_test_lstm, Y_pred_lstm))

    gru = GRUPredictions(data_array, window_size=50, forecast_size=10, test_size=0.1, )
    Y_pred_gru = gru.get_predictions(opt=opt)
    Y_test_gru = gru.Y_test_one_dim[:len(Y_pred_gru)]
    mae.append(metrics.mean_absolute_error(Y_test_gru, Y_pred_gru))
    return mae, name

def get_time_memory(path, opt=False):
    run_time = []
    memory = []

    data_array, name = get_processed_data(path)

    start_time = time.time()
    tracemalloc.start()
    esn_simple = ESNPredictions(data_array, window_size=50, forecast_size=10, test_size=0.1, model_type='simple')
    Y_pred_simple = esn_simple.get_predictions(opt=opt)
    time_req = time.time() - start_time
    memory_req = tracemalloc.get_traced_memory()[1]
    run_time.append(time_req)
    memory.append(memory_req)
    tracemalloc.stop()

    start_time = time.time()
    tracemalloc.start()
    esn_deep = ESNPredictions(data_array, window_size=50, forecast_size=10, test_size=0.1, model_type='deep')
    Y_pred_deep = esn_deep.get_predictions(opt=opt)
    time_req = time.time() - start_time
    memory_req = tracemalloc.get_traced_memory()[1]
    run_time.append(time_req)
    memory.append(memory_req)
    tracemalloc.stop()

    start_time = time.time()
    tracemalloc.start()
    esn_hier = ESNPredictions(data_array, window_size=50, forecast_size=10, test_size=0.1, model_type='hierarchical')
    Y_pred_hier = esn_hier.get_predictions(opt=opt)
    time_req = time.time() - start_time
    memory_req = tracemalloc.get_traced_memory()[1]
    run_time.append(time_req)
    memory.append(memory_req)
    tracemalloc.stop()

    start_time = time.time()
    tracemalloc.start()
    arima = ArimaPredictions(data_array, forecast_size=10, test_size=0.2)
    Y_pred_arima = arima.get_predictions(opt=opt)
    time_req = time.time() - start_time
    memory_req = tracemalloc.get_traced_memory()[1]
    run_time.append(time_req)
    memory.append(memory_req)
    tracemalloc.stop()

    start_time = time.time()
    tracemalloc.start()
    lstm = LSTMPredictions(data_array, window_size=50, forecast_size=10, test_size=0.1, )
    Y_pred_lstm = lstm.get_predictions(opt=opt)
    time_req = time.time() - start_time
    memory_req = tracemalloc.get_traced_memory()[1]
    run_time.append(time_req)
    memory.append(memory_req)
    tracemalloc.stop()

    start_time = time.time()
    tracemalloc.start()
    gru = GRUPredictions(data_array, window_size=50, forecast_size=10, test_size=0.1, )
    Y_pred_gru = gru.get_predictions(opt=opt)
    time_req = time.time() - start_time
    memory_req = tracemalloc.get_traced_memory()[1]
    run_time.append(time_req)
    memory.append(memory_req)
    tracemalloc.stop()

    return run_time, memory

def get_dataset_mean_mae(path, n_stocks=10):
    maes = []
    maes_with_names = {}
    processed_stocks = 0
    for file in tqdm(os.scandir(path)):
        print(file.path)
        processed_stocks += 1
        mae, name = get_mae_metrics(file.path)
        maes.append(mae)
        maes_with_names[name] = mae
        if processed_stocks >= n_stocks:
            break
    mean_maes = np.mean(maes, axis=0)
    return maes_with_names, mean_maes

def get_dataset_time_memory(path, n_stocks=10):
    time = []
    memory = []
    processed_stocks = 0
    for file in tqdm(os.scandir(path)):
        print(file.path)
        processed_stocks += 1
        time_one, memory_one = get_time_memory(file.path)
        time.append(time_one)
        memory.append(memory_one)
        if processed_stocks >= n_stocks:
            break
    mean_time = np.mean(time, axis=0)
    mean_memory = np.mean(memory, axis=0)
    return mean_time, mean_memory
