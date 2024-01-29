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

#для каждого файла с котировками возвращает метрику каждой из 6 моделей и название котировки
def get_metrics(path, metric_def, opt=True):
    metric = []
    data_array, name = get_processed_data(path)
    esn_simple = ESNPredictions(data_array, window_size=50, forecast_size=10, test_size=0.1, model_type='simple')
    Y_pred_simple = esn_simple.get_predictions(opt=opt)
    Y_test_simple = esn_simple.Y_test[:, 0][:len(Y_pred_simple)]
    metric.append(metric_def(Y_test_simple, Y_pred_simple[:len(Y_test_simple)]))

    esn_deep = ESNPredictions(data_array, window_size=50, forecast_size=10, test_size=0.1, model_type='deep')
    Y_pred_deep = esn_deep.get_predictions(opt=opt)
    Y_test_deep = esn_deep.Y_test[:, 0][:len(Y_pred_deep)]
    metric.append(metric_def(Y_test_deep, Y_pred_deep[:len(Y_test_deep)]))

    esn_hier = ESNPredictions(data_array, window_size=50, forecast_size=10, test_size=0.1, model_type='hierarchical')
    Y_pred_hier = esn_hier.get_predictions(opt=opt)
    Y_test_hier = esn_hier.Y_test[:, 0][:len(Y_pred_hier)]
    metric.append(metric_def(Y_test_hier, Y_pred_hier[:len(Y_test_hier)]))

    arima = ArimaPredictions(data_array, forecast_size=10, test_size=0.2)
    Y_pred_arima = arima.get_predictions(opt=opt)
    Y_test_arima = arima.Y_test[:len(Y_pred_arima)]
    metric.append(metric_def(Y_test_arima, Y_pred_arima[:len(Y_test_arima)]))

    lstm = LSTMPredictions(data_array, window_size=50, forecast_size=10, test_size=0.1, )
    Y_pred_lstm = lstm.get_predictions(opt=opt)
    Y_test_lstm = lstm.Y_test_one_dim[:len(Y_pred_lstm)]
    metric.append(metric_def(Y_test_lstm, Y_pred_lstm))

    gru = GRUPredictions(data_array, window_size=50, forecast_size=10, test_size=0.1, )
    Y_pred_gru = gru.get_predictions(opt=opt)
    Y_test_gru = gru.Y_test_one_dim[:len(Y_pred_gru)]
    metric.append(metric_def(Y_test_gru, Y_pred_gru))
    return metric, name


#для каждого файла с котировками возвращает предсказания значений с помощью каждой из 6 моделей и название котировки
def get_predictions(path, opt=True):
    predictions = {}
    data_array, name = get_processed_data(path)
    esn_simple = ESNPredictions(data_array, window_size=50, forecast_size=10, test_size=0.1, model_type='simple')
    Y_pred_simple = esn_simple.get_predictions(opt=opt)
    Y_test_simple = esn_simple.Y_test[:, 0][:len(Y_pred_simple)]

    esn_deep = ESNPredictions(data_array, window_size=50, forecast_size=10, test_size=0.1, model_type='deep')
    Y_pred_deep = esn_deep.get_predictions(opt=opt)
    Y_test_deep = esn_deep.Y_test[:, 0][:len(Y_pred_deep)]

    esn_hier = ESNPredictions(data_array, window_size=50, forecast_size=10, test_size=0.1, model_type='hierarchical')
    Y_pred_hier = esn_hier.get_predictions(opt=opt)
    Y_test_hier = esn_hier.Y_test[:, 0][:len(Y_pred_hier)]

    arima = ArimaPredictions(data_array, forecast_size=10, test_size=0.2)
    Y_pred_arima = arima.get_predictions(opt=opt)
    Y_test_arima = arima.Y_test[:len(Y_pred_arima)]

    lstm = LSTMPredictions(data_array, window_size=50, forecast_size=10, test_size=0.1, )
    Y_pred_lstm = lstm.get_predictions(opt=opt)
    Y_test_lstm = lstm.Y_test_one_dim[:len(Y_pred_lstm)]

    gru = GRUPredictions(data_array, window_size=50, forecast_size=10, test_size=0.1, )
    Y_pred_gru = gru.get_predictions(opt=opt)
    Y_test_gru = gru.Y_test_one_dim[:len(Y_pred_gru)]

    min_len = min([len(Y_test_simple), len(Y_test_deep), len(Y_test_hier),
              len(Y_test_arima), len(Y_test_lstm), len(Y_test_gru)])

    predictions['ESN_test'] = Y_test_simple[:min_len]
    predictions['ESN_pred'] = Y_pred_simple[:min_len]
    predictions['DSN_test'] = Y_test_deep[:min_len]
    predictions['DSN_pred'] = Y_pred_deep[:min_len]
    predictions['HSN_test'] = Y_test_hier[:min_len]
    predictions['HSN_pred'] = Y_pred_hier[:min_len]
    predictions['ARIMA_test'] = Y_test_arima[:min_len]
    predictions['ARIMA_pred'] = Y_pred_arima[:min_len]
    predictions['LSTM_test'] = Y_test_lstm[:min_len]
    predictions['LSTM_pred'] = Y_pred_lstm[:min_len]
    predictions['GRU_test'] = Y_test_gru[:min_len]
    predictions['GRU_pred'] = Y_pred_gru[:min_len]

    return predictions, name


#вовзращает кол-во затрачиваемой памяти и времени на обучение и предсказание каждой моделью значений 
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

#возвращает значение метрик для каждой отдельной акции и среднее значение для всех по всем моделям
def get_dataset_mean_metric(path, metric_func, n_stocks=10, opt=False):
    metrics_list = []
    metrics_with_names = {}
    processed_stocks = 0
    for file in tqdm(os.scandir(path)):
        if processed_stocks >= n_stocks:
            break
        print(file.path)
        metric, name = get_metrics(file.path, metric_func, opt)
        metrics_list.append(metric)
        metrics_with_names[name] = metric
        processed_stocks += 1
    if n_stocks > 1: 
        mean_metrics = np.mean(metric, axis=0)
    else:
        mean_metrics=metric
    return metrics_with_names, mean_metrics

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
