from utils.dataset_predictions import get_dataset_mean_metric, get_dataset_time_memory, get_predictions
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import os


def create_files_metric_n_stocks(n_stocks, metric_func, opt):
    full_mses, mean_mse = get_dataset_mean_metric(path="data/stocks_s&p500", n_stocks=n_stocks, metric_func=metric_func, opt=opt)
    models_maes = {'ESN': mean_mse[0], 'DESN': mean_mse[1], 'HESN': mean_mse[2],
                   'ARIMA': mean_mse[3], 'LSTM': mean_mse[4], 'GRU': mean_mse[5]}
    pd.DataFrame(full_mses).to_csv(f"{n_stocks}stocks_mse_opt.csv")
    pd.DataFrame(models_maes, index=[0]).to_csv(f"{n_stocks}stocks_mean_mse_opt.csv")


def create_file_time_memory_n_stocks(n):
    n_stocks = n
    time, memory = get_dataset_time_memory(path="data/stocks_s&p500", n_stocks=n_stocks)
    time_memory = {'ESN': [time[0], memory[0]], 'DESN': [time[1], memory[1]], 'HESN': [time[2], memory[2]],
                   'ARIMA': [time[3], memory[3]], 'LSTM': [time[4], memory[4]], 'GRU': [time[5], memory[5]]}
    pd.DataFrame(time_memory).to_csv(f"{n_stocks}stocks_time_memory_opt.csv")

def create_files_with_predictions_n_stocks(n_stocks, path):
    processed_stocks = 0
    for file in tqdm(os.scandir(path)):
        processed_stocks += 1
        preds, name = get_predictions(file.path)
        pd.DataFrame(preds).to_csv(f"results_stocks/{name}predictions_opt.csv")
        if processed_stocks >= n_stocks:
            break

if __name__ == '__main__':
    create_files_metric_n_stocks(1, metrics.mean_absolute_error, opt=False)
