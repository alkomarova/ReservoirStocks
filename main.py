from utils.dataset_predictions import get_dataset_mean_mae, get_dataset_time_memory
import pandas as pd
import os


def create_files_mse_n_stocks(n_stocks):
    full_maes, mean_mae = get_dataset_mean_mae(path="data/stocks_s&p500", n_stocks=n_stocks)
    models_maes = {'ESN': mean_mae[0], 'DESN': mean_mae[1], 'HESN': mean_mae[2],
                   'ARIMA': mean_mae[3], 'LSTM': mean_mae[4], 'GRU': mean_mae[5]}
    pd.DataFrame(full_maes).to_csv(f"{n_stocks}stocks_mae_opt.csv")
    pd.DataFrame(models_maes, index=[0]).to_csv(f"{n_stocks}stocks_mean_mae_opt.csv")


def create_file_time_memory_n_stocks(n):
    n_stocks = n
    time, memory = get_dataset_time_memory(path="data/stocks_s&p500", n_stocks=n_stocks)
    time_memory = {'ESN': [time[0], memory[0]], 'DESN': [time[1], memory[1]], 'HESN': [time[2], memory[2]],
                   'ARIMA': [time[3], memory[3]], 'LSTM': [time[4], memory[4]], 'GRU': [time[5], memory[5]]}
    pd.DataFrame(time_memory).to_csv(f"{n_stocks}stocks_time_memory_opt.csv")


if __name__ == '__main__':
    create_file_time_memory_n_stocks(10)
