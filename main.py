from utils.dataset_predictions import get_dataset_mean_mae
import pandas as pd
import os

if __name__ == '__main__':
    n_stocks = 10
    full_maes, mean_mae = get_dataset_mean_mae(path = "data/stocks_s&p500", n_stocks = n_stocks)
    models_maes = {'ESN': mean_mae[0], 'DESN': mean_mae[1], 'HESN': mean_mae[2],
                   'ARIMA': mean_mae[3], 'LSTM': mean_mae[4], 'GRU': mean_mae[5]}
    pd.DataFrame(full_maes).to_csv(f"{n_stocks}stocks_mae_opt.csv")
    pd.DataFrame(models_maes, index=[0]).to_csv(f"{n_stocks}stocks_mean_mae_opt.csv")