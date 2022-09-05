import os
import pandas as pd
from db.bin.data_tertiary import TertiaryData
from pathlib import Path
import warnings
from datetime import timedelta
from termcolor import colored

from scipy.signal import butter,lfilter
# from db.bin.indicators import lowpass_filter ### returns 0 for all the currencies when called getArimaGarch(train_df)
from tqdm import tqdm

from statsmodels.tsa.stattools import arma_order_select_ic
from statsmodels.tsa.arima.model import ARIMA
import arch

class ArimaGarchFeatures(TertiaryData):

    def __init__(self):
        super().__init__()

    def butter_lowpass_filter(self, data, cutoff, fs, order):
        fs = 30.0       # sample rate, Hz
        cutoff = 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
        nyq = 0.5 * fs  # Nyquist Frequency
        order = 2       # sin wave can be approx represented as quadratic
        # n = int(T * fs) # total number of samples

        normal_cutoff = cutoff / nyq
        # Get the filter coefficients
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = lfilter(b, a, data)
        return y

    def smoothData(self, data):
        data = self.butter_lowpass_filter(data, order=8, cutoff=0.2)
        return data

    def optimiseParams(self, data):

        train_for_params = data[-4320:] ### PARAMS are based on the 1 month of hourly historical data
        params = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=Warning)
            for c in list(train_for_params.columns):
                res = arma_order_select_ic(train_for_params[c].values, max_ar=4, max_ma=2, ic='aic')
                # print(f'{c} mle retvals: {res.mle_retvals}')
                params[c] = (res.aic_min_order[0], 0, res.aic_min_order[1])
        return params

    def getArimaGarch(self, training_subset):
        '''
        Get volatility with ARIMA+GARCH prediction pd.Series for the all data.columns
        '''
        DATA_PATH= os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../","data/merge")
        )
        print("Creating ArimaGarch row")

        training_subset = pd.read_csv(os.path.join(DATA_PATH, "tertiary", "logs_.csv"), parse_dates=0, index_col=0)
        # rets = pd.read_csv(os.path.join(DATA_PATH, "tertiary", "rets_.csv"), parse_dates=0, index_col=0)

        training_subset = self.smoothData(training_subset)
        training_subset.index = pd.to_datetime(training_subset.index)

        ### Calc params with the above function (last 6 months of the data)
        # params = self.optimiseParams(training_subset)
        ### Default params from last six months of the trainset
        params = {'AUD': (1, 0, 2), 'CAD': (0, 0, 2), 'CHF': (0, 0, 2), 'EUR': (0, 0, 2), 'GBP': (0, 0, 2), 'JPY': (0, 0, 2), 'NZD': (2, 0, 0), 'USD': (2, 0, 0)}

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=Warning)
            warnings.warn = lambda *a, **kw: False

            arima_forc_dict = {'next_hour_index': []}
            vol_dict = {'next_hour_index': []}
            for column in list(training_subset.columns):

                arima = ARIMA(training_subset[column], order=params[column])
                arima_fit = arima.fit()
                forecast_results = arima_fit.get_forecast(1, alpha=0.05) # get_forecast( ) define the length of predictions
                # let's use train.index to populate indexes with dates
                arima_forc_dict['next_hour_index'] = training_subset.index[-1] + timedelta(hours=1) # hourly indexes we predict
                arima_forc_dict[column] = list(forecast_results.predicted_mean) # forecast values
                print(forecast_results.predicted_mean.index)

                # Use residuals in the Garch
                garch = arch.arch_model(arima_forc_dict[column], p=1, q=1)
                garch_fitted = garch.fit()
                # Use GARCH to predict the residual
                garch_forecast = garch_fitted.forecast(horizon=1,reindex=False)
                predicted_et = garch_forecast.mean['h.1'].iloc[-1]

                # Combine both models' output: yt = mu + et
                volatility = arima_forc_dict[column] + predicted_et
                # Put them into dict
                vol_dict['next_hour_index'] = arima_forc_dict['next_hour_index']
                vol_dict[column] = volatility
                # print(colored(f'\n \n Predicted volatility for {column} currency is: {volatility}','red'))

            vol_df = pd.DataFrame.from_dict(vol_dict).set_index('next_hour_index')
            vol_df.index = pd.to_datetime(vol_df.index.values)
            # print(colored(f'\n Output at date time {vol_df.index} Dataframe {vol_df}','red'))
            return vol_df

    def evaluateArimaGarch(self, data, last_hours=336):
        '''
        Loop getVolatility function over the rows and produce df with predictions
        Create dataframe and file with the all prediction and create a file
        '''
        ### Split to limit training dataset, 4230 is equal to 6 months depth of hourly data
        split = 4320
        #### Define i for window function and increasing number of rows
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=Warning)
            warnings.warn = lambda *a, **kw: False

            for i in tqdm(range(last_hours)): # 336 times splits the data
                ### Training subset for the getVolatility Arima+Garch model. Train on smaller portion
                train_df = data[-3*split-i:-last_hours-i]
                # train_df = train_df[-4320:]
                ### Get output for the each row with GetVolatility() using ARIMA+GARCH
                vol_row_df = self.getArimaGarch(train_df)
                ### Save row to csv file
                file_path = os.path.join(DATA_PATH, "arimagarch")
                Path(file_path).mkdir(parents=True, exist_ok=True)
                # with open(os.path.join(file_path, "arimagarch.csv"), 'a') as f:
                vol_row_df.to_csv(os.path.join(file_path, "arimagarch.csv"), index = True, header = False, mode='a')
        return

    def getMasterArimaGarch(data, last_hours=336):
        '''
        Loop getVolatility function over the rows and produce df with predictions
        Create dataframe and file with the all prediction and create a file
        '''

        ### Split only to limit nr of rows output and time of processing
        split = 2160 # 3 months
        #### Define i for window function and increasing number of rows
        volatilities = {}
        for i in tqdm(range(last_hours)):
            train_df = data[-split-i:-last_hours-i]
            ### Optional training subset for the getVolatility Arima+Garch model. Train on smaller portion
            # train_df = train_df[-4320:]
            ### Get output for the each row with GetVolatility() using ARIMA+GARCH
            vol_row_df = self.getArimaGarch(train_df)
            volatilities[i] = vol_row_df.values.flatten().tolist()
            ### Increase i for next window for the trainset

        volatilities_df = pd.DataFrame(volatilities).T
        volatilities_df.columns = vol_row_df.columns

        print("\n### CREATING arimagarch.csv ###")
        file_path = os.path.join(DATA_PATH, "arimagarch")
        Path(file_path).mkdir(parents=True, exist_ok=True)
        data.to_csv(os.path.join(file_path, "arimagarch.csv"))

        return volatilities_df

if __name__ == "__main__":
    ArimaGarchFeatures().getMasterArimaGarch()
