import numpy as np
import pandas as pd
import yfinance as yf
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)





class Retreival:

    class AssertTickers:
        def __init__(self):
            self.ticker_list = []

        def inputTicker(self): 
            ticker_input = str(input(f'Enter Ticker Here: '))
            self.ticker_list.append(ticker_input.upper())

        def run(self):
            while True:
                self.inputTicker()
                add = str(input(f'Add Another? (Y/N): '))
                if add.upper() != 'Y':
                    break
            print(f'Final list of tickers: {self.ticker_list}')
            return self.ticker_list
        

    class FetchData:
        def __init__(self, 
                    ticker_list,
                    period_input,
                    sampling_frequency_input):
            
            self.ticker_list = ticker_list
            self.period_input = period_input
            self.sampling_frequency_input = sampling_frequency_input
        
        def fetch(self):
            data_dict = {}
            for ticker in self.ticker_list:
                yf_ticker = yf.Ticker(ticker)
                data = yf.download(ticker,
                                    period=self.period_input,
                                    interval=self.sampling_frequency_input,
                                    auto_adjust=True,
                                    progress=False)
                data_dict[ticker] = data[['Close','Volume']]
            return data_dict
        
        def chunk_by_day(self, data_dict):
            chunked_data_dict = {}
            for ticker, data in data_dict.items():
                daily_chunks = {}
                for date, day_data in data.groupby(data.index.date):
                    daily_chunks[date] = day_data
                chunked_data_dict[ticker] = daily_chunks
            return chunked_data_dict
        
        def run(self):
            data = self.fetch()
            chunked_data = self.chunk_by_day(data)
            return chunked_data



    def __init__(self):
        self.assert_tickers = self.AssertTickers()
    
    def motherRun(self):
        self.assert_tickers.run()
        period_input = str(input(
            'Enter Period Here (e.g., "5d", "1mo", "3mo", "9mo", "1y", ...): '))
        sampling_frequency_input = str(input(
            'Enter Sampling Frequency Here (e.g., "1m", "5m", "30m", "1h", "1d", ...): '))
        
        fetch_data = self.FetchData(self.assert_tickers.ticker_list,
                                    period_input,
                                    sampling_frequency_input)
        fetch_data.run()

class Transformation:

    class LogReturn:
        def __init__(self, data):
            self.data = data

        def compute_log_returns(self):
            log_return_dict = {}
            for ticker, daily_data in self.data.items():
                log_returns = {}
                for date, day_data in daily_data.items():
                    # Calculate log returns for the "Close" column
                    day_data['LogReturn'] = (day_data['Close'] / day_data['Close'].shift(1)).apply(np.log)
                    log_returns[date] = day_data[['LogReturn']]  # Store only the LogReturn column
                log_return_dict[ticker] = log_returns
            return log_return_dict

        def run(self):
            log_returns = self.compute_log_returns()
            return log_returns




# CLASS RETREIVAL
################################################################################################################
# allz = Allz()
# allz.motherRun()
test = Retreival().FetchData(['NVDA', 'AAPL'], '5d', '5m').run()
test_key = list(test['NVDA'].keys())[0]
#print(test['NVDA'][test_key])
# print(test['NVDA'])
# print(type(test['NVDA']))
################################################################################################################



# CLASS LOG RETURN
################################################################################################################

# Usage
# Assuming 'test' is the data you fetched earlier using Retreival
log_return_transformation = Transformation.LogReturn(test)
log_returns = log_return_transformation.run()

# Example: Print log returns for the first ticker
#print(log_returns['NVDA'])
test_log_returns_nvda = log_returns['NVDA']
listed_tlrnv = list(test_log_returns_nvda.values())[0]
#print(test_log_returns_nvda)

print(listed_tlrnv)

################################################################################################################

