import numpy as np
import pandas as pd

hist_prices = [1.05,
                1.45,
                0.96,
                1.15,
                2.30,
                4.77,
                9.10,
                4.31,
                4.32,
                0.15,
                1.13,
                1.00,
                0.02
                ]

def logReturns (hist_prices: pd.Series):

    prices = [np.log((hist_prices[i+1]-hist_prices[i])/hist_prices[i]) for i in hist_prices]



