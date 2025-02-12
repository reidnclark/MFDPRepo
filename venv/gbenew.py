

# Libraries
################################################################################
import pandas as pd
import numpy as np
import math
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch import arch_model
import matplotlib.pyplot as plt
import os

print(plt.get_backend())

print(f"Current Working Directory: {os.getcwd()}")

pd.set_option('display.max_rows', None)

################################################################################


# Functions
################################################################################
# 1. Retreive Data:

def retreiveData (ticker_input: str,
                    period_input: str,
                    sampling_interval_input: str
                    ) -> pd.DataFrame:

    raw_data = yf.download(ticker_input,
                            period=period_input,
                            interval=sampling_interval_input,
                            progress=False)
    
    return raw_data[['Close', 'Volume']]


################################################################################
# 2. Group Data by Date (Listify):

def group_byDate (raw_data: pd.DataFrame
                    ) -> list:

    raw_data['Date'] = raw_data.index.date
    grouped_data = [ intraday_instance for day, intraday_instance in raw_data.groupby('Date') ] # Note: data.groupby('Date') is dictifying (day, instance within day), hence 'i, data'.
    return grouped_data


################################################################################
# 3. Transform Data into Near-Final State, by computing the:
#    -  Volume-Weighted Average Price (VWAP)
#    -  Logarithmic Price Change (Log Return)
#    -  Short-Term Moving Average (SMA)
#    -  Long(er)-Term Moving Average (LMA)


def transform_groupedData(grouped_data: list,
                            sma_period: int,
                            lma_period: int
                            ) -> list[pd.DataFrame]:

    transformed_data = []
    for data in grouped_data:

        data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
        data['Log Return'] = np.log(data['Close'] / data['Close'].shift(1)) * 100
        data['SMA'] = data['Close'].rolling(window=sma_period).mean() # Calculate SMA (every 6 periods, or 30 minutes)
        data['LMA'] = data['Close'].rolling(window=lma_period).mean() # Calculate LMA (every 12 periods, or 1 hour)

        transformed_data.append(data) # Append to the transformed data list

    return transformed_data # Optional: Combine all daily transformed data into a single DataFrame --> transformed_data = pd.concat(transformed_data)


################################################################################
# 4. Retreive the Relative-Strength Index (Momentum Oscillator) Seperately:


def computeRSI (transformed_data: list[pd.DataFrame],
                rsi_interval: int
                ) -> list[pd.Series]:

    extract_values = [ transformed_data[i][['Close', 'Date']] for i in range(len(transformed_data)) ]

    daily_rsis = []
    for day in extract_values:

        extract_adj_close = pd.Series([ day['Close'].iloc[i].iloc[0] for i in range(len(day['Close'])) ])

        gain_or_loss = extract_adj_close.diff()
        gain = gain_or_loss.where(gain_or_loss > 0, 0)
        loss = gain_or_loss.where(gain_or_loss < 0, 0)

        avg_gain = gain.rolling(window=rsi_interval, min_periods=1).mean()
        avg_loss = loss.abs().rolling(window=rsi_interval, min_periods=1).mean()
        rs = avg_gain / (avg_loss)
        rsi = 100 - (100 / (1 + rs))

        daily_rsis.append(rsi)

    return daily_rsis


################################################################################
# 5.


def add_rsi_to_transformed_data(transformed_data, daily_rsis):
    """
    Adds RSI values to each DataFrame in transformed_data.

    Parameters:
    transformed_data (list of DataFrames): A list of DataFrames containing SMA and other columns.
    daily_rsis (list of pd.Series): A list of RSI Series, one for each DataFrame in transformed_data.

    Returns:
    list of DataFrames: The transformed data with added RSI column.
    """

    # Iterate through each DataFrame in transformed_data and corresponding RSI series
    for i in range(len(transformed_data)):
        df = transformed_data[i]
        rsi_series = daily_rsis[i]

        # Ensure that the number of RSI values matches the DataFrame length
        if len(rsi_series) > len(df):
            rsi_series = rsi_series[:len(df)]  # truncate if RSI series is too long
        elif len(rsi_series) < len(df):
            # Align the RSI series to the DataFrame length by filling missing values with NaN
            rsi_series = rsi_series.append(pd.Series([None] * (len(df) - len(rsi_series)), index=df.index))  # extend if too short

        # Add RSI values as a new column to the DataFrame
        df['RSI'] = rsi_series.values

    return transformed_data


################################################################################
# 6.


def plotter(transformed_data: list[pd.DataFrame]):
    # Number of days in the list
    num_days = len(transformed_data)
    # Decide how many plots per row (e.g., 3 per row)
    plots_per_row = 3
    # Calculate the number of rows required
    nrows = int(np.ceil(num_days / plots_per_row))
    ncols = plots_per_row
    # Create subplots with 'nrows' rows and 'ncols' columns
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 5), sharey=False)
    # Flatten the axes array to make iteration easier (in case it's a 2D array)
    axes = axes.flatten()
    # Loop through each day's data and plot
    for i, day_data in enumerate(transformed_data):

        ax = axes[i]

        # Create secondary y-axis for RSI (right axis)
        ax2 = ax.twinx()  # This creates a second y-axis sharing the same x-axis

        # Create third y-axis for Log Return (hidden axis)
        ax3 = ax.twinx()  # Third axis for Log Return

        # Offset the third axis to the right to avoid overlap with the second axis
        ax3.spines['right'].set_position(('outward', 60))

        # Plot Adj Close, VWAP, SMA, and LMA on the primary y-axis (left)
        ax.plot(day_data.index, day_data['Close'], color='blue', linewidth=1.1, label='Adj Close')
        ax.plot(day_data.index, day_data['VWAP'], color='darkblue', linewidth=1.4, label='VWAP')
        ax.plot(day_data.index, day_data['SMA'], color='magenta', linewidth=0.8, label='SMA',alpha=1)
        ax.plot(day_data.index, day_data['LMA'], color='black', linewidth=1.1, label='LMA',alpha=1)

        # Set y-axis limits for Adj Close to center it
        min_price = day_data['Close'].min().iloc[0]
        max_price = day_data['Close'].max().iloc[0]
        price_range = max_price - min_price
        ax.set_ylim(min_price - price_range * 0.1, max_price + price_range * 0.1)  # Add padding around the price

        # Plot RSI on the secondary y-axis (right)
        ax2.plot(day_data.index, day_data['RSI'], color='crimson', linewidth=0.5, label='RSI', alpha=0.7, marker='^', markersize=4)

        # Set y-axis limits for RSI between 0 and 100
        ax2.set_ylim(0, 100)

        # Plot Log Return on the third y-axis (hidden axis)
        ax3.plot(day_data.index, day_data['Log Return'], color='darkblue', linewidth=2, label='Log Return', alpha=0.1)

        # Set titles, labels, and legends for each subplot
        ax.set_title(f"Price, VWAP, Log Return, SMA, LMA, and RSI for {day_data['Date'].iloc[0]}")
        ax.set_xlabel('Time')
        ax.set_ylabel('Price / VWAP / SMA / LMA ($)')
        ax2.set_ylabel('RSI', color='red')

        # Hide the third axis (Log Return) from displaying its label and ticks
        ax3.set_ylabel('')
        ax3.set_yticks([])

        # Add legends for both y-axes
        ax.legend(loc='upper left', fontsize=8)  # Shrink legend
        ax2.legend(loc='upper right', fontsize=8)  # Shrink legend

        # # Add horizontal lines at RSI levels 70 and 30
        ax2.axhline(y=70, color='orangered', linestyle='--', linewidth=1.3, alpha=0.5)
        ax2.axhline(y=30, color='orangered', linestyle='--', linewidth=1.3, alpha=0.5)

        ax.grid(True, alpha=0.9)  # Grid for price / VWAP plot

    # If there are any empty subplots (when num_days < nrows * ncols), hide them
    for i in range(num_days, len(axes)):
        axes[i].axis('off')

    # Rotate x-axis labels for readability
    plt.xticks(rotation=90)

    # Adjust layout to prevent overlap of subplots
    plt.tight_layout()

    plt.savefig('/home/vboxuser/MFDPRepo/output_plot_new.png')

    plt.show(block=True)


################################################################################

data = retreiveData('NVDA', '1mo', '5m')
grouped_data = group_byDate(data)
transformed_data = transform_groupedData(grouped_data, 3, 5)
daily_rsis = computeRSI(transformed_data, 3)
transformed_data_with_rsi = add_rsi_to_transformed_data(transformed_data, daily_rsis)

plotter(transformed_data_with_rsi)


# Simple plot test
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("New Plot")
plt.show()