import yfinance as yf


ticker = yf.Ticker('NVDA')

print(ticker)

print(ticker.info)