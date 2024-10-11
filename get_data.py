import pandas as pd
import yfinance as yf
import numpy as np

# tickers_list = ['AAPL', 'WMT', 'IBM', 'MU', 'BA', 'AXP', 'NVDA', 'AMGN', 'F', 'BP', 'META', 'AMD', 'INTC']
def get_data(
    tickers_list: list,
    folder: str = 'data'
) -> None:
    """Downloading stocks data and save to folder.

    Args:
        tickers_list (list): List of stocks tickers,
        folder (str, optional): Folder to download. Defaults to 'data'.
    """

    # Import pandas
    data = pd.DataFrame(columns=tickers_list)

    # Fetch the data

    for ticker in tickers_list:
        data[ticker] = yf.download(ticker,'2014-01-01','2022-08-01')['Adj Close']

    data.to_csv(f'{folder}/stocks.csv')

    portfolio = np.sum(np.array([data.iloc[:, i] for i in range(len(data.columns))]), axis=0)
    portfolio = pd.DataFrame({'portfolio': portfolio}, index=data.index)

    portfolio.to_csv(f'{folder}/portfolio.csv')