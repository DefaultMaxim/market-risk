import numpy as np
from scipy.stats import norm, t
from typing import Callable, Union
import pandas as pd
from numba import jit

class MarketRisk:
    """Class calculates Value at Risk and Expected shortfoll, base on distribution.
    """
    def __init__(
        self,
        data: np.ndarray,
    ) -> None:
        
        self.dat: np.ndarray = data
        self.data_returns: np.ndarray = np.diff(data) / data[1:]
        
    def risk_calc(
        self,
        distribution: Callable,
        alpha: float = 0.05,
        dx: float = 0.0001,
        horizon: int = 1,
    ) -> Union[tuple, np.ndarray]:
        """Calculates VaR and ES by predicted returns.

        Args:
            distribution (Callable): distribution student or normal.
            alpha (float, optional): significance level. Defaults to 0.05.
            dx (float, optional): steps to make linear space. Defaults to 0.0001.
            horizon (int, optional): horizon to forecast. Defaults to 1.

        Returns:
            Union[tuple, np.ndarray]: VaR and Expected shortfall
        """
        
        x = np.arange(-1, 1, dx)
        
        if distribution == norm:
            
            mu, sig = distribution.fit(self.data_returns)
            
            self.var = distribution.ppf(1 - alpha)*sig - mu
            self.es = 1/alpha * distribution.pdf(distribution.ppf(alpha))*sig - mu
        
        if distribution == t:
            
            nu, mu, sig = distribution.fit(self.data_returns)
            mu_norm, sig_norm = norm.fit(self.data_returns)
            xanu = distribution.ppf(alpha, nu)
            
            self.var = np.sqrt((nu - 2)/nu) * distribution.ppf(1-alpha, nu)*sig_norm - horizon*mu_norm
            self.es = -1/alpha * 1/(1 - nu) * (nu - 2 + xanu**2) * distribution.pdf(xanu, nu)*sig_norm - horizon*mu_norm
            
        return self.var, self.es


@jit(cache=True)
def calculate_es(
    data: np.ndarray,
    var: float,
) -> float:
    """
    Calculates Expected Shorfall.

    Args:
        data (np.ndarray): Prices data.
        var (float): Value at Risk value.

    Returns:
        float: Expected shorfall value.
        
    """
    # try to make calc for many vars
    data_returns = np.diff(data) / data[1:]

    tail_rets = data_returns[data_returns < var]
    es = np.mean(tail_rets)
        
    return es


def historical(
    data: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """
    The historical method calculate VaR and ES.

    Args:
        data (Union[pd.DataFrame, np.ndarray]): A DataFrame of prices.
        alpha (float): Significance level (quantile level). Defaults to 0.05.

    Returns:
        np.ndarray: A list of objects with VaR and ES.
        
    References:
    [investopedia](https://www.investopedia.com/articles/04/092904.asp)
    
    """
    
    data_returns = np.diff(data) / data[1:]
    
    var = np.percentile(data_returns, alpha*100, method='lower')
    es = calculate_es(data, var)
    
    return np.concatenate((var, es))


def parametric(
    data: np.ndarray,
    alpha: float,
    std: float,
    distribution: Callable = norm,
    **kwargs,
) -> np.ndarray:
    """
    

    Args:
        data (np.ndarray): _description_
        alpha (float): _description_
        std (float): _description_
        distribution (Callable, optional): _description_. Defaults to norm.

    Returns:
        np.ndarray: _description_
    """
    
    data_returns = np.diff(data) / data[1:]
    
    z_val = distribution.ppf(alpha, **kwargs)
    
    var = np.mean(data_returns) + z_val*std
    es = calculate_es(data=data, var=var)
    
    return np.concatenate((var, es))
    
def monte_carlo(
    data: np.ndarray,
    alpha: float,
    distribution: Callable = norm,
    n_sim: int = 10_000,
    **kwargs
) -> np.ndarray:
    """
    

    Args:
        data (np.ndarray): _description_
        alpha (float): _description_
        distribution (Callable, optional): _description_. Defaults to norm.
        n_sim (int, optional): _description_. Defaults to 10_000.

    Returns:
        np.ndarray: _description_
    """
    
    sim_returns = distribution.rvs(size=n_sim, **kwargs)
    
    sim_returns = np.sort(sim_returns)
    
    var = -sim_returns[int(n_sim * (1-alpha))]
    es = calculate_es(var)
    
    return np.concatenate((var, es))