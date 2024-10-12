import numpy as np
from scipy.stats import norm, t
from typing import Callable, Union, Literal
import pandas as pd
from numba import jit
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")


@jit(cache=True)
def calculate_es(
    data: np.ndarray,
    var: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    

    Args:
        data (np.ndarray): 
            A DataFrame of prices.
        alpha (Union[float, np.ndarray]): 
            Value at risk.

    Returns:
        Union[float, np.ndarray]: A list of Expected shorfall.
    """

    data_returns = np.diff(data) / data[1:]

    es = np.zeros_like(var)
    
    for key, val in enumerate(var):
        
        tail_rets = data_returns[data_returns < val]
        
        #check if tail_rets is empty
        if len(tail_rets) == 0:
            
            # warnings.warn('VaR level is too high. No returns < VaR level.')
            
            es[key] = val
            
            continue
        
        es[key] = np.mean(tail_rets)
        
    return es


def historical(
    data: np.ndarray,
    alpha: Union[float, np.ndarray] = np.array([0.05]),
) -> np.ndarray:
    """
    The historical method calculate VaR and ES.

    Args:
        data (np.ndarray): 
            A DataFrame of prices.
        alpha (Union[float, np.ndarray]): 
            Significance level (quantile level). Defaults to np.array([0.05]).

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
    alpha: Union[float, np.ndarray],
    std: Union[float, np.ndarray],
    distribution: Callable = norm,
    **kwargs,
) -> np.ndarray:
    """
    Variance-covariance method. Calculate VaR using mean and variance of TimeSeries data.
    
    Args:
        data (np.ndarray): 
            A DataFrame of prices.
        alpha (Union[float, np.ndarray]): 
            Significance level (quantile level). Defaults to np.array([0.05]).
        std (Union[float, np.ndarray]): 
            List of standart deviation.
        distribution (Callable, optional): 
            TimeSeries returns distribution from scipy.stats. Defaults to norm.

    Returns:
        np.ndarray: A list of objects with VaR and ES.
    """
    
    data_returns = np.diff(data) / data[1:]
    
    z_val = distribution.ppf(alpha, **kwargs)
    
    var = np.mean(data_returns) + z_val*std
    es = calculate_es(data=data, var=var)
    
    return np.concatenate((var, es))
    
def monte_carlo(
    data: np.ndarray,
    alpha: Union[float, np.ndarray],
    distribution: Callable = norm,
    n_sim: int = 10_000,
    **kwargs
) -> np.ndarray:
    """
    
    
    Args:
        data (np.ndarray): 
            A DataFrame of prices.
        alpha (Union[float, np.ndarray]): 
            Significance level (quantile level). Defaults to np.array([0.05]).
        distribution (Callable, optional): 
            TimeSeries returns distribution from scipy.stats. Defaults to norm.
        n_sim (int, optional): 
            Number of monte-carlo simulations. Defaults to 10_000.

    Returns:
        np.ndarray: A list of objects with VaR and ES.
    """
    data_returns = np.diff(data) / data[1:]
    
    sim_returns = distribution.rvs(size=n_sim, **kwargs)
    
    sim_returns = np.sort(sim_returns)
    
    var = np.array([-sim_returns[int(n_sim * (1-alpha))]
           for alpha in alpha])
    es = calculate_es(data=data_returns, var=var)
    
    return np.concatenate((var, es))


def garch(
    data: np.ndarray,
    alpha: Union[float, np.ndarray],
    vol: Literal['GARCH', 'ARCH', 'EGARCH', 'HARCH'] = 'GARCH',
    p: int = 1,
    o: int = 0,
    q: int = 1,
    distribution: Callable = norm,
    **kwargs,
) -> np.ndarray:
    """
    

    Args:
        data (np.ndarray): 
            A DataFrame of prices.
        alpha (Union[float, np.ndarray]): 
            Significance level (quantile level). Defaults to np.array([0.05]).
        vol (Literal[&#39;GARCH&#39;, &#39;ARCH&#39;, &#39;EGARCH&#39;, &#39;HARCH&#39;], optional): 
            volatily model. Defaults to 'GARCH'.
        p (int, optional): 
            Lag order of the symmetric innovation. Defaults to 1.
        o (int, optional): 
            Lag order of the asymmetric innovation. Defaults to 0.
        q (int, optional): 
            Lag order of lagged volatility or equivalent. Defaults to 1.

    Returns:
        np.ndarray: A list of objects with VaR and ES.
    """
    data_returns = np.diff(data) / data[1:]
    
    # distribution.__class__.__name__[:-4] because in scipy.stats dist named: norm_gen, t_gen and etc.
    model = arch_model(data_returns, vol=vol, p=p, o=o, q=q, **kwargs)
    model = model.fit(disp='off', **kwargs)
    
    cond_vol = model.conditional_volatility[-1]
    
    var = np.array([distribution.ppf(alpha) * cond_vol
           for alpha in alpha])
    
    es = calculate_es(data=data, var=var)
    
    return np.concatenate((var, es))
    