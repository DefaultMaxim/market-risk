import numpy as np
from scipy.stats import norm, t
from typing import Callable, Union

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
            
        
    
    
    