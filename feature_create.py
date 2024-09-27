import pandas as pd
import numpy as np

class FeaturesCreate:
    
    def __init__(
        self,
        data: pd.DataFrame, 
        first_obs_day: str
    ):
        
        self.data = data
        self.first_obs_day = first_obs_day
    
    
    def add_features(
        self
    ) -> pd.DataFrame:
        
        self.mouth_features()
        self.days_features()
        
        return self.data
    
            
    def mouth_features(
        self
    ):
        
        data = self.data.copy()
        
        months = np.empty(shape=(12, len(data)))
        months.fill(0)
        
        for i in range(len(data)):
            months[data.Date[i].month - 1][i] = 1
        
        for i in range(len(months)):
            data[f'month_{i}'] = months[i]
        
        self.data = data
        
    
    def days_features(
        self
    ):
        """_summary_

        Args:
            data (pd.DataFrame): Market data
            first_obs_day (str): First day in obseve (Example: monday)

        Returns:
            pd.DataFrame: New dataframe with features.
        """
        
        data = self.data.copy()
        
        first_days_idxs = {'first_monday_idx': 0, 
                        'first_tuesday_idx' : 1, 
                        'first_wednesday_idx' : 2, 
                        'first_thursday_idx' : 3, 
                        'first_friday_idx' : 4}

        first_obs_day = self.first_obs_day.upper()
        
        
        if first_obs_day == 'monday'.upper():
            first_days_idxs = {k: (v)%5 for k,v in first_days_idxs.items()}
            
        elif first_obs_day == 'tuesday'.upper():
            first_days_idxs = {k: (v-1)%5 for k,v in first_days_idxs.items()}
            
        elif first_obs_day == 'wednesday'.upper():
            first_obs_day = {k: (v-2)%5 for k,v in first_days_idxs.items()}
            
        elif first_obs_day == 'thursday'.upper():
            first_days_idxs = {k: (v-3)%5 for k,v in first_days_idxs.items()}
            
        elif first_obs_day == 'friday'.upper():
            first_days_idxs = {k: (v-4)%5 for k,v in first_days_idxs.items()}
            
        
        mondays = pd.Series(data=[0.]*len(data), name='is_monday')
        tuesdays = pd.Series(data=[0.]*len(data), name='is_tuesday')
        wednesdays = pd.Series(data=[0.]*len(data), name='is_wednesday')
        thursdays = pd.Series(data=[0.]*len(data), name='is_thursday')
        fridays  = pd.Series(data=[0.]*len(data), name='is_friday')
        
                
        for i in range(len(data.Date)):
            if (data.Date[i] - data.Date[first_days_idxs['first_monday_idx']]).days % 7 == 0:
                mondays[i] = 1
            else:
                mondays[i] = 0
        
        for i in range(len(data.Date)):
            if (data.Date[i] - data.Date[first_days_idxs['first_tuesday_idx']]).days % 7 == 0:
                tuesdays[i] = 1
            else:
                tuesdays[i] = 0
        
        for i in range(len(data.Date)):
            if (data.Date[i] - data.Date[first_days_idxs['first_wednesday_idx']]).days % 7 == 0:
                wednesdays[i] = 1
            else:
                wednesdays[i] = 0
                
        for i in range(len(data.Date)):
            if (data.Date[i] - data.Date[first_days_idxs['first_thursday_idx']]).days % 7 == 0:
                thursdays[i] = 1
            else:
                thursdays[i] = 0
        
        for i in range(len(data.Date)):
            if (data.Date[i] - data.Date[first_days_idxs['first_friday_idx']]).days % 7 == 0:
                fridays[i] = 1
            else:
                fridays[i] = 0
                
                
        data['is_monday'] = mondays
        data['is_tuesday'] = tuesdays
        data['is_wednesday'] = wednesdays
        data['is_thursday'] = thursdays
        data['is_friday'] = fridays

        self.data = data