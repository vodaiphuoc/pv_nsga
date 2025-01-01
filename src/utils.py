from typing import Tuple
import numpy as np
import pandas as pd

class CostMapping(object):
    def __init__(self,
                cost_effi_path:str,
                cost_shgc_path:str
            ) -> None:
        
        # mapping from `effi` to its cost
        cost_effi_df = pd.read_csv(cost_effi_path)
        self.effi2cost = {
            cost_effi_df.at[row_ith,'Effi'].item(): cost_effi_df.at[row_ith,'Cost_Effi'].item()         
            for row_ith in range(len(cost_effi_df))
            }
        self.effi2cost_keys = list(self.effi2cost.keys())

        # mapping from `shgc` to its cost
        cost_shgc_df = pd.read_csv(cost_shgc_path)
        self.shgc2cost = {
            cost_shgc_df.at[row_ith,'SHGC'].item(): cost_shgc_df.at[row_ith,'Cost_SHGC'].item() 
            for row_ith in range(len(cost_shgc_df))
            }
        self.shgc2cost_keys = list(self.shgc2cost.keys())
    
    def search(self, 
               effi_value: float, 
               shgc_value: float
               )->Tuple[float]:
        effi_min_idx = np.argmin([abs(_effi - _default_effi) for _effi, _default_effi in 
                                zip([effi_value]*len(self.effi2cost), self.effi2cost_keys)]
                                )
        
        shgc_min_idx = np.argmin([abs(_shgc - _default_shgc) for _shgc, _default_shgc in 
                                zip([shgc_value]*len(self.shgc2cost), self.shgc2cost_keys)]
                                )
        
        return (self.effi2cost[self.effi2cost_keys[effi_min_idx]],
                self.shgc2cost[self.shgc2cost_keys[shgc_min_idx]]
        )
