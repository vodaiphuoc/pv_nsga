from typing import Tuple, List
import numpy as np
import pandas as pd
from src.data_models import Individual
import matplotlib.pyplot as plt
import matplotlib

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

def visualize_fronts(fronts: List[np.ndarray], 
                     population: List[Individual], 
                     limit_num_fronts:int = 10):
    if limit_num_fronts <= len(fronts):
        fronts = fronts[:limit_num_fronts]
    
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(fronts)))
    for ith, (front, color) in enumerate(zip(fronts, colors)):
        # x axis is `cost`
        x_data = [ population[ith].fitness_cost for ith in front]
        # y axis is `energy`
        y_data = [ population[ith].fitness_energy for ith in front]

        plt.scatter(x_data, y_data, color=color, label=f'front {ith+1}')

    plt.xlabel('Cost')
    plt.ylabel('Energy')

    plt.legend()
    plt.savefig('result.png')