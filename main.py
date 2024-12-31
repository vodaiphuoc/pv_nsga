from sklearn.ensemble import RandomForestRegressor
import pickle
import pandas as pd
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass, field, asdict
import numpy as np
from copy import deepcopy
import functools

@dataclass(frozen=False)
class Individual:
    """
    Representation of an individual
    """
    # passive fields
    Orian: float	
    COP: float
    LPD: float
    UvR: float
    
    # active fields
    SHGC: float
    Effi: float
    
    PV_1: float
    PV_2: float
    PV_3: float
    PV_4: float

    # compute later
    fitness_energy: float = field(default=0)
    fitness_cost: float = field(default=0)


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

class NSGA(CostMapping):

    def __init__(self,
            checkpoint_path:str = 'checkpoint/model.pkl',
            total_data_path:str = 'assets/PV_DataOfficeSample.csv',
            cost_effi_path:str = 'assets/effi_cost.csv',
            cost_shgc_path:str = 'assets/shgc_cost.csv'
            ) -> None:
        super().__init__(cost_effi_path, cost_shgc_path)

        # deserialize model
        with open(checkpoint_path,'rb') as fp:
            self.model = pickle.load(fp)

        # total data
        Individual_fields = [ele for ele in list(Individual.__dataclass_fields__.keys()) 
                             if ele != 'fitness_energy' and ele != 'fitness_cost']

        self.total_data = pd.read_csv(total_data_path)[Individual_fields]

    
    def _init_population(self)->List[Individual]:
        """Convert data in dataframe to `Individual`"""
        return [
                Individual(**self.total_data.loc[row_ith].to_dict())
                for row_ith in range(len(self.total_data))
            ]

    def _compute_individual_fitness(self, individual: Individual)->None:
        # search nearest values
        unit_cost_effi, unit_cost_shgc = self.search(individual.Effi, individual.SHGC)
        # compute costs
        cost_effi = unit_cost_effi*(individual.PV_1 + individual.PV_2 + individual.PV_3 + individual.PV_4)
                
        cost_shgc = unit_cost_shgc*(4.0-(individual.PV_1 + individual.PV_2 + individual.PV_3 + individual.PV_4))
        
        # compute produce energy
        data_dict = asdict(individual)
        del data_dict['SHGC']
        del data_dict['fitness_energy']
        del data_dict['fitness_cost']

        energy_consumption = self.model.predict(pd.DataFrame(data_dict,index=[0]))

        individual.fitness_energy = energy_consumption.item()
        individual.fitness_cost = cost_effi + cost_shgc

        return None
    
    @staticmethod
    def _is_dominant(individual: Individual,other_individual: Individual)->bool:
        larger_or_equal = [individual.fitness_cost >= other_individual.fitness_cost,
                           individual.fitness_energy >= other_individual.fitness_energy]
        
        larger = [individual.fitness_cost > other_individual.fitness_cost,
                           individual.fitness_energy > other_individual.fitness_energy]

        if np.all(larger_or_equal) and np.any(larger):
            return True
        else:
            return False

    @staticmethod
    def calculate_pareto_fronts(population: List[Individual])->List[np.ndarray]:
    
        # Calculate dominated set for each individual
        domination_sets = []
        domination_counts = []
        for fitnesses_1 in population:
            
            current_dimination_set = set()
            domination_counts.append(0)

            for i,fitnesses_2 in enumerate(population):
                if NSGA._is_dominant(fitnesses_1,fitnesses_2):
                    current_dimination_set.add(i)
                
                elif NSGA._is_dominant(fitnesses_2,fitnesses_1):
                    domination_counts[-1] += 1

            domination_sets.append(current_dimination_set)

        domination_counts = np.array(domination_counts)
        fronts = []
        while True:
            current_front = np.where(domination_counts==0)[0]
            if len(current_front) == 0:
                break

            fronts.append(current_front)

            for individual in current_front:
                domination_counts[individual] = -1
                dominated_by_current_set = domination_sets[individual]
                for dominated_by_current in dominated_by_current_set:
                    domination_counts[dominated_by_current] -= 1
                
        return fronts
    
    @staticmethod
    def fronts_to_nondomination_rank(fronts):
        nondomination_rank_dict = {}
        for i,front in enumerate(fronts):
            for x in front:   
                nondomination_rank_dict[x] = i
        return nondomination_rank_dict 

    @staticmethod
    def calculate_crowding_metrics(population: List[Individual], 
                                   fronts:List[np.ndarray]
                                   )->List[float]:
        population = deepcopy(population)
        # normalize fitness values
        all_energy = [indi.fitness_energy for indi in population]
        min_energy = min(all_energy)
        max_energy = max(all_energy)
        energy_range = max_energy - min_energy

        all_cost = [indi.fitness_cost for indi in population]
        min_cost = min(all_cost)
        max_cost = max(all_cost)
        cost_range = max_cost - min_cost

        for individual in population:
            individual.fitness_energy = (individual.fitness_energy - min_energy)/energy_range
            individual.fitness_cost = (individual.fitness_cost - min_cost)/cost_range

        # get crowding metrics
        crowding_metrics = [0]*len(population)

        for front in fronts:
            # for cost
            sorted_front = sorted(front,key = lambda x : population[x].fitness_cost)
            crowding_metrics[sorted_front[0]] = np.inf
            crowding_metrics[sorted_front[-1]] = np.inf

            if len(sorted_front) > 2:
                for i in range(1,len(sorted_front)-1):
                    crowding_metrics[sorted_front[i]] += \
                        population[sorted_front[i+1]].fitness_cost - population[sorted_front[i-1]].fitness_cost
                
            # for energy
            sorted_front = sorted(front,key = lambda x : population[x].fitness_energy)
            crowding_metrics[sorted_front[0]] = np.inf
            crowding_metrics[sorted_front[-1]] = np.inf

            if len(sorted_front) > 2:
                for i in range(1,len(sorted_front)-1):
                    crowding_metrics[sorted_front[i]] += \
                        population[sorted_front[i+1]].fitness_energy - population[sorted_front[i-1]].fitness_energy

        return crowding_metrics




    def run(self):
        population = self._init_population()

        for indi in population:
            self._compute_individual_fitness(indi)

        fronts = NSGA.calculate_pareto_fronts(population)
        print(fronts)
        
        crowding_metrics = NSGA.calculate_crowding_metrics(population,fronts)
        print(crowding_metrics)

        nondomination_rank_dict = NSGA.fronts_to_nondomination_rank(fronts)
        


if __name__ == '__main__':
    a = NSGA()
    a.run()