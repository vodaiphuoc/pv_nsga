from sklearn.ensemble import RandomForestRegressor
import pickle
import pandas as pd
import random
from typing import List, Tuple, Dict, Literal
import numpy as np
from copy import deepcopy
from dataclasses import asdict

from src.data_models import Individual, GA_params, get_individual_fields, get_field_min_max
from src.utils import CostMapping
from src.operations import GA_Operations
from tqdm import tqdm

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
        # self.total_data = pd.read_csv(total_data_path)[get_individual_fields(type='init')]

    
    def _init_population(self, population_size:int)->List[Individual]:
        """Convert data in dataframe to `Individual`"""
        population = []

        for _ in range(population_size):
            init_params = {}
            for k in get_individual_fields(type='init'):
                _min_max = get_field_min_max(k)
                assert len(_min_max) == 2
                init_params[k] = random.uniform(_min_max['min'], _min_max['max'])
            population.append(Individual(**init_params))

        return population

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
        """
        Convert Pareto fronts to non-dominant rank as dict
        """
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

    def run_one_generation(self, 
                           population: List[Individual], 
                           param: GA_params
                           )->List[Individual]:
        # also called `Non-dominated sorting`
        fronts = NSGA.calculate_pareto_fronts(population)
        
        crowding_metrics = NSGA.calculate_crowding_metrics(population,fronts)

        nondomination_rank_dict = NSGA.fronts_to_nondomination_rank(fronts)

        # parent selection
        parent_indicies = GA_Operations.parent_selection(type= param.parent_selection_type,
                                                nondomination_rank_dict= nondomination_rank_dict, 
                                                crowding_metrics= crowding_metrics,
                                                half_pop_size=param.half_pop_size
                                                )

        parents = [population[idx] for idx in parent_indicies]
        
        # Offspring creation
        offspring = []
        for parent_ith in range(0, len(parents)//2):
            # normal crossover in GA
            offspring1, offspring2 = GA_Operations.crossover(
                parent1=parents[parent_ith*2],
                parent2=parents[parent_ith*2+1],
                crossover_rate= param.crossover_rate
                )
            # normal mutation in GA
            offspring1 = GA_Operations.mutation(offspring1, param.mutation_rate)
            offspring2 = GA_Operations.mutation(offspring2, param.mutation_rate)

            offspring.extend([offspring1, offspring2])

        # get fitness values of new offsprings        
        for indi in offspring:
            self._compute_individual_fitness(indi)

        # Combine population and offspring
        combined_population = population + offspring
        return combined_population

    def run(self, param = GA_params()):
        population = self._init_population(param.population_size)

        for indi in population:
            self._compute_individual_fitness(indi)

        for _ in tqdm(range(param.generations), total= param.generations):
            population = self.run_one_generation(population,param)

        # get final fronts
        final_fronts = NSGA.calculate_pareto_fronts(population)

        print(final_fronts)



if __name__ == '__main__':
    a = NSGA()
    a.run()