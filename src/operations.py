from src.data_models import Individual, get_individual_fields, get_field_min_max
import random
from copy import deepcopy
from typing import Tuple, List, Dict, Literal
import functools
import numpy as np

class GA_Operations(object):

    @staticmethod
    def parent_selection(type:Literal['binary_tournament_selection','sort_direct_selection'],
                         nondomination_rank_dict: Dict[int, int], 
                         crowding_metrics: List[float],
                         half_pop_size:int
                         ):
        def _nondominated_compare(a,b):
            # domination rank, smaller better
            if nondomination_rank_dict[a] > nondomination_rank_dict[b]:  
                return -1
            elif nondomination_rank_dict[a] < nondomination_rank_dict[b]:
                return 1
            else:
                # crowding metrics, larger better
                if crowding_metrics[a] < crowding_metrics[b]:   
                    return -1
                elif crowding_metrics[a] > crowding_metrics[b]:
                    return 1
                else:
                    return 0
        
        total_indicies = list(range(len(crowding_metrics)))

        if type == 'binary_tournament_selection':
            rand_selected_index = np.random.choice(total_indicies, 
                                                   size= (half_pop_size,2), 
                                                   replace= False)
            
            return [a 
                    if (_cmp:=_nondominated_compare(a,b)) == 1 
                    else b if _cmp == -1 else random.choice([a,b])
                    for (a,b) in rand_selected_index 
            ]

        else:
            return sorted(total_indicies,
                      key = functools.cmp_to_key(_nondominated_compare),
                      reverse=True)[:half_pop_size]

    @staticmethod
    def crossover(parent1: Individual, 
                  parent2: Individual,
                  crossover_rate:float
                  )->Tuple[Individual]:

        if random.random() < crossover_rate:
            
            total_fields = get_individual_fields(type='init')
            _break_fields = random.randint(1, len(total_fields)-1)
            crossover_fields =  total_fields[: _break_fields]
            
            offspring1_data = {k1: v1 
                               if k1 in crossover_fields else v2
                                for ((k1,v1),(_,v2)) 
                                in zip(parent1.init_to_dict.items(), 
                                       parent2.init_to_dict.items())
            }
            offspring2_data = {k1: v2 
                               if k1 in crossover_fields else v1
                                for ((k1,v1),(_,v2)) 
                                in zip(parent1.init_to_dict.items(), 
                                       parent2.init_to_dict.items())
            }
            assert len(offspring1_data) == 10
            offspring1 = Individual(**offspring1_data)
            offspring2 = Individual(**offspring2_data)
        else:
            offspring1 = Individual(**deepcopy(parent1.init_to_dict))
            offspring2 = Individual(**deepcopy(parent2.init_to_dict))

        assert offspring1.fitness_energy == 0.0
        return offspring1, offspring2

    @staticmethod
    def mutation(individual: Individual, mutation_rate:float)->Individual:
        mutated_param = deepcopy(individual.init_to_dict)

        for k, v in mutated_param.items():
            if random.random() < mutation_rate:
                _min_max = get_field_min_max(k)

                mutated_param[k] = random.uniform(_min_max['min'], _min_max['max'])

        return Individual(**mutated_param)
