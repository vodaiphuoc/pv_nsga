from pydantic.dataclasses import dataclass, Field
from pydantic import computed_field
from typing import Dict, List, Literal, Iterable, Tuple
import annotated_types

@dataclass(frozen=False)
class Individual:
    """
    Representation of an individual
    """
    # passive fields
    Orian: float = Field(ge=0.0, le=360.0)
    COP: float = Field(ge=2.6, le=7.0)
    LPD: float = Field(ge=7.0, le=13.0)
    UvR: float = Field(ge=0.622, le=3.0)
    
    # active fields
    SHGC: float = Field(ge=0.17, le=0.89)
    Effi: float = Field(ge=0.1, le=0.25)
    
    PV_1: float = Field(ge=0.0, le=1.0)
    PV_2: float = Field(ge=0.0, le=1.0)
    PV_3: float = Field(ge=0.0, le=1.0)
    PV_4: float = Field(ge=0.0, le=1.0)

    # compute later
    fitness_energy: float = Field(default=0.0)
    fitness_cost: float = Field(default=0.0)

    @computed_field
    def all_to_dict(self) -> Dict[str,float]:
        """
        Convert current `Individual` instance 
        to dictionary, include all fields
        """
        return self.__dict__


    @computed_field
    def init_to_dict(self) -> Dict[str,float]:
        """
        Convert current `Individual` instance 
        to dictionary, exclude `fitness_energy` and `fitness_cost`
        """
        return { k:v 
                for k, v in self.__dict__.items() 
                if k!= 'fitness_energy' and k!='fitness_cost'
                }


def get_individual_fields(type: Literal['all','init'])->List[str]:
    if type == 'init':
        return [ele for ele in list(Individual.__dataclass_fields__.keys()) 
                if ele != 'fitness_energy' and ele != 'fitness_cost'
            ]
    else:
        return list(Individual.__dataclass_fields__.keys())


def get_field_min_max(field_name:str)->Dict[str,float]:
    """Query min, max values of a field"""
    try:
        min_max = {}
        for cmp_type in Individual.__dataclass_fields__[field_name].default.metadata:
            if isinstance(cmp_type, annotated_types.Ge):
                min_max['min'] = cmp_type.ge
            elif isinstance(cmp_type, annotated_types.Le):
                min_max['max'] = cmp_type.le
            else:
                raise NotImplementedError
        return min_max
            
    except KeyError:
        print(f'field name must be in list: {list(Individual.__dataclass_fields__.keys())}')
        return None


@dataclass(frozen=True)
class GA_params:
    generations: int = 10
    population_size:int = 1000
    half_pop_size:int = 20
    parent_selection_type:str = 'binary_tournament_selection'
    crossover_rate:float = 0.5
    mutation_rate:float = 0.6