
import os
import sys
from utils import load_config
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
from dataframe_handler import create_dict, generation_aa_library
from GA import run_genetic_algorithm


def initialize_globals():


    global config, data_paths, ga_settings, smiles_csv_path, top_set, bbdict, ga_params

    config = load_config()  # Ensure this function is properly defined to load your config.json
    data_paths = config['data_paths']
    ga_settings = config['genetic_algorithm_settings']

    # Paths and settings
    smiles_csv_path = data_paths['smiles_csv']
    top_set = generation_aa_library(price_limit=ga_settings['price_limit'], smiles_csv_path=smiles_csv_path)
    bbdict = create_dict(top_set)

    ga_params = {
    'num_epochs': ga_settings['num_epochs'],
    'convergence_percentage': ga_settings['convergence_percentage'],
    'mutation_fraction': ga_settings['mutation_fraction'],
    'crossover_fraction': ga_settings['crossover_fraction'],
    'long_liver_fraction': ga_settings['long_liver_fraction'],
    'population_size': ga_settings['population_size'],
    'min_len': ga_settings['min_len'],
    'max_len': ga_settings['max_len'],
    'type_of_cyclization': ga_settings['type_of_cyclization'],
    'plot': False  # Set this according to your needs, e.g., for visual output or not
    }

def main():
    initialize_globals()
    GA_population = run_genetic_algorithm(top_set, bbdict, **ga_params)
    return GA_population

if __name__ ==  '__main__':
    main()