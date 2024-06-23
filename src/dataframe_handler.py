import os
import sys
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
from rdkit import Chem
# from rdkit.Chem import AllChem
# from rdkit.Chem import rdmolops
# from rdkit.Chem import rdCoordGen
import pandas as pd
import numpy as np
from filtering_db import filter_for_non_polar_AA, price_filter
path_to_cyclpept_ml = os.path.abspath('cyclpept-ML-models')
sys.path.append(path_to_cyclpept_ml)
from models.apply_model import make_predictions

def generation_aa_library(price_limit, smiles_csv_path):
    df = pd.read_csv(smiles_csv_path)  # Use the SMILES CSV path from config
    df['Name'] = df['Name'].str.replace(' ', '_', regex=False)
    non_polar_df = filter_for_non_polar_AA(df)
    prefiltered_AA_library = price_filter(non_polar_df, price_limit)
    return prefiltered_AA_library

def create_dict(df):
    df.reset_index(drop=True, inplace=True)
    bbdict = {}
    for idx, aa in enumerate(df['Name']):
        bbdict[aa] = df['SMILES'][idx]

    bbdict['res2'] = 'SCCCNC(OCC1C(C=CC=C2)=C2C3=C1C=CC=C3)=O'
    bbdict['mpa'] = 'O=C(O)CCS'
    bbdict['linker1'] = 'O=C(CCl)CCl'

    return bbdict

def final_sequence(aa_sequence):    
    final_sequence = 'mpa'+' '+ aa_sequence+ ' ' + 'res2'
    return final_sequence

def to_smiles(list_of_mols):
    smiles_list = []
    for mol in list_of_mols:
        smiles_list.append(Chem.MolToSmiles(mol))
    return smiles_list

def generate_permeability_df(list_structures):
    smiles_list = to_smiles(list_structures)
    
    permeability = make_predictions(smiles_list)
    
    if isinstance(permeability, np.ndarray):
        permeability = permeability.tolist()

    df = pd.DataFrame({
        'Mol': list_structures,
        'SMILES': smiles_list,
        'Permeability': permeability
    })
    
    return df


def filter_valid_smiles(smiles_list):
    """
    Filter the list of SMILES strings to include only those that can be
    converted to valid RDKit molecule objects.
    """
    valid_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:  # If the molecule is not None
            valid_smiles.append(smi)
    return valid_smiles

def fix_nan_permeability(population_df):
    """
    Fixes NaN values in the 'Permeability' column of the DataFrame by predicting
    new values based on the SMILES representation of the molecules.

    Parameters:
    - population_df (pd.DataFrame): The DataFrame containing the population data.
    - make_predictions_func (function): The function used to make permeability predictions.

    Returns:
    - None: The function updates the DataFrame in place.
    """
    # Identify rows with NaN in the 'Permeability' column
    nan_rows = population_df[population_df['Permeability'].isna()]

    # Process each identified row
    for index, row in nan_rows.iterrows():
        broken_mol = row['Mol']  # Get the molecule object
        broken_mol.UpdatePropertyCache()
        Chem.SanitizeMol(broken_mol)
        # Chem.SanitizeMol(broken_mol)

        broken_smiles = Chem.MolToSmiles(broken_mol)  # Convert to SMILES
        broken_smiles_list = [broken_smiles]  # Create a list with the SMILES string
        fixed_permeability = make_predictions(broken_smiles_list)  # Fix the permeability
        fixed_permeability_value = fixed_permeability[0]  # Assuming make_predictions returns a list
        # Update the DataFrame
        population_df.at[index, 'Permeability'] = fixed_permeability_value  # Update 'Permeability'
        population_df.at[index, 'SMILES'] = broken_smiles


def generate_permeability_df(peptides_and_structures):
    """
    Generates a DataFrame with sequences, valid structures, SMILES, and predicted permeability.
    Attempts to fix NaN values in the 'Permeability' column.

    Parameters:
    - peptides_and_structures (list): List of tuples containing peptide sequences and their structures.

    Returns:
    - pd.DataFrame: DataFrame containing sequences, structures, SMILES, and permeability predictions.
    """
    sequences = []
    valid_structures = []  # Stores only valid structures
    for seq, struct in peptides_and_structures:
        if struct is not None:  # Check if the structure is valid
            sequences.append(seq)
            valid_structures.append(struct)

    # Convert only valid structures to SMILES
    smiles_list = [Chem.MolToSmiles(mol) for mol in valid_structures]

    # Filter out invalid SMILES strings
    valid_smiles_list = filter_valid_smiles(smiles_list)
    if len(valid_smiles_list) < len(smiles_list):
        missing_count = len(smiles_list) - len(valid_smiles_list)
        print(f"Warning: {missing_count} SMILES strings were invalid and excluded from predictions.")

    # Make predictions using the list of valid SMILES strings
    permeability = make_predictions(valid_smiles_list) if valid_smiles_list else []

    if isinstance(permeability, np.ndarray):
        permeability = permeability.tolist()

    # Generate DataFrame
    df = pd.DataFrame({
        'Sequence': sequences,
        'Mol': valid_structures,
        'SMILES': valid_smiles_list + [''] * (len(sequences) - len(valid_smiles_list)),  # Pad the SMILES list with empty strings if any were filtered out
        'Permeability': permeability + [np.nan] * (len(sequences) - len(valid_smiles_list))  # Pad the permeability list with NaNs if any SMILES were filtered out
    })

    # Attempt to fix NaN values in the 'Permeability' column
    max_attempts = 5
    if df['Permeability'].isna().any():
        attempts = 0
        while df['Permeability'].isna().any() and attempts < max_attempts:
            try:
                fix_nan_permeability(df)
                attempts += 1
            except Exception as e:
                print(f"Error fixing NaN values on attempt {attempts + 1}: {e}")
                attempts += 1  # Increment attempts to allow for retry up to max_attempts

    return df

