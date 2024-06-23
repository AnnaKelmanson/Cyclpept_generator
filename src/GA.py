import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from dataframe_handler import create_dict, final_sequence, generate_permeability_df
import random

from reactions import build_peptide, create_df_row_for_child_structure, disulfide_cyc, linker_cyc

def generate_initial_cyclic_peptide_population(top_set, population_size, min_len, max_len, allow_repetitions, type_of_cyclization):
    bbdict = create_dict(top_set)
    linker = Chem.MolFromSmiles(bbdict['linker1'])
    peptides_and_structures = []

    unique_names = top_set['Name'].unique()
    for _ in range(population_size):
        sequence_length = random.randint(min_len, max_len)
        if allow_repetitions:
            sequence_names = random.choices(unique_names, k=sequence_length)
        else:
            sequence_names = random.sample(list(unique_names), min(sequence_length, len(unique_names)))
        sequence = ' '.join(sequence_names)
        full_sequence = final_sequence(sequence)
        linear_peptide = build_peptide(full_sequence, bbdict)
        if type_of_cyclization == 'disulfide':
            product = disulfide_cyc(linear_peptide)
        else:
            product = linker_cyc(linear_peptide, linker)
        peptides_and_structures.append((sequence_names, product))

    return peptides_and_structures


def calculate_df_metrics(df,threshold=-6):

    mean_permeability=df['Permeability'].mean()
    max_permeability=df['Permeability'].max()
    count_high_permeability = df['Permeability'].ge(threshold).sum()
    permeability_percentage = (count_high_permeability / len(df))

    return mean_permeability, max_permeability, permeability_percentage

def mutate_sequence(sequence, bbdict, delete=False, add=False):
    """
    Mutates, deletes, or adds a random amino acid in a given sequence.

    Parameters:
    - sequence (list of str): The original sequence of amino acids.
    - bbdict (dict): Dictionary where keys are amino acid names or identifiers.
                     The last three entries are not to be used for mutation or addition.
    - delete (bool): If True, a random amino acid is deleted instead of mutated.
    - add (bool): If True, a random amino acid is added to the sequence.

    Returns:
    - list of str: A new sequence with one amino acid mutated, deleted, or added.
    """
    if not sequence:
        raise ValueError("The input sequence is empty.")

    # Exclude the last three entries from the dictionary for mutation or addition
    valid_aas = list(bbdict.keys())[:-3]

    if delete:
        # Select a random position in the sequence to delete
        mutation_index = random.randint(0, len(sequence) - 1)
        return [aa for i, aa in enumerate(sequence) if i != mutation_index]
    elif add:
        # Select a random amino acid to add to the sequence
        if not valid_aas:
            raise ValueError("The bbdict does not contain enough entries for addition.")
        new_aa = random.choice(valid_aas)
        # Select a random position in the sequence to add
        add_index = random.randint(0, len(sequence))
        return sequence[:add_index] + [new_aa] + sequence[add_index:]
    else:
        # Select a random position in the sequence to mutate
        mutation_index = random.randint(0, len(sequence) - 1)
        current_aa = sequence[mutation_index]
        valid_aas = [aa for aa in valid_aas if aa != current_aa]
        
        if not valid_aas:
            raise ValueError("No valid amino acids available for mutation.")
        new_aa = random.choice(valid_aas)
        
        # Create a new sequence with the mutated amino acid
        mutated_sequence = sequence[:]
        mutated_sequence[mutation_index] = new_aa
        return mutated_sequence

def crossover(sequences):
    """
    Perform crossover between two sequences (lists) contained in a tuple at random points.
    
    Parameters:
    - sequences (tuple of lists): Tuple containing two sequences for crossover.
    
    Returns:
    - tuple: A tuple containing the two new sequences after crossover.
    """
    if len(sequences) != 2:
        raise ValueError("Input must be a tuple containing exactly two sequences.")

    seq1, seq2 = sequences

    # Check for minimum length of sequences to ensure valid crossover points
    if len(seq1) < 2 or len(seq2) < 2:
        raise ValueError("Both sequences must contain at least two elements for crossover.")

    # Random crossover points
    point1 = np.random.randint(1, len(seq1))
    point2 = np.random.randint(1, len(seq2))
    
    # Create new sequences by swapping parts from crossover points
    new_seq1 = seq1[:point1] + seq2[point2:]
    new_seq2 = seq2[:point2] + seq1[point1:]
    
    return (new_seq1, new_seq2)


def selection(df_input, fraction, method='elitist'):
    """
    Selects top entries based on the 'Permeability' column in a DataFrame.

    Args:
    df (pd.DataFrame): DataFrame containing a 'Permeability' column.
    fraction (float): Fraction of the DataFrame to select.
    method (str): Selection method, 'elitist' for top fraction directly, 'roulette' for probabilistic selection.

    Returns:
    pd.DataFrame: Selected subset of the original DataFrame.
    """
    # Normalize the 'Permeability' to positive scale
    df = df_input.copy()
    df['Normalized_Perm'] = df['Permeability'] - df['Permeability'].min()

    if method == 'elitist':
        # Select the top fraction based on normalized permeability
        n_select = int(len(df) * fraction)
        selected_df = df.nlargest(n_select, 'Normalized_Perm')
    elif method == 'roulette':
        # Convert to probability distribution (higher permeability has higher chance)
        df['Selection_Prob'] = df['Normalized_Perm'] / df['Normalized_Perm'].sum()
        selected_df = df.sample(n=int(len(df) * fraction), weights='Selection_Prob', replace=False)
    else:
        raise ValueError("Method should be either 'elitist' or 'roulette'.")

    return selected_df.drop(columns=['Normalized_Perm', 'Selection_Prob'], errors='ignore')


def calculate_similarity(seq1, seq2):
    # Calculate the number of dissimilar elements between two sequences
    return sum(el1 != el2 for el1, el2 in zip(seq1, seq2))

def select_parents(df, allow_cheating=False):
    # Calculate similarity for each pair of individuals
    pairs_similarity = {}
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            seq1 = df.iloc[i]['Sequence']
            seq2 = df.iloc[j]['Sequence']
            similarity = calculate_similarity(seq1, seq2)
            # Store the pair and their combined permeability
            pairs_similarity[(i, j)] = (similarity, df.iloc[i]['Permeability'] + df.iloc[j]['Permeability'])
    
    # Sort pairs by least similarity (most dissimilar first) and then by higher combined permeability
    sorted_pairs = sorted(pairs_similarity.items(), key=lambda item: (-item[1][0], item[1][1]))
    
    # Create a list to store the selected parent pairs
    selected_pairs = []
    
    # Keep track of selected parents if allow_cheating is False
    selected_parents = set()
    
    # Select the most dissimilar pairs with the highest combined Permeability values
    for pair, _ in sorted_pairs:
        parent1_index, parent2_index = pair
        if not allow_cheating:
            # Skip if either parent has already been selected
            if parent1_index in selected_parents or parent2_index in selected_parents:
                continue
            selected_parents.update([parent1_index, parent2_index])
        parent1_seq = df.iloc[parent1_index]['Sequence']
        parent2_seq = df.iloc[parent2_index]['Sequence']
        selected_pairs.append((parent1_seq, parent2_seq))
    
    return selected_pairs


def expand_2aa_seq(df, bbdict, type_of_cyclization):
    new_generation_df = df.copy()
    filtered_df = new_generation_df[new_generation_df['Sequence'].apply(lambda x: len(x) == 2)]
    unique_smiles = set()  # Set to keep track of unique SMILES strings
    expanded_dfs = []

    for seq in filtered_df['Sequence']:
        expansion_successful = False
        attempts = 0
        while not expansion_successful and attempts < 100:  # Allow up to 100 attempts for a unique mutation
            child_df = create_df_row_for_child_structure(mutate_sequence(seq, bbdict, add=True), bbdict, type_of_cyclization)
            child_smiles = child_df.loc[0, 'SMILES']

            if child_smiles not in unique_smiles:
                unique_smiles.add(child_smiles)
                expanded_dfs.append(child_df)
                expansion_successful = True
            attempts += 1

    # Concatenate all unique children DataFrames if any
    if expanded_dfs:
        mutated_df = pd.concat(expanded_dfs, ignore_index=True)
    else:
        mutated_df = pd.DataFrame()

    new_generation_df = new_generation_df[~new_generation_df['Sequence'].apply(lambda x: len(x) == 2)]
    new_generation_df = pd.concat([new_generation_df, mutated_df], ignore_index=True)
    return new_generation_df

def generate_children_df(pairs_of_sequences, bbdict, type_of_cyclization):
    children_dfs = []
    unique_smiles = set()
    children_needed = 2 * len(pairs_of_sequences)  # Expecting two children per pair

    # Loop through each pair and attempt to generate two unique children
    for pair in pairs_of_sequences:
        successful_children = 0
        attempts = 0
        while successful_children < 2 and attempts < 100:  # Allow up to 100 attempts per pair to find unique children
            # Generate children via crossover
            kids = crossover(pair)
            for kid in kids:
                if successful_children >= 2:
                    break  # Break if we already have two successful children from this pair
                child_df = create_df_row_for_child_structure(kid, bbdict, type_of_cyclization)
                child_smiles = child_df.loc[0, 'SMILES']
                if child_smiles not in unique_smiles:
                    children_dfs.append(child_df)
                    unique_smiles.add(child_smiles)
                    successful_children += 1
            attempts += 1

        if successful_children < 2:
            print(f"Warning: Unable to generate 2 unique children for a pair after 100 attempts. Generated only {successful_children} children.")

    # Concatenate all the single-row dataframes into one dataframe
    if children_dfs:
        new_generation_df = pd.concat(children_dfs, ignore_index=True)
    else:
        new_generation_df = pd.DataFrame()

    return expand_2aa_seq(new_generation_df, bbdict, type_of_cyclization)


def mutate_generation(df_population, fraction, bbdict, type_of_cyclization):
    """
    Mutates a fraction of the sequences in a DataFrame and replaces them with new mutated versions,
    ensuring that the total number of mutations equals the intended fraction of the population,
    and regenerating mutations if they result in duplicate SMILES.

    Parameters:
    - df_population (pd.DataFrame): DataFrame containing a column 'Sequence'.
    - fraction (float): Fraction of rows to mutate (0 to 1).
    - bbdict (dict): Dictionary of amino acids with the last three entries not to be used for mutation or addition.

    Returns:
    - pd.DataFrame: DataFrame with specified rows mutated, maintaining the fraction of unique mutations.
    """
    df = df_population.copy()
    num_rows = int(len(df) * fraction)
    random_indices = random.sample(range(len(df)), num_rows)
    unique_smiles = set(df['SMILES'])  # Assuming the SMILES column exists in the DataFrame
    processed_indices = set()

    for idx in random_indices:
        if idx in processed_indices:
            continue  # Skip if this index has already been successfully mutated
        sequence = df.loc[idx, 'Sequence']
        attempts = 0
        mutation_successful = False

        while not mutation_successful and attempts < 100:  # Allow up to 100 attempts to find a unique mutation
            # Determine mutation method based on sequence length
            if len(sequence) > 3:
                delete = random.choice([True, False])
                add = False
            elif len(sequence) == 3:
                delete = False
                add = random.choice([True, False])
            else:
                delete = False
                add = False

            mutated_sequence = mutate_sequence(sequence, bbdict, delete=delete, add=add)
            mutated_row_df = create_df_row_for_child_structure(mutated_sequence, bbdict, type_of_cyclization)
            mutated_smiles = mutated_row_df.loc[0, 'SMILES']

            # Check if the mutated SMILES is unique
            if mutated_smiles not in unique_smiles:
                unique_smiles.add(mutated_smiles)
                df.loc[idx] = mutated_row_df.loc[0]
                mutation_successful = True
                processed_indices.add(idx)  # Mark this index as successfully processed
            attempts += 1

        if not mutation_successful:
            print(f"Failed to generate a unique mutation for index {idx} after 100 attempts.")

    return df

    
def append_long_livers(population, new_generation, long_liver_fraction):
    """
    Appends the top fraction of samples from the population to the new generation based on 'Permeability',
    ensuring no duplicates are added based on their unique 'SMILES' string. If duplicates prevent meeting the
    fraction count, it iteratively adds the next best unique samples.
    
    Parameters:
    - population (pd.DataFrame): DataFrame containing the population data with a 'Permeability' column.
    - new_generation (pd.DataFrame): DataFrame representing the new generation to which the top samples will be appended.
    - long_liver_fraction (float): Fraction (0 to 1) of the top samples from the population to append to the new generation.
    
    Returns:
    - pd.DataFrame: The updated new generation DataFrame with the top samples appended.
    """
    num_samples = int(len(population) * long_liver_fraction)  # Calculate the number of top samples to select
    top_samples = population.sort_values(by='Permeability', ascending=False)  # Sort the population by 'Permeability'
    
    unique_smiles_new_gen = set(new_generation['SMILES'])  # Get the set of unique SMILES already in the new generation
    added_samples_count = 0  # Counter for added samples
    added_samples = []  # List to collect rows to be added

    # Iterate over the top samples and add until the required fraction is met
    for index, row in top_samples.iterrows():
        if added_samples_count >= num_samples:
            break
        if row['SMILES'] not in unique_smiles_new_gen:
            added_samples.append(row)
            unique_smiles_new_gen.add(row['SMILES'])
            added_samples_count += 1

    # Append the filtered top samples to the new generation DataFrame
    if added_samples:
        filtered_top_samples = pd.DataFrame(added_samples)
        updated_new_generation = pd.concat([new_generation, filtered_top_samples], ignore_index=True)
    else:
        updated_new_generation = new_generation

    return updated_new_generation

def run_genetic_algorithm(top_set, bbdict, **kwargs):
    # Parameters are extracted directly from kwargs; assume they are always provided
    num_epochs = kwargs['num_epochs']
    convergence_percentage = kwargs['convergence_percentage']
    mutation_fraction = kwargs['mutation_fraction']
    crossover_fraction = kwargs['crossover_fraction']
    long_liver_fraction = kwargs['long_liver_fraction']
    population_size = kwargs['population_size']
    min_len = kwargs['min_len']
    max_len = kwargs['max_len']
    type_of_cyclization = kwargs['type_of_cyclization']
    plot = kwargs.get('plot', True) 
    # Initialize metrics tracking
    previous_metrics = None
    metrics_change = float('inf')
    
    # Lists to store metrics for plotting
    mean_permeabilities = []
    max_permeabilities = []
    permeability_percentages = []
    
    # Generate initial population
    peptides_and_structures = generate_initial_cyclic_peptide_population(
        top_set, population_size=population_size, min_len=min_len, max_len=max_len, 
        allow_repetitions=False, type_of_cyclization=type_of_cyclization)  
    population = generate_permeability_df(peptides_and_structures)
    print(f"Initial population generated with {len(population)} rows.")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1} / {num_epochs} starts...")

        # Selection based on crossover
        selected = selection(population, fraction=crossover_fraction)
        print(f"{len(selected)} parents selected for crossover based on top fraction.")

        # Parent selection without cheating
        pairs_of_sequences = select_parents(selected, allow_cheating=False)
        print(f"{len(pairs_of_sequences)} pairs of parents selected for crossover.")

        # Generate children and apply mutations
        new_generation_df = generate_children_df(pairs_of_sequences, bbdict, type_of_cyclization)
        print(f"Children generated, total rows before mutation: {len(new_generation_df)}.")
        
        mutated_df = mutate_generation(new_generation_df, mutation_fraction/crossover_fraction, bbdict, type_of_cyclization)
        print(f" Total rows after mutation: {len(mutated_df)}.")

        # Append long-living individuals from the old generation
        prefinal_generation = append_long_livers(population, mutated_df, long_liver_fraction=long_liver_fraction)
        final_generation = expand_2aa_seq(prefinal_generation, bbdict, type_of_cyclization)
        print(f"Final generation size after appending long-livers: {len(final_generation)}.")
        
        
        # Calculate metrics to measure success
        mean_permeability, max_permeability, permeability_percentage = calculate_df_metrics(final_generation)
        mean_permeabilities.append(mean_permeability)
        max_permeabilities.append(max_permeability)
        permeability_percentages.append(permeability_percentage)
        print(f"Epoch {epoch+1}: Mean Permeability = {mean_permeability}, Max Permeability = {max_permeability}, Permeability Percentage = {permeability_percentage}")

        # Check for convergence
        current_metrics = (mean_permeability, max_permeability, permeability_percentage)
        if previous_metrics is not None:
            metrics_change = max(
                abs((current_metrics[i] - previous_metrics[i]) / previous_metrics[i] if previous_metrics[i] != 0 else 0)
                for i in range(len(current_metrics))
            ) * 100  # Convert to percentage
            print(f"Metrics change percentage: {metrics_change}%")

        if metrics_change <= convergence_percentage:
            print(f"Convergence reached at epoch {epoch+1} with percentage change {metrics_change}%. Stopping...")
            break
        
        # Update population and metrics for the next generation
        population = final_generation
        previous_metrics = current_metrics
        
    # Plotting the results if enabled
    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        axes[0].plot(mean_permeabilities, label='Mean Permeability', color='blue')
        axes[0].set_title('Mean Permeability Across Epochs')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Mean Permeability')
        axes[0].grid(True)

        axes[1].plot(max_permeabilities, label='Max Permeability', color='red')
        axes[1].set_title('Max Permeability Across Epochs')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Max Permeability')
        axes[1].grid(True)

        axes[2].plot(permeability_percentages, label='Permeability Percentage', color='green')
        axes[2].set_title('Permeability Percentage Across Epochs')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Percentage')
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

    return population