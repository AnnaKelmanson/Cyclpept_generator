
def generate_experimental_setup(GA_population, size_limit):
    global top_set
    df = GA_population.sort_values(by='Permeability', ascending=False)

    unique_strings = set()

    for sequence in df['Sequence']:
        for item in sequence:
            unique_strings.add(item)
            if len(unique_strings) >= size_limit:
                break
        if len(unique_strings) >= size_limit:
            break

    result_list = list(unique_strings)
    filtered_top_set = top_set[top_set['Name'].isin(result_list)]

    return filtered_top_set, result_list

def filter_sequences(df, target_strings):
    """
    Filters a DataFrame to return only rows where the 'Sequence' column contains
    the specified string(s).

    Parameters:
    df (pd.DataFrame): DataFrame to be filtered.
    target_strings (str or list): String or list of strings to look for in each 'Sequence'.

    Returns:
    pd.DataFrame: A DataFrame containing only the rows where 'Sequence' contains the target string(s).
    """
    if isinstance(target_strings, str):
        target_strings = [target_strings]  # Convert single string to list for uniform processing

    # Use apply to create a mask where we check if any of the target strings are in the 'Sequence' list
    mask = df['Sequence'].apply(lambda seq: any(item in seq for item in target_strings))
    return df[mask].sort_values(by='Permeability', ascending=False)

def filter_exclusive_sequences(df, target_strings):
    """
    Filters a DataFrame to return only rows where the 'Sequence' column lists
    exclusively contain strings that are present in the specified target strings.

    Parameters:
    df (pd.DataFrame): DataFrame to be filtered.
    target_strings (str or list): String or list of strings that each 'Sequence' list must exclusively contain.

    Returns:
    pd.DataFrame: A DataFrame containing only the rows where 'Sequence' lists exclusively contain the target string(s).
    """
    if isinstance(target_strings, str):
        target_strings = [target_strings]  # Convert single string to list for uniform processing

    # Use apply to create a mask where we check if all items in the 'Sequence' list are within the target strings
    mask = df['Sequence'].apply(lambda seq: all(item in target_strings for item in seq))

    # Filter the DataFrame based on the mask
    return df[mask].sort_values(by='Permeability', ascending=False)