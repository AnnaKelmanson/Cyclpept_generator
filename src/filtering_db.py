import pandas as pd

def filter_for_non_polar_AA(df):
    return df.loc[df['Type of side chaind'] != 'Polar']

def price_filter(df, price_limit):
    return df.loc[df['Cheapest Price (EUR/g)'] <= price_limit]

