import pandas as pd
import os


def load_dataframes(dataset_dir: str):
    dataframes = []
    filenames = [
        'alley_inventory.csv',
        'categories.csv',
        'items.csv',
        'transaction_items__prior.csv',
        'transaction_items__train.csv',
        'transactions.csv'
    ]

    for filename in filenames:
        filepath = os.path.join(dataset_dir, filename)
        dataframe = pd.read_csv(filepath)
        dataframes.append(dataframe)

    return tuple(dataframes)
