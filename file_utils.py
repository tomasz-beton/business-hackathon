import pandas as pd


def load_dataframes(dataset_dir: str):
    items = pd.read_csv(f'{dataset_dir}/items.csv', na_values='unknown',
                        dtype={'item_id': 'Int64', 'alley_id': 'Int64', 'category': 'Int64'},
                        index_col=['item_id', 'alley_id', 'category'])
    items.drop(columns='Unnamed: 0', inplace=True)

    categories = pd.read_csv('data/categories.csv', index_col='category_id')
    categories.drop(columns='Unnamed: 0', inplace=True)

    alley_inventory = pd.read_csv('data/alley_inventory.csv', index_col='alley_id')

    transaction_items__prior = pd.read_csv('data/transaction_items__prior.csv', na_values='unknown',
                                           dtype={'transaction_id': 'Int64', 'item_id': 'Int64',
                                                  'add_to_cart_order': 'Int64', 'previous_bought': 'Int64'})

    transaction_items__prior.drop(columns='Unnamed: 0', inplace=True)
    transaction_items__prior.dropna(subset=['transaction_id'], inplace=True)
    transaction_items__prior.set_index(['transaction_id', 'item_id'], inplace=True)

    transaction_items__train = pd.read_csv('data/transaction_items__train.csv', na_values='unknown',
                                           dtype={'transaction_id': 'Int64', 'item_id': 'Int64',
                                                  'add_to_cart_order': 'Int64', 'previous_bought': 'Int64'})

    transaction_items__train.drop(columns='Unnamed: 0', inplace=True)
    transaction_items__train.dropna(subset=['transaction_id'], inplace=True)
    transaction_items__train.set_index(['transaction_id', 'item_id'], inplace=True)

    transactions = pd.read_csv('data/transactions.csv', na_values='unknown',
                               dtype={'transaction_id': 'Int64', 'customer_id': 'Int64', 'transaction_number': 'Int64',
                                      'day_of_week': 'Int64', 'time_of_day': 'Int64',
                                      'days_since_prior_order': 'Int64'})

    transactions.drop(columns='Unnamed: 0', inplace=True)
    transactions.dropna(subset=['transaction_id'], inplace=True)
    transactions.set_index(['transaction_id', 'customer_id'], inplace=True)

    dataframes = [items, categories, alley_inventory, transaction_items__prior, transaction_items__train, transactions]

    return tuple(dataframes)
