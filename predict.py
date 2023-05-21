import pickle
import argparse
import pandas as pd

def load_transactions(path):
    transactions = pd.read_csv(path, na_values='unknown',
                               dtype={'transaction_id': 'Int64', 'customer_id': 'Int64', 'transaction_number': 'Int64',
                                      'day_of_week': 'Int64', 'time_of_day': 'Int64',
                                      'days_since_prior_order': 'Int64'})

    transactions.drop(columns='Unnamed: 0', inplace=True)
    transactions.dropna(subset=['transaction_id'], inplace=True)
    transactions.set_index('transaction_id', inplace=True)

    return transactions


def prepare_data(transactions):
    df = client_item_features.reset_index(1).join(transactions.reset_index().set_index('customer_id'), how='inner')
    df = df.join(client_features, how='inner', rsuffix='__client')
    df = df.join(item_features, how='inner', on='item_id', rsuffix='__item')
    df = df.set_index('item_id', append=True)
    df = df.drop(columns=["transaction_number", "eval_set", "days_since_prior_order", "transaction_id"])
    df = df.rename(
        columns={
            'n_transactions': 'n_transactions__client_item',
            'n_transactions__client': 'n_transactions',
            "reorder_rate": "reorder_rate__client_item",
            "reorder_rate__item": "reorder_rate",
        }
    )

    return df

def predict_and_order(df, model, threshold=0.5, limit=5):
    df["prediction"] = model.predict_proba(df)[:, 1]

    df = df.sort_values(by=['customer_id', 'prediction'], ascending=[True, False])
    df = df.loc[df.prediction > threshold]
    preds_df = df.reset_index(1).groupby('customer_id').agg(
        {
            'item_id': lambda x: list(x)[:limit],
         'prediction': lambda x: list(x)[:limit]
         }
    )

    return preds_df


def load_model(model_path='model.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--file', type=str, required=True)
    argparser.add_argument('--out', type=str, default='predictions.csv')
    argparser.add_argument('--model_path', type=str, default='model.pkl')
    args = argparser.parse_args()

    model = load_model()

    item_features = pd.read_parquet("features/item_features.parquet")
    client_features = pd.read_parquet("features/client_features.parquet")
    client_item_features = pd.read_parquet("features/client_item_features.parquet")

    transactions = load_transactions(args.file)
    df = prepare_data(transactions)
    preds_df = predict_and_order(df, model)

    preds_df.to_csv(args.out)

    print("Predictions:")
    print(preds_df)

    print(f"Predictions saved to {args.out}")
