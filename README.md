# Hackathon Business 2.0

## Re-order based recommendations
We propose a recommender model that for each client with a history of purchases, we recommend a list of products that the client is likely to buy in the next purchase.

### Features
We can engineer features in separate categories:
- Client features
- Product features
- Client-Product features
- Transaction features

For client, product and client-product features, we can use the information from the previous transactions - `transaction_items__prior.csv`.
For transaction features, we can use the information from the last transaction - `transaction_items__train.csv`.

### Prediction
We can use a classification model to predict whether a client will buy a product in the next transaction (`transaction_items__train.csv`)

### Model
We can use a gradient boosting model.

### Evaluation
We can use the F1 score to evaluate the model. For threshold independent evaluation, we can use the ROC-AUC score.

Metrics achieved on the validation set (20% of the training set):
- F1 score: 0.37
- ROC-AUC score: 0.82

### Drawbacks
This method does not cover the case of new clients or new products.
Client and product features should be re-computed periodically.

## Association rules mining model
Another model that we can use is association rules mining using Apriori algorithm. We can use the information from the previous transactions - `transaction_items__prior.csv`. To classify groups of products frequently bought together.

### Model
Apriori algorithm is a classical algorithm for frequent itemset mining and association rule learning over transactional databases. It proceeds by identifying the frequent individual items in the database and extending them to larger and larger item sets as long as those item sets appear sufficiently often in the database. The frequent itemsets determined by Apriori can be used to determine association rules which highlight general trends in the database.

### Evaluation
Association rules have been found for a total of 100 products belonging to the top 10 most selling categories