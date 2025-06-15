import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load the dataset as list of transactions using correct path
with open(r'C:/Users/LENOVO\Downloads/market basket analysis/groceries.csv.csv', 'r') as file:
    transactions = [line.strip().split(',') for line in file.readlines()]

# Convert list of transactions to one-hot encoded DataFramex
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
dataset = pd.DataFrame(te_ary, columns=te.columns_)


# Apply Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(dataset, min_support=0.05, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Save the results to CSV files
frequent_itemsets.to_csv('frequent_itemsets.csv', index=False)
rules.to_csv('association_rules.csv', index=False)

print("Frequent Itemsets and Association Rules have been saved as CSV files.")
