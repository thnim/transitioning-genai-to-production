import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv('repository_dataset - README_category.csv')
df_bool = df[['What', 'How', 'Who', 'Reference', 'Contribution']] == 'Yes'

frequent_itemsets = apriori(df_bool, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])