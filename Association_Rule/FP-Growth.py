import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth,association_rules

#load the csv file
df=pd.read_csv('Association_Rule/FPgrowth.csv')

#convert the items columns into a list of lists
transactions=df['Items'].apply(lambda x:x.split(','))

#Apply one-hot encoding using TransactionEncoder
te=TransactionEncoder()
te_ary=te.fit(transactions).transform(transactions)
df_encoded=pd.DataFrame(te_ary,columns=te.columns_)#Converts the NumPy array into a readable DataFrame
                                                   #Sets the column names to item names from te.columns_

#Apply FP-Growth Algorithms
frequent_itemsets=fpgrowth(df_encoded,min_support=0.5,use_colnames=True)

#Generate Association Rules
rules=association_rules(frequent_itemsets,metric='confidence',min_threshold=0.6)

# Print Results
print("Frequent Itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules:")
print(rules)