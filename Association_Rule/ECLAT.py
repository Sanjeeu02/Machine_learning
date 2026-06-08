import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth #ECLAT is similar in concept
from itertools import combinations

#load csv
df=pd.read_csv('Association_Rule/ECLAT.csv')

#convert string items into list of items
transactions=df['Items'].apply(lambda x:x.split(','))

#transaction Encoder
te=TransactionEncoder()
te_ary=te.fit(transactions).transform(transactions)
df_encoded=pd.DataFrame(te_ary,columns=te.columns_)

#convert Dataframe to vertical format for ECLAT(items->transaction ID list)
vertical={}
for items in df_encoded.columns:
    vertical[items]=set(df_encoded.index[df_encoded[items]==True])

#ECLAT function
def eclat(prefix,items,min_support):
    frequent_itemsets=[]
    while items:
        i,itids=items.pop()
        support=len(itids)
        if support>=min_support:
            frequent_itemsets.append((prefix + [i],support))
            suffix=[]
            for j,ojtids in items:
                intersection=itids & ojtids
                if len(intersection)>=min_support:
                    suffix.append((j,intersection))
                    frequent_itemsets.extend(eclat(prefix+[i],min_support))
    return frequent_itemsets

#convert vertical format to list for processing
items=list(vertical.items)

#set minimum support
min_support=2

#Run ECLAT
frequent_itemsets=eclat([],items,min_support)

#print results
print("Frequent Itemsets(min support={}):\n".format(min_support))
for itemset,support in frequent_itemsets:
    print(f"{itemset}=>support:{support}")
    
