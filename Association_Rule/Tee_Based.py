import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import tree 
import matplotlib.pyplot as plt

#load dataset
df=pd.read_csv('teebased.csv')
df['Items']=df['Items'].apply(lambda x:x.split(','))

#one-hot encode the items(transaction->features)
mlb=MultiLabelBinarizer()
X=mlb.fit_transform(df['Items'])
items_columns=mlb.classes_

#let's pretend we're predicting if milk is bought or not (binary classification)
y=X[:,list(items_columns).index('milk')]

#remove milk from features set(to avoid data leakage)
X=pd.DataFrame(X,columns=items_columns)
X=X.drop(columns='milk')

#train/test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

#Train Decision Tree
clf=DecisionTreeClassifier(criterion='entropy',max_depth=3)
clf.fit(X_train,y_train)

#visualize the decision tree 
plt.figure(figsize=(12,8))
tree.plot_tree(clf,feature_names=X.columns,class_names=['No,Yes'],filled=True)
plt.show()

#predict  and  Evaluate
print("test accuracy:",clf.score(X_test,y_test))
