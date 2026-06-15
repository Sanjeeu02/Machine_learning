import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

#load data
df=pd.read_csv("students_pass_data.csv")

#data exploration
print(df.head())
print(df.info())
print("\n summary statistic\n")
print(df.describe())
print("\n class distribution\n")
print(df['pass'].value_counts())

#EDA-visualizations
sns.pairplot(df,hue="pass")
plt.suptitle("pass vs fail trends",y=1.02)
plt.show()

#correlation heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(),annot=True,cmap="coolwarm",fmt=".2f")
plt.title("Feature correlation")
plt.show()

#model preparation
X=df[['hours_studied','chapters_completed','mock_tests_attended']]
y=df['pass']

#split data 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#train logistic regression
model=LogisticRegression()
model.fit(X_train,y_train)

#predict
y_pred=model.predict(X_test)

#model Evalution
print("\n Accuracy:",accuracy_score(y_test,y_pred))
print("\n confusion matrix:\n",confusion_matrix(y_test,y_pred))
print("\n classification report/n",classification_report(y_test,y_pred))

#custom input prediction
hours=float(input("enter the studied:"))
chapters=int(input("enter  chapters completed:"))
mocks=int(input("enter the mock tests attended:"))
input_data=np.array([[hours,chapters,mocks]])
prediction=model.predict(input_data)[0]
probability=model.predict_proba(input_data)[0][1]
print(f"\n probability of passing:{probability:.2f}")
print("Fail" if prediction == 0  else "Pass")
