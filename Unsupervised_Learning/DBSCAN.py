import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN 
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('Dbscan.csv')

X=df[['Age','BloodPressure','Cholesterol','Glucose']]

scaler=StandardScaler()
x_scaled=scaler.fit_transform(X)

dbscan=DBSCAN(eps=1.2,min_samples=2)
clusters=dbscan.fit_predict(x_scaled)

df['Cluster']=clusters
print(df)

sns.scatterplot(x='Age',y='Cholesterol',hue='Cluster',palette='Set1',data=df)
df.hist()
plt.title("DBSCAN Clustering on Medical Data")
plt.show()