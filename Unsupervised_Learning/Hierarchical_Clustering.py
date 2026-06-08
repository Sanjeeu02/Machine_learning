import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.cluster import AgglomerativeClustering

#Load dataset
data=pd.read_csv('Unsupervised_Learning/hierarchy.csv')

#extract features
features=data[['PatientID','Age','BloodPressure','Cholesterol','HeartRate']]

#plot dendrogram
linked=linkage(features,method='ward')

plt.figure(figsize=(10,6))
dendrogram(linked,labels=data['patientID'].values,orientation='top',distance_sort='descending',show_leaf_counts=True)
plt.title('Hierarchy clustering dendrogram')
plt.xlabel('patients')
plt.ylabel('distance')
plt.show()

#apply Agglomerative clustering
cluster=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
data['cluster']=cluster.fit_predict(features)

#visualize cluster in 2d
sns.scatterplot(data=data,x='Age',y='Cholestrol',hue='Cluster',palette='Set1',s=100)
plt.title('patient cluster based on age & cholestrol')
plt.xlabel('Age')
plt.ylabel('Cholestrol')
plt.grid(True)
plt.show()