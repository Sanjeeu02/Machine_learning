import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
df=pd.read_csv('Supervised_Learning/rain.csv')
print("/n first 5 rows of dataset:/n",df.head())
print("/n Dataset info:/n")
print(df.info())
print("/n summary statistic:/n")
print(df.describe())
print("/n Missing value:/n",df.isnull().sum())
#Data Visualization(EDA)
df.hist(figsize=(10,8),bins=30,edgecolor='black')
plt.suptitle('Feature Distribution')
plt.tight_layout()
plt.show()

# Correlation matrix heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm',fmt=".2f")
plt.title("correlation Matrix")
plt.show()

#pairplot for visual inspection
sns.pairplot(df)
plt.suptitle("Feature Relationship",y=1.02) #y= avoid the overlap
plt.show()

#box plot to detect outliers
plt.figure(figsize=(12,6))
for i,col in enumerate(df.columns):
    plt.subplot(1,4,i+1)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of{col}')
plt.tight_layout()
plt.show()

#Machine Learning Phase
#Feature selection
X=df[['temperature','humidity','wind_speed']]
y=df['rainfall']

#Data splitting
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#model training
model=LinearRegression()
model.fit(X_train,y_train)

#Prediction
y_pred=model.predict(X_test)

#Evaluation Metrics
print("/n model coefficients:")
print(model.coef_)
print("model intercept:")
print(model.intercept_)
print("Evaluation metrics")
print("mean squared error(MSE):")
print(mean_squared_error(y_test,y_pred))
print("Root mean squared error(RMSE):")
print(np.sqrt(mean_squared_error(y_test,y_pred)))
print("Mean Absolute Error(MAE):")
print(mean_absolute_error(y_test,y_pred))
print("r2 Score\n",r2_score(y_test,y_pred))

# Actual vs Predict plot
plt.figure(figsize=(10,5))
plt.plot(range(len(y_test)),y_test.values,label='Actual',marker='o')
plt.plot(range(len(y_test)),y_pred,label='predicted',marker='X')
plt.xlabel("simple index")
plt.ylabel("Rainfall")
plt.title('actual vs Predicted Rainfall')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#User Prediction
print("/n enter the value to predict rainfall:")
temp=float(input("temperature(^c):"))
humidity=float(input("humidity(%)"))
wind_speed=float(input("wind speed(km/h):"))
user_input=np.array([[temp,humidity,wind_speed]])
predicted_rainfall=model.predict(user_input)
print(f"\n Predicted Rainfall:{predicted_rainfall[0]:.2f}mm")


