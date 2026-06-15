import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_graphviz 
import joblib
import os

#load dataset
df=pd.read_csv("diseases_data_2000.csv")

#encode disease label
le=LabelEncoder()
df["Disease"]=le.fit_transform(df["Disease"])

#prepare feature and target
X=df.drop("Disease",axis=1)
y=df["Disease"]

#split Dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#train with random forest + adaboost
base_rf=RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42)
model=AdaBoostClassifier(estimator=base_rf,n_estimators=10,learning_rate=0.5,random_state=42)
model.fit(X_train,y_train)

#evaluation model
y_pred=model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print("classification report",classification_report(y_test,y_pred,target_names=le.classes_))

#save the model and label encoder
joblib.dump(model,"disease_predictor_rf_boost.pkl")
joblib.dump(le,"label_encoder.pkl")

#predict disease fron user input
#example user input(0/1)
user_input=[1,0,1,1,1,1,1,1,0]

#change this as needed
user_input_np=np.array(user_input).reshape(1,-1)

#load the saved model
model=joblib.load("disease_predictor_rf_boost.pkl")
le=joblib.load("label_encoder.pkl")
predicted_label=model.predict(user_input_np)
predicted_disease=le.inverse_transform(predicted_label)
print("predicted disease:",predicted_disease[0])

#train a separate random forest
viz_rf=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42)
viz_rf.fit(X_train,y_train)

#select one decision tree from the forest
estimator=viz_rf.estimators_[0]

#plot the selected tree using matplotlib
plt.figure(figsize=(20,10))
plot_tree(estimator,feature_names=X.columns,class_names=le.classes_,filled=True,rounded=True,max_depth=3,fontsize=10)
plt.title("visualization of one decision tree",fontsize=16)
plt.show()




