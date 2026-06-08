import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

#load dataset
df=pd.read_csv('loan_approval_data.csv')

#encode categorical features
le_employment=LabelEncoder()
le_marital=LabelEncoder()
le_education=LabelEncoder()

df["EmploymentStatus"]=le_employment.fit_transform(df["EmploymentStatus"])
df["MaritalStatus"]=le_marital.fit_transform(df["MaritalStatus"])
df["Education"]=le_education.fit_transform(df["Education"])

#train the model
X=df.drop("Approved",axis=1)
y=df["Approved"]

#model train
model=DecisionTreeClassifier(max_depth=5)
model.fit(X,y)

#visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(model,feature_names=X.columns,class_names=["Not Approved","Approved"],filled=True)
plt.title("Decision tree for loan Approval")
plt.show()

#take user input
print("\n Enter the application details:")
age=int(input("enter the age(21-45):"))
income=int(input("enter the income:"))
loan_amount=int(input("enter the loan amount:"))
credit_score=int(input("credit score (300-850):"))
employment_status=input("employment status (employed/unemployed/self-employed):")
marital_status=input("marital status(single/married/divorced):")
education=input("education(high school/bachelor's/masters/phD):")

#Encode input
try:
    employment_encoded=le_employment.transform([employment_status])[0]
    marital_encoded=le_marital.transform([marital_status])[0]
    education_encoded=le_education.transform([education])[0]
except ValueError as e:
    print("Error: invalid category entered.")
    print("Details;",e)
    exit()

#create input dataframe
input_data=pd.DataFrame([{
    "Age":age,
    "Income":income,
    "LoanAmount":loan_amount,
    "CreditScore":credit_score,
    "EmploymentStatus":employment_encoded,
    "MaritalStatus":marital_encoded,
    "Education":education_encoded
}])

#predict
prediction=model.predict(input_data)[0]
print("\n Loan Approval Result")
print("loan approved" if prediction == 1 else "loan not approved")

