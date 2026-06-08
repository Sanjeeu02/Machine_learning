import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

#load dataset
data=pd.read_csv('spam.csv',encoding='latin-1')[['v1','v2']]
data.columns=['label','text']
data['label']=data['label'].map({'ham':0,'spam':1})

#vectorization
vectorizer=CountVectorizer()
X=vectorizer.fit_transform(data['text'])
y=data['label']

#split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#train model
model=MultinomialNB()
model.fit(X_train,y_train)

#test accuracy
print(f"Accuracy:{model.score(X_test,y_test)*100:.2f}%")

#predict on new text file
def predict_email_file(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        text=f.read()
    vect_text=vectorizer.transform([text])
    prediction=model.predict(vect_text)[0]
    print("\n Email content:\n",text)
    print("\n prediction:","SPAM" if prediction == 1 else "NOT SPAM")

#prediction your uploaded email
predict_email_file("sample_email.txt")
