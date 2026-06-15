import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score

#configuration
IMAGE_SIZE=(100,100)#resize all image to 100*100
DATA_PATH="C:\\Users\\DELL 0424\\OneDrive\\Desktop\\ML\\train"#path to train folder

#load dataset
def load_image():
    X=[]
    y=[]
    class_labels=os.listdir(DATA_PATH)
    for label in class_labels:
        folder=os.path.join(DATA_PATH,label)
        for file in os.listdir(folder):
            file_path=os.path.join(folder,file)
            img=cv2.imread(file_path)
            if img is not None:
                img=cv2.resize(img,IMAGE_SIZE)
                img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                X.append(img.flatten())
                y.append(label)
    return np.array(X),np.array(y)
print("loading dataset")
X,y=load_image()
print("dataset loaded:",X.shape)

#pca for dimensionality reduction
pca=PCA(n_components=2)#2 components for svm graph
X_pca=pca.fit_transform(X)

#train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_pca,y,test_size=0.2,random_state=42)

#svm model
svm=SVC(kernel='rbf',C=10,gamma='scale')#rbf kernal with high c for high accuracy
svm.fit(X_train,y_train)

#evaluation
y_pred=svm.predict(X_test)
print("\n classification report",classification_report(y_test,y_pred))
print("\n Accuracy:",accuracy_score(y_test,y_pred))

#predict user uoloaded image
def predict_user_image(image_path):
    img=cv2.imread(image_path)
    img=cv2.resize(img,IMAGE_SIZE)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_flat=img.flatten().reshape(1,-1)
    img_pca=pca.transform(img_flat)
    prediction=svm.predict(img_pca)
    print("prediction for upload image:",prediction[0])
predict_user_image("C:\\Users\\DELL 0424\\OneDrive\\Desktop\\ML\\train\\daisy\\11642632_1e7627a2cc_jpg.rf.8794534201606e49eee701066c2c5c82.jpg")
    