import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

#load and preprocess image
def load_image_from_folder(folder,image_size=(64,64)):
    images=[]
    filenames=[]
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png','.jpg','.jpeg')):
            img_path=os.path.join(folder,filename)
            img=cv2.imread(img_path)
            if img is not None:
                img=cv2.resize(img,image_size)
                images.append(img)
                filenames.append(img_path)
    return np.array(images),filenames

#path to your folder of 1000 images
folder_path="C:\\Users\\DELL 0424\\OneDrive\\Desktop\\k-means"
images,image_paths=load_image_from_folder(folder_path)
print(f"loaded{len(images)}images")

#flatten the image
X=images.reshape(len(images),-1)
#(num_images,width*height*3)

#apply k-means clustering
k=5 #you can change the number of culsters
kmeans=KMeans(n_clusters=k,random_state=42)
labels=kmeans.fit_predict(X)
print(labels)

#get cluster centers
centers=kmeans.cluster_centers_
#scattter plot of points,color by cluster
plt.scatter([x[0] for x in X],[x[1] for x in X],c=labels,cmap='viridis',marker='o')

#plot centroid
plt.scatter(centers[:,0],centers[:,1],s=200,marker='x')
plt.title("simple k-means Example")
plt.xlabel("X-axis")
plt.ylabel("y-axis")
plt.show()


#visualize sample image
def show_cluster_example(images,labels,k,sample_per_cluster=5):
    
    plt.figure(figsize=(sample_per_cluster*2,k*2))
    for cluster in range(k):
        cluster_indices=np.where(labels == cluster)[0][:sample_per_cluster]
        for i,idx in enumerate(cluster_indices):
            plt.subplot(k,sample_per_cluster,cluster*sample_per_cluster+i+1)
            plt.imshow(cv2.cvtColor(images[idx],cv2.COLOR_BGR2RGB))
            plt.title(f"cluster{cluster}")
            plt.tight_layout()
            plt.show()
show_cluster_example(images,labels,k)