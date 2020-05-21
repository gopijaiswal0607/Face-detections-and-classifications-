# Recognise faces using some classification algorithm
#like logistic ,KNN,SVM etc
# 1. load the training data (numpy array of all the persons)
       # x- values are stored in the numpy arrays
       # y-values we need to assign for each person
# 2. Read a video stream using opencv
# 3. extract faces out of it
# 4. use knn to find the prediction of face (int)
# 5. map te predicted id to name of the user
# Display the prediction on the screen bounding box and name
import numpy as np
import cv2
import os


#########  KNN  CODE  ###########

def distance(x1,x2):
    #Eucledian
    return np.sqrt(((x1-x2)**2).sum())
def knn(train,test,k=5):
    dist=[]
    m=train.shape[0]
    for i in range(m):
        # Get the vector and label
        ix=train[i,:-1]
        iy=train[i,-1];
        # computer the distance from test point
        d=distance(test,ix)
        dist.append([d,iy])
    # sort based on distance and get top k    
    dk=sorted(dist,key=lambda x:x[0])[:k]    
    # Retrive only the labels
    labels=np.array(dk)[:,-1]
    # get frequencies of each label
    output=np.unique(labels,return_counts=True)
    # Find max frequency and clrrespongin label
    index=np.argmax(output[1])
    return output[0][index]
##############################################
## working with video stream
# init camera
#cap=cv2.VideoCapture(0)   # id =0 for default videostream  if we want to working on video or photo then give path like "my_video.mp4" 
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
skip=0;
# a numpy array to store the image
face_data=[]
dataset_path='./data/'

labels=[]

class_id=0 # Labels for the given file
name={}  # Mapping btw id  -name


# Data Preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        # Create a mapping btw thw class id and name
        name[class_id]=fx[:-4]
        print("loaded "+fx)
        data_item=np.load(dataset_path+fx)
        face_data.append(data_item)
        
        
        # Create Labels for the class
        target=class_id*np.ones((data_item.shape[0],))
        class_id+=1
        labels.append(target)
#print(np.array(face_data).shape)
#print(np.array(labels).shape)      
face_dataset=np.concatenate(face_data,axis=0)#reshape((-1,1))
face_labels=np.concatenate(labels,axis=0).reshape((-1,1))


# we can see the shape of data item in data folder of numpy type
print(face_dataset.shape)
print(face_labels.shape)
trainset=np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)

# Testing
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if ret==False:
        continue
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    for face in faces:
        x,y,w,h=face
        
        # Get the face 
        
        offest=10
        face_section=frame[y-offest:y+h+offest,x-offest:x+w+offest]  # padding the image from all side by offest size
        # resize the face
        face_section=np.resize(face_section,(100,100))
        
        
        #Predicted Label(out)
        out=knn(trainset,face_section.flatten())
        #Display on the screen the name and rectangle around it
        pred_name=name[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
     
    cv2.imshow("Faces",frame)
    key=cv2.waitKey(1)& 0xFF
    if key==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
        
