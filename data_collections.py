# Write a python Script that captures images from your webcam video stream
# Extracts all faces from the images frame (using haarcascades)

# Stores tha Face imformation into numpy arrays
#  Detect Faces and show bounding box
#  Flatten the largest face images and save in a numpy array
# Repeat the largest face impage and save in a numpy array
# Repeat the above for multiple people to genrate training data

## working with video stream
import cv2
import numpy as np
# init camera
cap=cv2.VideoCapture(0)   # id =0 for default videostream  if we want to working on video or photo then give path like "my_video.mp4" 
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
skip=0;
# a numpy array to store the image
face_data=[]
dataset_path='./data/'
file_name=input("Enter the name of person")
while True:
    ret,Frame=cap.read()
    if ret==False:
        continue
    grey_frame=cv2.cvtColor(Frame,cv2.COLOR_BGR2GRAY)    
    faces=face_cascade.detectMultiScale(grey_frame,1.3,5)  # contain two more parameter scaling factor and no of neighbours
  # sorting the faces
    faces=sorted(faces,key=lambda f:f[2]*f[3])   # sort on the basis of largest image




  # cv2.imshow("video frame",Frame)
  # cv2.imshow("Gray frame",gray_frame)
    
    # taking the largest face on the basis of area(f[2]*f[3])
    for face in faces[-1:]:
        x,y,w,h=face
        cv2.rectangle(Frame,(x,y),(x+w,y+h),(0,255,255),2)  # this return the quardinate of image and color of boundary 
        # extract (crop out the required face ): Region of interest
        offest=10 
        face_section=Frame[y-offest:y+h+offest,x-offest:x+w+offest]  # padding the image from all side by offest size
        # resize the face
        face_section=np.resize(face_section,(100,100))
        # store every 10th faces
        skip+=1;
        if(skip%10==0):
            face_data.append(face_section)
            print(len(face_data))
        cv2.imshow("face section",face_section)
        
        
    cv2.imshow("video frame",Frame) 
   # cv2.imshow("face section",face_section)
  
   # cv2.imshow("Gray frame",grey_frame)
    keyPressed=cv2.waitKey(1)& 0xFF
    if keyPressed==ord('q'):
        break
# convert face list as numpy array
face_data=np.asarray(face_data)
face_data=np.reshape(face_data,(face_data.shape[0],-1))
# save the data
print(face_data.shape)
np.save(dataset_path+file_name +'.npy',face_data)

print("data successfully save",dataset_path+file_name+'.npy')
cap.release()
cv2.destroyAllWindows()



""""Scale factor : parameter specifie that how many image reduce at each image scale.
Basically scale factor specifie two scale your pyramid
minNeighbour :
             specifi that how many neighbour each condidate rectangle should have .
             this parameter affect the quality of image .Higher the value result in less detection but with higher quality.
             3~5 is good value for it .
"""
