import numpy as np

import cv2

import os

############### K-NN Code ###################################################################

# K - Nearest Neighbour

def k_nearest_neighbour(train,test,k=5):
    
    
    
    dist = []
        # No. of images in the Training data and x.shape[1] denotes the corresponding labels
    
    for i in range(train.shape[0]):
        # get the vector and label
        
       
        
        ix = train[i,:-1]
        iy = train[i,-1]
        
        # Compute the distance from the test point
        
        d = distance(test,ix)
        dist.append([d,iy])
        
    
        
        #sort based upon distance and get top of K
    
    
    
    dk = sorted(dist,key=lambda x:x[0])
    
        
    
    dk=dk[:k]
        
        #Retreive only the labels
    labels = (np.array(dk))[:,-1]
        #labels = np.array(dk)[:,-1]
    
        
        
        
        
        
        #Get frequencies of each label
        
    output = np.unique(labels,return_counts=True)
        
        
        #Find Max. frequency and corresponding label
        
    index = output[1].argmax()
        
        
        
    return output[0][index]
        
        
        
def distance(x1,x2):
        return np.sqrt(sum((x1-x2)**2))
    
    
##############################################################################################


cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
skip = 0
face_data = []
label =[]

data_path = './data/'

class_id = 0 #Id for given file

names ={} #Mapping between names and id

#Data Preparation

for fx in os.listdir(data_path):
    
    
    if fx.endswith('.npy'):
        print("Loaded "+fx)
        ## Create a mapping between class_id and the name
        names[class_id] = fx[:-4]
        
        data_item = np.load(data_path+fx)
        face_data.append(data_item)
        
        
        
        #Create labels for the class
        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1
        print("Class_id = ",class_id)
        label.append(target)
 


        
face_dataset =np.concatenate(face_data,axis=0)
face_labels = np.concatenate(label,axis=0).reshape((-1,1))         #It changes the face_labels matrix into a column





training_dataset = np.concatenate((face_dataset,face_labels),axis=1)   # Concatenating the label as part of the column                                                                          in the joint matrix of x any y

############################### Testing ######################################################

while True:
    
    ret,frame = cap.read()
    
    if ret == False:
        continue
        
        
    faces = face_cascade.detectMultiScale(frame,1.3,5)
    
    for face in faces:
        
        x,y,w,h = face
        
        # get the face ROI
        
        offset = 10
        
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        
        ## Predicted label(out)
        
        out = k_nearest_neighbour(training_dataset,face_section.flatten())
        
        
        
        # display on the screen the name and rectangle around it
        
        pred_name = names[int(out)]
       
        
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        # 1. Image
        # 2. Text_data that you want to write
        # 3. Coordinate where you want the text
        # 4. Type of Font
        # 5. Font scale
        # 6. Color
        # 7. Thickness
        # for better_look line type is "Line_AA"
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        
  
    cv2.imshow("Faces",frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
