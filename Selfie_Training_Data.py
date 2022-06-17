import cv2

import numpy as np

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
skip = 0
face_data = []
data_path = './data/'

file_name = input("Enter the name of the person whose face we are scanning: ")

while True:
    ret,frame = cap.read()
    
    if ret == False:
        continue
        
        
    cv2.imshow("Frame",frame)
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(frame,1.3,5)
    faces = sorted(faces, key=lambda f:f[2]*f[3],reverse=True)  #Sorting the faces with w,h parameters
    
  
    
    
    for face in faces:
        x,y,w,h = face
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        
    
    
        # pick the section of face i.e Region Of Interest. increasing the boundary lines by 10 pixels
        skip+=1
        offset = 10
        x,y,w,h = face
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        
        # Store the image of every 10th face from the video streaming in a file    
        if skip%10 == 0:
            face_data.append(face_section)
            print(len(face_data))
        
        
    cv2.imshow("Frame",frame)
    cv2.imshow("Face_Section",face_section)
    
    
    
    
    
    key_pressed = cv2.waitKey(1) & 0xFF
    
    if key_pressed == ord('q'):
        break
        

#converting our face list into a numpy array

face_data = np.asarray(face_data)

face_data = face_data.reshape((face_data.shape[0],-1))     # No. of rows = No. of captured faces
print(face_data.shape)


#Save this data into file system

np.save(data_path+file_name+'.npy',face_data)
print("Data successfully saved at "+data_path+file_name+'.npy')


        
cap.release()
cv2.destroyAllWindows()
