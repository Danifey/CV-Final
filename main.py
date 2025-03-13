
import cv2  


# python -m venv venv
# venv/Scripts/activate.bat
# pip install opencv-python
# python main.py

# https://github.com/opencv/opencv/tree/master/data/haarcascades

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
#smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# capture frames from a camera 
cap = cv2.VideoCapture(0)
  
# loop runs if capturing has been initialized. 
while True:  
  
    # reads frames from a camera 
    ret, img = cap.read()  
  
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
    # Detects faces of different sizes in the input image 
    faces = face_cascade.detectMultiScale(gray, 1.3, 4) 
  
    for (x,y,w,h) in faces: 
        # To draw a rectangle in a face  
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)  
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w] 

        # Detects Eyes
        eyes = face_cascade.detectMultiScale(roi_gray)  
    
        #To draw a rectangle in eyes 
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2) 
            
        # Detects Eyes
        smiles = face_cascade.detectMultiScale(roi_gray)  
    
        #To draw a rectangle in eyes 
        for (ex,ey,ew,eh) in smiles: 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,255),2) 

    # Display an image in a window 
    cv2.imshow('img',img) 
  
    # Stop if Escape or Exit button is pressed
    if cv2.waitKey(30) == 27 or cv2.getWindowProperty('img', 0) == -1: 
        break
  
# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()  
