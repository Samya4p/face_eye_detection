      #face_eye_detection

import cv2
                                                         
    #LOAD THE CLASSIFIER FILES FROM YOUR DEVICE
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

           #START CAPTURING THE VIDEO
video_capture = cv2.VideoCapture(0)

while True:
     
        #FRAME VARIABLE WILL READ THE VIDEO
   ret, frame = video_capture.read()
    
    #CONVERT THE FRAME INTO GRAYSCALE
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   faces = faceCascade.detectMultiScale( gray, scaleFactor=1.1, minNeighbors=16, minSize=(30,30) )
    
    #POINTS IS A LIST CREATED FOR STORING THE DIAGONAL COORDINATEs
   points=[]
 
        #DIAGONAL COORDINATES OF FACE TO BE DETECTED
  
   for (x, y, w, h) in faces:
     cv2.rectangle(frame, (x, y), (x+w, y+h), (150, 255, 0), 1)                             
     points=[x,y,w,h]
       
        #FOR EYE RECOGNITION ON DETECTED FACES
   
   if len(points)==4:
    roi_frame=frame[y:y+h,x:x+w]  
    eye=eye_cascade.detectMultiScale(roi_frame,1.03,5,minSize=(30,30))
    for (ex,ey,ew,eh) in eye:
      cv2.rectangle(roi_frame,(ex,ey),(ex+ew,ey+eh),(255,150,0),1,cv2.LINE_AA)
      
    # SHOW VIDEO
   cv2.imshow('Video', frame)

   
   if cv2.waitKey(1) & 0xFF == ord('q'):
        break

     #RELEASE VIDEO
video_capture.release()
cv2.destroyAllWindows()
