import cv2
import numpy as np
from datetime import datetime

file1 = open("E:\\Conda Projects\\FD\\logs\\MyFile.txt","a") 


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('E:\\Conda Projects\FD\\trainer\\trainer.yml')
cascadePath = "E:\\Conda Projects\\FD\\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, im = cam.read()
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(conf<50):
            if(Id==1):
                Id="Anwesh"
                name = Id
        else:
            Id="Unknown"
            name = Id
        cv2.putText(im,str(Id), (x,y+h),font, 1,(255,0,0))
    cv2.imshow('im',im)
    
    now = datetime.now()
    current_time = str(now.strftime("%H:%M:%S"))
    file1.write(name+ " " + current_time + "\n")
    

    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
file1.close()