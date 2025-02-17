import cv2
cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('E:\\Conda Projects\\FD\\haarcascade_frontalface_default.xml')

Id= int(input('Enter your ID: '))
sampleNum=0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        #incrementing sample number 
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        cv2.imwrite("E:\\Conda Projects\\FD\\Dataset\\User."+str(Id) +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('frame',img)
    #wait for 100 miliseconds 
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is more than 99
    elif sampleNum>99:
        break
cam.release()
cv2.destroyAllWindows()