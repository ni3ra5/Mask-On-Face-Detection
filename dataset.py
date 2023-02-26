import cv2
import os
from datetime import datetime
cam = cv2.VideoCapture(0)
cam.set(3, 640) #video width
cam.set(4, 480) #video height
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input('\n enter input id ==>  ')
print("\n [INFO] Recording dataset...")

start = datetime.now()
# starting face count with 'i'
i = 0
while(True):
    ret, img = cam.read()
    img = cv2.flip(img, 1, -1)
    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_detector.detectMultiScale(bw, 1.3, 5)
    for (x,y,w,h) in face:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        i += 1
        # save the image in dataset folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' +  
                    str(i) + ".jpg", bw[y:y+h,x:x+w])
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff # To exit press 'ESC'
    if k == 27:
        break
    elif i >= 60: # Take 60 face sample and stop video
        end = datetime.now()
        print("\n [INFO] Time Taken: ")
        print(end-start)
        break
# cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()