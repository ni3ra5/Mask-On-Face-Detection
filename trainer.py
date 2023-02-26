import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime
# Path is where face datasets are stored
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
# Below function to get pictures and labels 
def getImagesAndLabels(path):
    img_path = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in img_path:
        PIL_img = Image.open(imagePath).convert('L') # grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
print ("\n [INFO] Faces are being trained. This will take a while...")
start = datetime.now()
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
# Save the trained faces in trainer/trainer.yml
recognizer.write('trainer/trainer.yml') 
# Show the number of trained faces
end = datetime.now()
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
print("\n [INFO] Time Taken: ")
print(end-start)
