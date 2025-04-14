import cv2
import face_recognition
import pickle
import os
import firebase_admin
from dotenv import load_dotenv
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage


load_dotenv()

cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS_PATH"))
firebase_admin.initialize_app(cred, {
    'databaseURL': os.getenv("FIREBASE_DATABASE_URL"),
    'storageBucket': os.getenv("STORAGE_BUCKET")
})


# Importing the face images                        
FolderPath = 'images'
PathList = os.listdir(FolderPath)
print(PathList)
imgList = []
employeeId = []

for path in PathList:
    imgList.append(cv2.imread(os.path.join(FolderPath, path)))
    employeeId.append(os.path.splitext(path)[0])

    fileName = os.path.join(FolderPath, path)
    bucket = storage.bucket()

    # print(path)
    # print(os.path.splitext(path)[0])

print(employeeId)

def FindEnCodings (imagesList):
    encodelist = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)

    return encodelist

print("Encoding started ...")
encodeListKnown = FindEnCodings(imgList)
encodeListKnownWithId = [encodeListKnown, employeeId]
print("Encoding complete")

file = open("EncodeFile.p",'wb')
pickle.dump(encodeListKnownWithId,file)
file.close()
print("File saved")