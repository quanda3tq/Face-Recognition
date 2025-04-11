import cv2
import face_recognition
import pickle
import os

# Importing the face images                        
FolderPath = 'images'
PathList = os.listdir(FolderPath)
print(PathList)
imgList = []
employeeId = []

for path in PathList:
    imgList.append(cv2.imread(os.path.join(FolderPath, path)))
    employeeId.append(os.path.splitext(path)[0])
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