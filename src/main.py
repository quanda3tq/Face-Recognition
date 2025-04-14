import cv2
import os
import pickle
import face_recognition
import numpy as np
import cvzone
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

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread('resources/background.png')

# Importing the mode images into a list                        
FolderModePath = 'resources/Modes'
ModePathList = os.listdir(FolderModePath)
imgModeList = []

for path in ModePathList:
    imgModeList.append(cv2.imread(os.path.join(FolderModePath, path)))
    
# print(len(ModePathList))

# Load the encoding file
print("Loading encode file ...")
file = open('EncodeFile.p','rb')
encodeListKnownWithId = pickle.load(file)
file.close()
encodeListKnown, employeeId = encodeListKnownWithId
# print(employeeId)
print("encode file loaded")


modeType = 0
counter = 0
id = 0


while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS , cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print("matches", matches)
        # print("faceDis", faceDis)

        matchIndex = np.argmin(faceDis)
        # print("Match Index", matchIndex)

        if matches[matchIndex]:
            # print("know face detected")
            # print(employeeId[matchIndex])
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 *4, y2 *4, x1 *4
            bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)

            id = employeeId[matchIndex]

            if counter == 0:
                counter = 1
                modeType = 1
    
    if counter != 0:
        if counter == 1:
            employeeInfo = db.reference(f'Employees/{id}').get()
            print(employeeInfo)

        cv2.putText(imgBackground,str(employeeInfo['tatol_attendance']), (861,125),
                    cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
        cv2.putText(imgBackground,str(employeeInfo['name']), (808,455),
                    cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
        cv2.putText(imgBackground,str(employeeInfo['dapartment']), (1006,550),
                    cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
        cv2.putText(imgBackground, str(id),(1006, 493),
                    cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
        
        counter += 1


    # cv2.imshow("Webcam", img)
    cv2.imshow("Face attendance", imgBackground)
    cv2.waitKey(1)


