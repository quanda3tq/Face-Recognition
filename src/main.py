import cv2
import os
import pickle
import face_recognition
import numpy as np
import cvzone


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


while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS , cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[0]

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


    # cv2.imshow("Webcam", img)
    cv2.imshow("Face attendance", imgBackground)
    cv2.waitKey(1)


