import cv2
import numpy as np
import face_recognition
import os
import winsound
from datetime import datetime
from cv2 import VideoWriter
from cv2 import VideoWriter_fourcc
from cvzone import FPS

path = 'Images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = face_recognition.load_image_file(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


encodeListKnown = findEncodings(images)
print('Encoding Complete')


frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
BG = cv2.imread("Background.png")
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

now = datetime.now()
filename = now.strftime('%H:%M:%S')
video = VideoWriter(f'webcam.avi', VideoWriter_fourcc(*'MP42'), 25.0, (frameWidth, frameHeight))

fpsReader = FPS()

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        print(matches)
        matchIndex = np.argmin(faceDis)
        if faceDis[matchIndex] < 0.50:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.putText(BG, "DANGER !!!", (492, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
            winsound.Beep(2000, 1500)
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    fps, img = fpsReader.update(img)
    BG[207:207 + frameHeight, 567:567 + frameWidth] = img

    if success:
        cv2.imshow("Portal", BG)
        video.write(img)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
video.release()
