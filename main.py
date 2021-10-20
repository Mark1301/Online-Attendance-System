import cv2
import numpy as np
import face_recognition

imgUtsav = face_recognition.load_image_file('ImagesBasic/utsav.jpg')
imgUtsav = cv2.cvtColor(imgUtsav,cv2.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file('ImagesBasic/Nandini.jpg')
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgUtsav)[0]
encodeUtsav = face_recognition.face_encodings(imgUtsav)[0]
cv2.rectangle(imgUtsav,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgtest)[0]
encodetest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeUtsav],encodetest)
faceDis = face_recognition.face_distance([encodeUtsav],encodetest)
print(results,faceDis)
cv2.putText(imgtest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)


cv2.imshow('Utsav Vadhar',imgUtsav)
cv2.imshow('Utsav Test',imgtest)
cv2.waitKey(0)
