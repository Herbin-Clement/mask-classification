# Face recognition

# Importing the libraries
import cv2
import os

from lib import Data_visualisation, Model

image = 0 # compteur d'image
capture_fodler = 'image_capture' # non du dossier destination

#création du dosier contenant les images
try:
    if not os.path.exists(capture_fodler):
        os.makedirs(capture_fodler)
except OSError:
    print("ERROR")
    


# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Defining a function that will do the detections   
def detect(gray, frame, ret):
    global image
    clear_image = frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        if ret:
            image_name = "./" + capture_fodler + '/image_' + str(image) + ".jpg" # non de l'image
            cv2.imwrite(image_name, clear_image) # on capture l'image

            #passade dans le model

            image+=1

        else:
            break
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # capture de l'image 
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+h]
        eye = eye_cascade.detectMultiScale(roi_gray, 1.1, 2)
        for (ex, ey, ew, eh) in eye:
            cv2.rectangle(roi_color, (ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2)
    return frame

# Doing some face recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame, ret)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
cv2.waitKey(1)