import cv2
import os

from gpg import Data

from lib import Data_visualisation, Model

image = 0 # compteur d'image
capture_fodler = 'image_capture' # non du dossier destination

#création du dosier contenant les images
try:
    if not os.path.exists(capture_fodler):
        os.makedirs(capture_fodler)
except OSError:
    print("ERROR")
    
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect(gray, frame, ret, model):
    global image
    clear_image = frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        if ret:
            image_name = "./" + capture_fodler + '/image_' + str(image) + ".jpg" # non de l'image
            cv2.imwrite(image_name, clear_image) # on capture l'image
            Data_visualisation.predictImage(image_name, model)
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

def detect_from_video(model):
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canvas = detect(gray, frame, ret, model)
        cv2.imshow('Video', canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)