import cv2
import os

from gpg import Data

from lib import Data_visualisation
    
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect(gray, frame, model, width, height):
    global image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        x, y, w, h = update_edge_face(x, y, w, h, width, height, 100, 100)
        roi_color = frame[y:y+h, x:x+h]
        pred = Data_visualisation.predict_image(roi_color, model, verbose=True)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = ["1", "2", "3", "4"]
        cv2.putText(frame, text[pred-1], (x, y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        image+=1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return frame

def detect_from_video(model):
    vcap = cv2.VideoCapture(0)
    
    width  = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    while True:
        _, frame = vcap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canvas = detect(gray, frame, model, width, height)
        cv2.imshow('Video', canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vcap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def update_edge_face(x, y, w, h, width, height, shiftX, shiftY):
    x = x - shiftX // 2
    y = y - shiftY // 2
    w = w + shiftX
    h = h + shiftY
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + w > width:
        w = width - x - 1
    if y + h > height:
        h = height - y + 1
    return int(x), int(y), int(w), int(h)