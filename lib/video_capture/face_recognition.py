import cv2
import os

from lib import Data_visualisation

image = 0 # compteur d'image
capture_fodler = 'image_capture' # non du dossier destination
    
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect(gray, frame, model, width, height):
    global image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        x, y, w, h = update_edge_face(x, y, w, h, width, height, 100, 100)
        roi_color = frame[y:y+h, x:x+h]
        image_name = "./" + capture_fodler + '/image_' + str(image) + ".jpg" # non de l'image
        cv2.imwrite(image_name, roi_color) # on capture l'image
        _, i = Data_visualisation.predict_validation_image(image_name, model, verbose=False)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = ["BIEN", "NEZ", "NUL", "NO MASQUE"]
        cv2.putText(frame, text[i-1], (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        image+=1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return frame

def detect_from_video(model):
    try:
        if not os.path.exists(capture_fodler):
            os.makedirs(capture_fodler)
    except OSError:
        print("ERROR")
    
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