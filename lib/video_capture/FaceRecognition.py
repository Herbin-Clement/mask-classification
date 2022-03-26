import cv2


from lib import Data_visualisation
    

class FaceRecognition():
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def __init__(self, model):
        self.model = model

    def detect(self, gray, frame, width, height):
        '''
        detect from a image if there are faces.

        :param gray:
        :param frame: image
        :param width: width of the image
        :param height: height of the image 
        
        :return: the final image with a prediction if a face is detect 
        '''
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            x, y, w, h = self.update_edge_face(x, y, w, h, width, height, 100, 100)
            roi_color = frame[y:y+h, x:x+h]
            pred = Data_visualisation.predict_image(roi_color, self.model, verbose=True)
            font = cv2.FONT_HERSHEY_SIMPLEX
            # text = ["1", "2", "3", "4"]
            text = ["BIEN", "NEZ", "BOUCHE", "PAS DE MASQUE"]
            cv2.putText(frame, text[pred-1], (x, y - 10), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return frame

    def detect_from_video(self):
        '''
        Launch the camera and analyse the video
        '''
        vcap = cv2.VideoCapture(0)

        width  = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        while True:
            _, frame = vcap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            canvas = self.detect(gray, frame, width, height)
            cv2.imshow('Video', canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) <1:
                break
        vcap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def update_edge_face(self, x, y, w, h, width, height, shiftX, shiftY):
        '''
        :param x: x coordinate     
        :param y: y coordinate
        :param w: width of square 
        :param h: height of the square 
        :param shiftX:
        :param ShiftY:
        :param width: width of the frame
        :param height: height of the frame

        rtype: tuple
        '''
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