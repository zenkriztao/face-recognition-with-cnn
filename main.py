import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('./Harcascade/haarcascade_frontalface_default.xml')
classifier=load_model('./Models/model_v_47.hdf5')
class_labels={0: 'Marah', 1: 'Jijik', 2: 'Takut', 3: 'Senang', 4: 'Netral', 5: 'Sedih', 6: 'Terkejut'}

cap = cv2.VideoCapture(0)
while True:
    ret,img = cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    allfaces = []
    rects = []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (100, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        allfaces.append(roi_gray)
        rects.append((x, w, y, h))
    i = 0
    for face in allfaces:
        roi = face.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = classifier.predict(roi)[0]
        label = class_labels[preds.argmax()]
        label_position = (rects[i][0] + int((rects[i][1]/2)),
                      abs(rects[i][2] - 10))
        i = + 1
        cv2.putText(img, label, label_position,
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("Software Deteksi Wajah dan Emosi",img)

    if cv2.waitKey(1) == 13:  
        break

cap.release()
cv2.destroyAllWindows()


