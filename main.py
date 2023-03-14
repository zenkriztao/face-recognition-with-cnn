import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

class_face = cv2.CascadeClassifier('./Harcascade/haarcascade.xml')
class_models = load_model('./Models/model_v_47.hdf5')
labels = {0: 'Marah', 1: 'Jijik', 2: 'Takut', 3: 'Senang', 4: 'Netral', 5: 'Sedih', 6: 'Terkejut'}

cap = cv2.VideoCapture(0)
while True:
    success,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    fde = class_face.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    randomfde = []
    arr = []
    for (x, y, w, h) in fde:
        cv2.rectangle(img, (x, y), (x+w, y+h), (100, 0, 0), 2)
        color_gray = gray[y:y+h, x:x+w]
        color_gray = cv2.resize(color_gray, (48, 48), interpolation=cv2.INTER_AREA)
        randomfde.append(color_gray)
        arr.append((x, w, y, h))
    i = 0
    for face in randomfde:
        accurate = face.astype("float") / 255.0
        accurate = img_to_array(accurate)
        accurate = np.expand_dims(accurate, axis=0)
        predictions = class_models.predict(accurate)[0]
        label = labels[predictions.argmax()]
        coordinate = (arr[i][0] + int((arr[i][1]/2)),
                      abs(arr[i][2] - 10))
        i = + 1
        cv2.putText(img, label, coordinate,
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 0), thickness=2)
    cv2.imshow("Software Deteksi Wajah dan Emosi",img)

    if cv2.waitKey(1) == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()

