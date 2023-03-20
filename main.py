import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import sys
import streamlit as st
 
# Load Model
class_face = cv2.CascadeClassifier('./Harcascade/haarcascade.xml')
class_models = load_model('./Models/model_k1.hdf5')
labels = {0: 'Marah', 1: 'Jijik', 2: 'Takut', 3: 'Senang', 4: 'Netral', 5: 'Sedih', 6: 'Terkejut'}

# Load Streamlit

# Load Video

# # Start Streamlit
if st.button('Start'):
    st.write("Deteksi wajah dan emosi dimulai Kelompok 1")
    st.write("Tekan tombol 'Stop' untuk menghentikan deteksi wajah dan emosi")
    st.write("Tekan tombol 'Exit' untuk keluar dari program")
    st.write("Tekan tombol 'Restart' untuk memulai deteksi wajah dan emosi dari awal")
    screen = cv2.VideoCapture(0)
    # Looping Video Frame by Frame 
    while True:
        success,img = screen.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        fde = class_face.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        randomfde = []
        arr = []
        
    # Draw Rectangle and Predict Emotion 
        for (x, y, w, h) in fde:
            cv2.rectangle(img, (x, y), (x+w, y+h), (100, 0, 0), 2)
            
    # Roi (Region of Interest)
            roi_grayscale = gray[y:y+h, x:x+w]
            roi_grayscale = cv2.resize(roi_grayscale, (48, 48), interpolation=cv2.INTER_AREA)
            randomfde.append(roi_grayscale)
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
    
    # Exit Program
        if cv2.waitKey(1) == ord('q'):  
            break
    
    # Release Video and Destroy All Windows Opened by OpenCV 
    screen.release()
    cv2.destroyAllWindows()
 

# #create PyQt5 app
# app = QApplication(sys.argv)
# #create a widget 
# window = QWidget()
# window.setWindowTitle('Software Deteksi Wajah dan Emosi') 
# window.setGeometry(100, 100, 300, 200)
# window.move(60, 15)
# #create a button in the window
# button = QPushButton('Exit', window)
# button.setToolTip('This is an example button')
# button.move(100,70)
# button.clicked.connect(app.quit)
# #show the window
# window.show()
# #start the app
# sys.exit(app.exec_())
  
     
