from PyQt5 import QtCore, QtGui, QtWidgets
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
import os


class Ui_MainWindow(object):
  
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Ana Ekran")
        MainWindow.resize(953, 657)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Progress Bar
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(30, 30, 871, 23))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")

        # ComboBox
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(520, 70, 241, 22))
        self.comboBox.setAcceptDrops(True)
        self.comboBox.setObjectName("comboBox")

        # TextEdit
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(30, 70, 131, 22))
        self.textEdit.setObjectName("textEdit")

        # Hedef Belirle
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(170, 70, 75, 22))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.belirle_button_clicked)

        # Egzersiz Belirle
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(774, 70, 91, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.egzersiz_button_clicked)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 953, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # ComboBox'a değerleri ekle
        self.comboBox.addItems(["Sağ El", "Sol El", "Sağ Kol", "Sol Kol"])

        # Yeni formun referansını tutacak üye değişken
        self.success_form = None

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Hedef Belirle"))
        self.pushButton_2.setText(_translate("MainWindow", "Egzersiz Belirle"))

    def belirle_button_clicked(self):
        # TextEdit'ten max değeri alıp ProgressBar'a atama yap
        max_value = int(self.textEdit.toPlainText())
        self.progressBar.setMaximum(max_value)
        
    def egzersiz_button_clicked(self):
        if self.comboBox.currentIndex() == 0:  # ComboBox'ın birinci değeri seçildi mi?
            self.process_webcam_feed_right()
        if self.comboBox.currentIndex() == 1:  # ComboBox'ın birinci değeri seçildi mi?
            self.process_webcam_feed_left()
        if self.comboBox.currentIndex() == 2:  # ComboBox'ın birinci değeri seçildi mi?
            self.process_webcam_feed_rightArm()
        if self.comboBox.currentIndex() == 3:  # ComboBox'ın birinci değeri seçildi mi?
            self.process_webcam_feed_leftArm()
    

        
    def show_success_form(self):

        self.success_form = QtWidgets.QWidget()
        self.success_form.setGeometry(783, 461, 400, 200)
        label = QtWidgets.QLabel(self.success_form)
        label.setGeometry(QtCore.QRect(50, 50, 300, 100))
        label.setFont(QtGui.QFont("Arial", 20, QtGui.QFont.Bold))
        label.setText("BÜYÜK BAŞARI")
        self.progressBar.setProperty("value", 0)
        self.success_form.show()

    
    def process_webcam_feed_right(self):
        
        model_path = 'RightHand.pkl'
        count = 0
        cap = cv2.VideoCapture(0)
        in_session = False  # Flag to track if the current session is ongoing
        with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            model = pickle.load(open(model_path, 'rb'))
            cv2.namedWindow('Raw Webcam Feed')
            cv2.moveWindow('Raw Webcam Feed', 644, 327)  # Set the position (x, y)
    
            while cap.isOpened():
                ret, frame = cap.read()
    
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        
    
                results = holistic.process(image)
    
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, 
                                                          mp.solutions.holistic.HAND_CONNECTIONS, 
                                                          mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                                          mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
                try:
                    pose = results.right_hand_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
    
                    row = pose_row
    
                    # Make Detections
                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]
    
                    if body_language_class == "acik":
                        if not in_session:
                            count += 1
                            in_session = True
    
                    elif body_language_class == "kapali":
                        in_session = False
    
                    if count >= self.progressBar.maximum():
                        self.show_success_form()
                        break
                        count = 0  # Reset count for the next turn
    
                    self.progressBar.setValue(count)
                    print(body_language_class, body_language_prob)
    
                    cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                    cv2.putText(image, 'SINIF'
                                , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, body_language_class.split(' ')[0]
                                , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, 'TAHMIN'
                                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2))
                                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                except Exception as error:
                    print(error)
                    pass
    
                cv2.imshow('Raw Webcam Feed', image)
    
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    
        cap.release()
        cv2.destroyAllWindows()
        
    def process_webcam_feed_left(self):
        
        model_path = 'LeftHand.pkl'
        count = 0
        cap = cv2.VideoCapture(0)
        in_session = False  # Flag to track if the current session is ongoing
        with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            model = pickle.load(open(model_path, 'rb'))
            cv2.namedWindow('Raw Webcam Feed')
            cv2.moveWindow('Raw Webcam Feed', 644, 327)  # Set the position (x, y)
    
            while cap.isOpened():
                ret, frame = cap.read()
    
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        
    
                results = holistic.process(image)
    
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, 
                                                          mp.solutions.holistic.HAND_CONNECTIONS, 
                                                          mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                                          mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
                try:
                    pose = results.left_hand_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
    
                    row = pose_row
    
                    # Make Detections
                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]
    
                    if body_language_class == "acik":
                        if not in_session:
                            count += 1
                            in_session = True
    
                    elif body_language_class == "kapali":
                        in_session = False
    
                    if count >= self.progressBar.maximum():
                        self.show_success_form()
                        break
                        count = 0  # Reset count for the next turn
    
                    self.progressBar.setValue(count)
                    print(body_language_class, body_language_prob)
    
                    cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                    cv2.putText(image, 'SINIF'
                                , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, body_language_class.split(' ')[0]
                                , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, 'TAHMIN'
                                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2))
                                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                except Exception as error:
                    print(error)
                    pass
    
                cv2.imshow('Raw Webcam Feed', image)
    
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    
        cap.release()
        cv2.destroyAllWindows()


    def process_webcam_feed_rightArm(self):
        mp_holistic= mp.solutions.holistic
        model_path = 'RightArm.pkl'
        count = 0
        cap = cv2.VideoCapture(0)
        in_session = False  # Flag to track if the current session is ongoing
        with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            model = pickle.load(open(model_path, 'rb'))
            cv2.namedWindow('Raw Webcam Feed')
            cv2.moveWindow('Raw Webcam Feed', 644, 327)  # Set the position (x, y)
    
            while cap.isOpened():
                ret, frame = cap.read()
    
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        
    
                results = holistic.process(image)
    
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    point_11 = (int(landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW].x * image.shape[1]), 
                                int(landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW].y * image.shape[0]))
                    point_13 = (int(landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST].x * image.shape[1]), 
                                int(landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST].y * image.shape[0]))
                    point_15 = (int(landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * image.shape[1]), 
                                int(landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * image.shape[0]))
                    
                    # Draw circles at the specified points
                    cv2.circle(image, point_11, 5, (0, 255, 0), -1)
                    cv2.circle(image, point_13, 5, (0, 255, 0), -1)
                    cv2.circle(image, point_15, 5, (0, 255, 0), -1)
                try:
                    landmarks = results.pose_landmarks.landmark
                    point_11 = np.array([landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW].x,
                                         landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW].y,
                                         landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW].z])              
        
                    point_13 = np.array([landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST].x,
                                         landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST].y,
                                         landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST].z])
        
                    point_15 = np.array([landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x,
                                         landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y,
                                         landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER].z])
                    
                    total = np.concatenate((point_11, point_13, point_15))
                    RHand_row = list(total)
                    
                  
                    row = RHand_row
    
                    # Make Detections
                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]
    
                    if body_language_class == "acik":
                        if not in_session:
                            count += 1
                            in_session = True
    
                    elif body_language_class == "kapali":
                        in_session = False
    
                    if count >= self.progressBar.maximum():
                        self.show_success_form()
                        break
                        count = 0  # Reset count for the next turn
    
                    self.progressBar.setValue(count)
                    print(body_language_class, body_language_prob)
    
                    cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                    cv2.putText(image, 'SINIF'
                                , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, body_language_class.split(' ')[0]
                                , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, 'TAHMIN'
                                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2))
                                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                except Exception as error:
                    print(error)
                    pass
    
                cv2.imshow('Raw Webcam Feed', image)
    
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    
        cap.release()
        cv2.destroyAllWindows()

    def process_webcam_feed_leftArm(self):
        mp_holistic= mp.solutions.holistic
        model_path = 'LeftArm.pkl'
        count = 0
        cap = cv2.VideoCapture(0)
        in_session = False  # Flag to track if the current session is ongoing
        with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            model = pickle.load(open(model_path, 'rb'))
            cv2.namedWindow('Raw Webcam Feed')
            cv2.moveWindow('Raw Webcam Feed', 644, 327)  # Set the position (x, y)
    
            while cap.isOpened():
                ret, frame = cap.read()
    
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        
    
                results = holistic.process(image)
    
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    point_11 = (int(landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW].x * image.shape[1]), 
                                int(landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW].y * image.shape[0]))
                    point_13 = (int(landmarks[mp_holistic.PoseLandmark.LEFT_WRIST].x * image.shape[1]), 
                                int(landmarks[mp_holistic.PoseLandmark.LEFT_WRIST].y * image.shape[0]))
                    point_15 = (int(landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * image.shape[1]), 
                                int(landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * image.shape[0]))
                    
                    # Draw circles at the specified points
                    cv2.circle(image, point_11, 5, (0, 255, 0), -1)
                    cv2.circle(image, point_13, 5, (0, 255, 0), -1)
                    cv2.circle(image, point_15, 5, (0, 255, 0), -1)
                try:
                    landmarks = results.pose_landmarks.landmark
                    point_11 = np.array([landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW].x,
                                         landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW].y,
                                         landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW].z])              
        
                    point_13 = np.array([landmarks[mp_holistic.PoseLandmark.LEFT_WRIST].x,
                                         landmarks[mp_holistic.PoseLandmark.LEFT_WRIST].y,
                                         landmarks[mp_holistic.PoseLandmark.LEFT_WRIST].z])
        
                    point_15 = np.array([landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER].x,
                                         landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER].y,
                                         landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER].z])
                    
                    total = np.concatenate((point_11, point_13, point_15))
                    RHand_row = list(total)
                    
                  
                    row = RHand_row
    
                    # Make Detections
                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]
    
                    if body_language_class == "acik":
                        if not in_session:
                            count += 1
                            in_session = True
    
                    elif body_language_class == "kapali":
                        in_session = False
    
                    if count >= self.progressBar.maximum():
                        self.show_success_form()
                        break
                        count = 0  # Reset count for the next turn
    
                    self.progressBar.setValue(count)
                    print(body_language_class, body_language_prob)
    
                    cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                    cv2.putText(image, 'SINIF'
                                , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, body_language_class.split(' ')[0]
                                , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, 'TAHMIN'
                                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2))
                                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                except Exception as error:
                    print(error)
                    pass
    
                cv2.imshow('Raw Webcam Feed', image)
    
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    
        cap.release()
        cv2.destroyAllWindows()    

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
