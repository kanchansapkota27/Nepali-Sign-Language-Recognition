#Sytem Imports
from PyQt5 import QtGui
import numpy as np
import sys
import threading
from functools import partial
import time
#UI Imports
from PyQt5.QtWidgets import QMessageBox,QAction
from PyQt5 import QtCore
from PyQt5.QtWidgets import  QApplication,QMainWindow,QMessageBox,QFileDialog,QAction
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSignal,pyqtSlot,QThread
#Image Processing and ML Imports
import qimage2ndarray as q2a
import cv2
from predictBackend import Predictor
#Audio Imports
import pyttsx3
#Custom Imports
from dashPage import Ui_MainWindow
from entryScreen import Ui_SplashWindow
from constants import indicator_label_padding,indicator_colors,nb_frames,voiceId,voiceSpeed




class MainWindow(QMainWindow):

    frame_display_signal=pyqtSignal(np.ndarray)
    update_current_predicition_signal=pyqtSignal(str)
    speak_signal=pyqtSignal(str)

    def __init__(self,videoPath,isTestVideo):
        super(MainWindow,self).__init__()
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        self.setMinimumSize(QtCore.QSize(1500,1000))
        self.showMaximized()
        self.setWindowTitle('NSL-Dashboard')
        self.videoType=videoPath
        self.isTestVideo=isTestVideo
        self.set_status_bar()
        self.ui.start_rec_btn.setShortcut('R')
        self.ui.stop_rec_btn.setShortcut('T')
        if self.isTestVideo:
            self.toggle_start_stop(False)

        #Fetch Initals
        self.vid_width=self.ui.video_frame.width()
        self.vid_height=self.ui.video_frame.height()
        self.ui.video_label.setScaledContents(True)
        #Status
        self.status={
            0:'Recording',
            1:'Prediciting',
            2:'Idle'
        }
        self.is_idle=True
        self.is_recording=False
        self.is_predicting=False
        #Empty Initializations
        self.current_predicition=''
        self.predicition_sequence=''
        #Signals
        self.frame_display_signal.connect(self.display_frame)
        self.update_current_predicition_signal.connect(self.update_current_predicition)
        #Connections
        self.video_running=True
        self.ui.actionChoose_Video.triggered.connect(self.choose_new_file)
        self.ui.actionHelp.triggered.connect(self.show_help)
        self.ui.actionAbout.triggered.connect(self.show_about)
        self.ui.actionExit.triggered.connect(exit)

        self.ui.start_rec_btn.clicked.connect(partial(self.toggle_status,self.status[0]))
        self.ui.stop_rec_btn.clicked.connect(partial(self.toggle_status,self.status[2]))
        self.ui.speak_output_btn.clicked.connect(self.begin_speak)
        self.ui.clear_btn.clicked.connect(self.clear_all)
        self.speak_class=SpeechWorker()
        self.speak_signal.connect(self.speak_class.get_sequence)
        #Start Video Thread
        self.initialize_video_thread()


    def set_status_bar(self):
        if self.videoType==0:
            self.ui.statusbar.showMessage('Operating in Live Video Mode')
        if self.videoType!=0 and self.isTestVideo:
            self.ui.statusbar.showMessage('Operating in Video Test Mode')
        if self.videoType!=0 and not self.isTestVideo:
            self.ui.statusbar.showMessage('Operating in Recorded Video Mode')

    def toggle_start_stop(self,isenabled):
        self.ui.start_rec_btn.setEnabled(isenabled)
        self.ui.stop_rec_btn.setEnabled(isenabled)
        if isenabled:
            self.ui.start_rec_btn.setToolTip("Press [R] to start record ")
            self.ui.stop_rec_btn.setToolTip("Press [R] to start record ")
        else:
            self.ui.start_rec_btn.setToolTip("Disabled on test mode")
            self.ui.stop_rec_btn.setToolTip("Disabled on test mode")

    def toggle_status(self,status):
        if status==self.status[0]:
            self.is_recording=True
            #UI Changes
            self.ui.indicator_label.setText('Recording')
            self.ui.indicator_label.setStyleSheet(f'padding:{indicator_label_padding};background-color:{indicator_colors["recording"]};border-radius:5;')
            #Button Deactivate
            self.ui.start_rec_btn.setEnabled(False)

        if status==self.status[1]:
            self.is_predicting=True
            self.ui.indicator_label.setText('Prediciting')
            self.ui.indicator_label.setStyleSheet(f'padding:{indicator_label_padding};background-color:{indicator_colors["predicting"]};border-radius:5;')
            self.toggle_start_stop(False)

        if status==self.status[2]:
            self.is_idle=True
            self.is_recording=False
            self.is_predicting=False
            self.ui.indicator_label.setText('Idle')
            self.ui.indicator_label.setStyleSheet(f'padding:{indicator_label_padding};background-color:{indicator_colors["idle"]};border-radius:5;')
            self.toggle_start_stop(True)

    def initialize_video_thread(self):
        self.video_thread=threading.Thread(target=self.display_video,daemon=True)
        self.video_thread.start()

    def central_image_maker(self,img_text,bg_color_rgb=(0,0,0)):
        width=self.vid_width
        height=self.vid_height
        blank_frame=np.zeros((width,height,3),np.uint8)
        rgb_2_bgr=tuple(reversed(bg_color_rgb))
        blank_frame[:]=rgb_2_bgr
        text=img_text
        font=cv2.FONT_HERSHEY_COMPLEX
        text_color=(255,255,255)
        fontScale=1.0
        thickness=2
        textSize=cv2.getTextSize(text,font,1,2)[0]
        textX=(blank_frame.shape[1]-textSize[0])//2
        textY=(blank_frame.shape[0]-textSize[1])//2
        cv2.putText(blank_frame,img_text,(textX,textY),font,fontScale,text_color,thickness,cv2.LINE_AA)
        return blank_frame


    @pyqtSlot(np.ndarray)
    def display_frame(self,frame):
        frame_img=q2a.array2qimage(frame)
        frame_img = frame_img.rgbSwapped()
        self.ui.video_label.setPixmap(QPixmap.fromImage(frame_img))

    @pyqtSlot(str)
    def update_current_predicition(self,predicition_text):
        self.current_predicition=predicition_text
        self.ui.current_prediciton_label.setText(predicition_text.upper())
        self.predicition_sequence+=' '+self.current_predicition
        self.ui.sequence_output.clear()
        self.ui.sequence_output.append(self.predicition_sequence)

    def clear_all(self):
        self.predicition_sequence=''
        self.current_predicition='-'
        self.ui.sequence_output.clear()
        self.ui.current_prediciton_label.setText(self.current_predicition)


    def display_video(self):
        frames_list=[]
        cap=cv2.VideoCapture(self.videoType)
        if self.isTestVideo:
            self.is_recording=True
            self.toggle_status(self.status[0])
        while self.video_running:
            ret,frame=cap.read()
            if not ret:
                self.is_recording=False
                break
            frame=cv2.resize(frame,(self.vid_width,self.vid_height))
            if self.isTestVideo or self.videoType!=0: #Last Changed
                time.sleep(0.03)
            self.frame_display_signal.emit(frame)
            if self.is_recording:
                frames_list.append(frame)
            if (len(frames_list)>nb_frames and self.is_recording==False and not self.isTestVideo):
                self.is_predicting=True
                self.toggle_status(self.status[1])
                print(len(frames_list))
                prediction=self.prediction(frames_list)
                self.update_current_predicition_signal.emit(str(prediction))
                self.toggle_status(self.status[2])
                frames_list.clear()
                self.is_predicting=False

        if self.isTestVideo and self.is_recording==False:
            self.is_predicting=True
            self.toggle_status(self.status[1])
            prediction=self.prediction(frames_list)
            self.update_current_predicition_signal.emit(str(prediction))
            self.toggle_status(self.status[2])
            frames_list.clear()
            self.is_predicting=False
            completed_frame=self.central_image_maker('Completed')
            self.frame_display_signal.emit(completed_frame)
            self.toggle_start_stop(False)


    def prediction(self,frames_list):
        predicitng_frames=self.central_image_maker("Predicting")
        self.frame_display_signal.emit(predicitng_frames)
        predictor_instance=Predictor(frames_list)
        predicted_label=predictor_instance.predict()
        return predicted_label

    def begin_speak(self):
        self.speak_class.start()
        self.speak_signal.emit(self.predicition_sequence)

    def choose_new_file(self):
        self.video_running=False
        self.splash=SplashScreen()
        self.splash.show()
        self.close()

    def show_help(self):
        help_string=r"""
        <ul>
        <li>Press [Start Button] or [R] key to start recording your nepali sign.<br></li>
        <li>Press [Stop Button] or [T] key to stop recording after completing sign.<br></li>
        <li>The system will predict the sign among labels of ['father', 'food', 'promise', 'tea', 'wife'].<br></li>
        <li>The current predicition will be shown in orange box.<br></li>
        <li>All the prediction for the given type will be shown in pink box.<br></li>
        <li>The clear button will clear all the outputs.<br></li>
        <hr>
        </ul>
        <b>Note:</b>The speak button will speak back the sequence output not the current one.
        """
        help_msg=QMessageBox(self)
        help_msg.setInformativeText(help_string)
        help_msg.setWindowTitle('Help')
        retval=help_msg.exec_()

    def show_about(self):
        about_string=r"""
        <b>Sharda University</b><br>

        <b>B.Tech Final Year Major Project</b><br>

        <b>Nepali Sign Language Recognition System</b><br>
        <hr><br>
        <b>Developed By:</b><br>
        <i>Kanchan Sapkota<br>
        Sailesh Rana<br>
        Santosh Kandal<br>
        Yugaraj Tamang</i><br>
        <hr><br>
        <b>Under Guidance of:</b><br>
        <i>Prof. Pankaj Sharma,SET</i>

        """
        about_msg=QMessageBox()
        about_msg.setWindowTitle('About Project')
        about_msg.setInformativeText(about_string)
        retval=about_msg.exec_()



class SpeechWorker(QThread):
    def __init__(self,parent=None):
        QThread.__init__(self,parent)
        self.sequence=''
        self.running=False

    def run(self):
        engine=pyttsx3.init()
        engine.setProperty('rate',voiceSpeed)
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[voiceId].id)
        engine.say(self.sequence)
        engine.runAndWait()

    @pyqtSlot(str)
    def get_sequence(self,sequence):
        self.sequence=sequence

class SplashScreen(QMainWindow):
    def __init__(self):
        super(SplashScreen,self).__init__()
        self.ui=Ui_SplashWindow()
        self.ui.setupUi(self)
        self.setMinimumSize(QtCore.QSize(1500,1000))
        self.showMaximized()
        #Connections
        is_test=self.ui.is_test_checkbox.isChecked()
        self.ui.continue_btn.clicked.connect(self.load_main)
        self.ui.choose_file_btn.clicked.connect(self.videoPicker)
        self.videoPath=0


    def videoPicker(self):
        selection=QFileDialog.getOpenFileName(self,'Select video file','./',"Video Files (*.avi *.mp4 *.mkv *.webm)")
        videoPath=selection[0]
        if videoPath!='':
            self.ui.videoPath_lineEdit.setText(videoPath)
            self.videoPath=videoPath
        else:
            self.ui.is_test_checkbox.setChecked(False)
            self.ui.videoPathtextEdit.setText('Live Video will be used.')


    def load_main(self):
        if len(self.ui.videoPath_lineEdit.text())<3:
            self.ui.is_test_checkbox.setChecked(False)
        is_test=self.ui.is_test_checkbox.isChecked()
        print("IS TEST",is_test)
        self.window=MainWindow(self.videoPath,is_test)
        self.window.show()
        self.close()



print('Compiling Exception')
def exception_hook(exctype, value, traceback):
        print(exctype, value, traceback)
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)

if __name__=='__main__':
    sys._excepthook = sys.excepthook
    sys.excepthook = exception_hook
    app=QApplication(sys.argv)
    windows=SplashScreen()
    windows.show()
    sys.exit(app.exec_())