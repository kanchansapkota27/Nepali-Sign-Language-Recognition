# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'finaldash.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(808, 498)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("")
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.video_frame = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.video_frame.sizePolicy().hasHeightForWidth())
        self.video_frame.setSizePolicy(sizePolicy)
        self.video_frame.setStyleSheet("QFrame#video_frame{\n"
"background-color:#2c3e50;\n"
"}")
        self.video_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.video_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.video_frame.setObjectName("video_frame")
        self.video_frame_layout = QtWidgets.QGridLayout(self.video_frame)
        self.video_frame_layout.setContentsMargins(7, -1, -1, -1)
        self.video_frame_layout.setObjectName("video_frame_layout")
        self.video_label = QtWidgets.QLabel(self.video_frame)
        self.video_label.setStyleSheet("QLabel#video_label{\n"
"color:white;\n"
"}")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setObjectName("video_label")
        self.video_frame_layout.addWidget(self.video_label, 0, 0, 1, 1)
        self.horizontalLayout.addWidget(self.video_frame)
        self.controls_frame = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.controls_frame.sizePolicy().hasHeightForWidth())
        self.controls_frame.setSizePolicy(sizePolicy)
        self.controls_frame.setStyleSheet("QFrame#controls_frame{\n"
"background-color:#7f8c8d;\n"
"}")
        self.controls_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.controls_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.controls_frame.setObjectName("controls_frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.controls_frame)
        self.verticalLayout.setObjectName("verticalLayout")
        self.controls_top_frame = QtWidgets.QFrame(self.controls_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.controls_top_frame.sizePolicy().hasHeightForWidth())
        self.controls_top_frame.setSizePolicy(sizePolicy)
        self.controls_top_frame.setStyleSheet("QFrame#controls_top_frame{\n"
"border-radius:3;\n"
"background-color:#778ca3;\n"
"}\n"
"")
        self.controls_top_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.controls_top_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.controls_top_frame.setObjectName("controls_top_frame")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.controls_top_frame)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.start_rec_btn = QtWidgets.QPushButton(self.controls_top_frame)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.start_rec_btn.setFont(font)
        self.start_rec_btn.setStyleSheet("QPushButton#start_rec_btn{\n"
"border:None;\n"
"background-color:#4cd137;\n"
"border-radius:5;\n"
"padding:30;\n"
"color:white;\n"
"}\n"
"\n"
"QPushButton#start_rec_btn:hover{\n"
"background-color:#44bd32;\n"
"}")
        self.start_rec_btn.setObjectName("start_rec_btn")
        self.horizontalLayout_2.addWidget(self.start_rec_btn)
        self.stop_rec_btn = QtWidgets.QPushButton(self.controls_top_frame)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.stop_rec_btn.setFont(font)
        self.stop_rec_btn.setStyleSheet("QPushButton#stop_rec_btn{\n"
"border:None;\n"
"background-color:#fd9644;\n"
"border-radius:5;\n"
"padding:30;\n"
"color:white;\n"
"}\n"
"\n"
"QPushButton#stop_rec_btn:hover{\n"
"background-color:#fa8231;\n"
"}\n"
"")
        self.stop_rec_btn.setObjectName("stop_rec_btn")
        self.horizontalLayout_2.addWidget(self.stop_rec_btn)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.indicator_label = QtWidgets.QLabel(self.controls_top_frame)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.indicator_label.setFont(font)
        self.indicator_label.setStyleSheet("QLabel#indicator_label{\n"
"padding:30;\n"
"border-radius:5;\n"
"background-color:#9c88ff;\n"
"color:#353b48;\n"
"}\n"
"")
        self.indicator_label.setAlignment(QtCore.Qt.AlignCenter)
        self.indicator_label.setObjectName("indicator_label")
        self.verticalLayout_2.addWidget(self.indicator_label)
        self.verticalLayout.addWidget(self.controls_top_frame)
        self.frame = QtWidgets.QFrame(self.controls_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setStyleSheet("border-radius:3;\n"
"background-color:#778ca3;")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.current_prediciton_label = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.current_prediciton_label.setFont(font)
        self.current_prediciton_label.setStyleSheet("QLabel#current_prediciton_label\n"
"{\n"
"padding:3;\n"
"border-radius:5;\n"
"background-color:#ffbe76;\n"
"}")
        self.current_prediciton_label.setAlignment(QtCore.Qt.AlignCenter)
        self.current_prediciton_label.setObjectName("current_prediciton_label")
        self.gridLayout_2.addWidget(self.current_prediciton_label, 0, 0, 1, 1)
        self.verticalLayout.addWidget(self.frame)
        self.controls_bottom_frame = QtWidgets.QFrame(self.controls_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.controls_bottom_frame.sizePolicy().hasHeightForWidth())
        self.controls_bottom_frame.setSizePolicy(sizePolicy)
        self.controls_bottom_frame.setStyleSheet("border-radius:3;\n"
"background-color:#778ca3;")
        self.controls_bottom_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.controls_bottom_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.controls_bottom_frame.setObjectName("controls_bottom_frame")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.controls_bottom_frame)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.sequence_output = QtWidgets.QTextEdit(self.controls_bottom_frame)
        font = QtGui.QFont()
        font.setFamily("MS Gothic")
        font.setPointSize(10)
        self.sequence_output.setFont(font)
        self.sequence_output.setStyleSheet("QTextEdit#sequence_output{\n"
"\n"
"background-color:#fab1a0;\n"
"border-radius:5;\n"
"\n"
"}")
        self.sequence_output.setDocumentTitle("")
        self.sequence_output.setReadOnly(True)
        self.sequence_output.setObjectName("sequence_output")
        self.gridLayout_3.addWidget(self.sequence_output, 0, 0, 1, 1)
        self.verticalLayout.addWidget(self.controls_bottom_frame)
        self.frame_2 = QtWidgets.QFrame(self.controls_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setStyleSheet("border-radius:3;\n"
"background-color:#778ca3;")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.speak_output_btn = QtWidgets.QPushButton(self.frame_2)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.speak_output_btn.setFont(font)
        self.speak_output_btn.setStyleSheet("QPushButton#speak_output_btn{\n"
"border:None;\n"
"background-color:#00a8ff;\n"
"border-radius:5;\n"
"padding:30;\n"
"color:white;\n"
"}\n"
"\n"
"QPushButton#speak_output_btn:hover{\n"
"background-color:#0097e6;\n"
"}")
        self.speak_output_btn.setObjectName("speak_output_btn")
        self.horizontalLayout_3.addWidget(self.speak_output_btn)
        self.clear_btn = QtWidgets.QPushButton(self.frame_2)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.clear_btn.setFont(font)
        self.clear_btn.setStyleSheet("QPushButton#clear_btn{\n"
"border:None;\n"
"background-color:#fc5c65;\n"
"border-radius:5;\n"
"padding:30;\n"
"color:white;\n"
"}\n"
"\n"
"QPushButton#clear_btn:hover{\n"
"background-color:#eb3b5a;\n"
"}")
        self.clear_btn.setObjectName("clear_btn")
        self.horizontalLayout_3.addWidget(self.clear_btn)
        self.verticalLayout.addWidget(self.frame_2)
        self.horizontalLayout.addWidget(self.controls_frame)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 808, 20))
        self.menubar.setObjectName("menubar")
        self.menuOptions = QtWidgets.QMenu(self.menubar)
        self.menuOptions.setObjectName("menuOptions")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setStyleSheet("QStatusBar{\n"
"color:#273c75;\n"
"padding:30;\n"
"}")
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionChoose_Video = QtWidgets.QAction(MainWindow)
        self.actionChoose_Video.setObjectName("actionChoose_Video")
        self.actionHelp = QtWidgets.QAction(MainWindow)
        self.actionHelp.setObjectName("actionHelp")
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.menuOptions.addAction(self.actionChoose_Video)
        self.menuOptions.addAction(self.actionHelp)
        self.menuOptions.addAction(self.actionAbout)
        self.menuOptions.addSeparator()
        self.menuOptions.addAction(self.actionExit)
        self.menubar.addAction(self.menuOptions.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.video_label.setText(_translate("MainWindow", "Loading"))
        self.start_rec_btn.setText(_translate("MainWindow", "Start"))
        self.stop_rec_btn.setText(_translate("MainWindow", "Stop"))
        self.indicator_label.setToolTip(_translate("MainWindow", "Current processing task indicator."))
        self.indicator_label.setText(_translate("MainWindow", "Idle"))
        self.current_prediciton_label.setToolTip(_translate("MainWindow", "Output of most recent prediciton"))
        self.current_prediciton_label.setText(_translate("MainWindow", "_"))
        self.sequence_output.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Gothic\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.speak_output_btn.setText(_translate("MainWindow", "Speak"))
        self.clear_btn.setText(_translate("MainWindow", "Clear"))
        self.menuOptions.setTitle(_translate("MainWindow", "Options"))
        self.statusbar.setToolTip(_translate("MainWindow", "Current Input Mode"))
        self.actionChoose_Video.setText(_translate("MainWindow", "Choose Video"))
        self.actionHelp.setText(_translate("MainWindow", "Help"))
        self.actionAbout.setText(_translate("MainWindow", "About"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))