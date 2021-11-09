# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 20:12:59 2021

@author: tanch
"""

from PyQt5.QtWidgets import * 
# from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QDesktopWidget
# from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import * 
from PyQt5.QtCore import * 
import sys
import time


#%% 

def print_val(val):
    print(val)
    return val

#%%


class KFoldWindow(QMainWindow, ):  # Inherits from / is a Main Window
    '''
    Sumamary:
    Creates a GUI window for human interaction that allows the user to \n
    choose the number of folds for K-fold validation in main_analysis.
    '''
    def __init__(self):
        '''
        Initialises the class KFoldWindow, the GUI window for user-interaction, and sets it to the front of screen.
        Returns
        -------
        None.

        '''
        super().__init__()
        self.kFold = None
        # setting title
        self.setWindowTitle("K-fold Selection Box")
        self.centreWindow()
        # calling method
        self.UiComponents()
        # showing all the widgets
        self.show()
        #set flag for Window to stay on top
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        
    def UiComponents(self):
        '''
        Draws the widgets in the GUI window.

        Returns
        -------
        None.

        '''
        #draw all widgets
        self.title = self.title()
        self.radio = self.KFoldRadio()
        # self.ok = self.OK_button()
        
        
    def centreWindow(self):
        '''
        Sets the GUI window in the centre of the screen.

        Returns
        -------
        None.

        '''
        #set window size
        self.setGeometry(0, 0, 1100, 400)
        # setting geometry
        qtRectangle = self.frameGeometry()
        # find centrepoint of screen
        centerPoint = QDesktopWidget().availableGeometry().center()
        #move window to centre
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

    def title(self):
        '''
        Defines the title widget containing the header text for the GUI window.

        Returns
        -------
        None.

        '''
        
        # setting color effect to the title
        color = QGraphicsColorizeEffect(self)
        color.setColor(Qt.black)
        # set font effect to the title
        font_head = QFont('Times', 15)
        font_head.setBold(True)

        #create a title widget
        head = QLabel("Choose between 5-fold OR 10-fold cross-validation", self)
      
        # setting font to the head
        head.setFont(font_head)
        
        # setting geometry to the head
        head.setGeometry(10, 10, 1080, 60)
      
        # setting alignment of the head
        head.setAlignment(Qt.AlignCenter)
        
        # setting colour of the head
        head.setGraphicsEffect(color)
    
    def KFoldRadio (self):
        '''
        Defines the radio buttons for the selection between 5-fold and 10-fold validation.

        Returns
        -------
        None.

        '''
        
        radiobutton = QRadioButton("5-fold", self)    
        # radiobutton.setChecked(True)
        radiobutton.kFold = 5
        radiobutton.toggled.connect(self.radioOnClicked)
        radiobutton.setGeometry(300, 80, 150, 60)
        
        radiobutton = QRadioButton("10-fold",self)
        radiobutton.kFold = 10
        radiobutton.toggled.connect(self.radioOnClicked)
        radiobutton.setGeometry(600, 80, 150, 60)
        # centralWidget = QWidget(self)  # create central widget
        # self.setCentralWidget(centralWidget)   # assign it to main window

        # vLayout = QVBoxLayout(self)  # Layout   
        # centralWidget.setLayout(vLayout)  # add layout to central widget

        # title = QLabel("Hello World", self)  # make text label
        # vLayout.addWidget(title)  # add text label to layout
    
    # def OK_button (self):
    #     okbutton = QPushButton("OK",self)
    #     okbutton.setGeometry(550, 200, 100, 60)
    #     # okbutton.setAlignment(Qt.AlignCenter)
    #     if self.radio.isChecked():
    #         okbutton.clicked.connect(lambda:self.close())
    #         print_val(self.kFold)
    #     else:
    #         self.title.setText("Choose using the radio-buttons!")
        
    def radioOnClicked(self):
        '''
        Defines the on-click effect of the radio buttons.
        
        The radio button sets the global K-fold variable and then closes the window.

        Returns
        -------
        None.

        '''
        radiobutton = self.sender()
        if radiobutton.isChecked():
            self.kFold = radiobutton.kFold
            print_val(radiobutton.kFold)
            self.close()
#%%
def window():
    '''
    Initalises displays the GUI window for the class KFoldWindow.
    

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    app = QApplication(sys.argv)  # start app
    mainWin = KFoldWindow()  # create main window
    mainWin.show()  # show it
    sys.exit( app.exec_() )  # close app when main window closed
    return app.KFold