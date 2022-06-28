# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 20:12:59 2021

@author: tanch
"""

from PyQt5.QtWidgets import QMainWindow, QLabel, QDesktopWidget, QRadioButton, QPushButton, QGraphicsColorizeEffect, QApplication
from PyQt5.QtCore import Qt 
from PyQt5.QtGui import QFont
import sys


#%% class for GUI window

class KFoldWindow(QMainWindow, ):  # Inherits from / is a Main Window
    '''
    K-fold Radio window
    
    Summary:
    Creates a GUI window for human interaction that allows the user to \n
    choose the number of folds for K-fold validation in main_analysis.
    '''
    def __init__(self):
        '''
        Initialise Main Window
        
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
        self.activateWindow()
        
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        
        
    def UiComponents(self):
        '''
        UI components
        
        Draws the widgets in the GUI window.

        Returns
        -------
        None.

        '''
        #draw all widgets
        self.title()
        self.KFoldRadio()
        self.OK_button()
        

    def centreWindow(self):
        '''
        Centre GUI window
        
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
        GUI Title Header
        
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
        self.head = QLabel("Choose between 5-fold OR 10-fold cross-validation", self)
      
        # setting font to the head
        self.head.setFont(font_head)
        
        # setting geometry to the head
        self.head.setGeometry(10, 10, 1080, 60)
      
        # setting alignment of the head
        self.head.setAlignment(Qt.AlignCenter)
        
        # setting colour of the head
        self.head.setGraphicsEffect(color)
    
    def KFoldRadio (self):
        '''
        K-fold Radio buttons
        
        Defines the radio buttons for the selection between 5-fold and 10-fold validation.

        Returns
        -------
        None.

        '''
        
        self.radiobutton = QRadioButton("5-fold", self)    
        # radiobutton.setChecked(True)
        self.radiobutton.kFold = 5
        self.radiobutton.toggled.connect(self.radioOnClicked)
        self.radiobutton.setGeometry(300, 80, 150, 60)
        
        self.radiobutton = QRadioButton("10-fold",self)
        self.radiobutton.kFold = 10
        self.radiobutton.toggled.connect(self.radioOnClicked)
        self.radiobutton.setGeometry(600, 80, 150, 60)
        # centralWidget = QWidget(self)  # create central widget
        # self.setCentralWidget(centralWidget)   # assign it to main window

        # vLayout = QVBoxLayout(self)  # Layout   
        # centralWidget.setLayout(vLayout)  # add layout to central widget

        # title = QLabel("Hello World", self)  # make text label
        # vLayout.addWidget(title)  # add text label to layout
    
    def OK_button (self):
        self.okbutton = QPushButton("OK",self)
        self.okbutton.setGeometry(500, 200, 100, 60)

        self.okbutton.clicked.connect(self.okOnClicked)

        
    def radioOnClicked(self):
        '''
        Radio Button on-toggle
        
        Defines the on-click effect of the radio buttons.
        
        The radio button sets the global K-fold variable and then closes the window.

        Returns
        -------
        None.

        '''
        radiobutton = self.sender()
        if radiobutton.isChecked():
            self.kFold = radiobutton.kFold

            # self.close()
            
    def okOnClicked(self):

        if self.kFold != None:
            self.close()
        else:
            self.head.setText("Choose validation folds using the radio-buttons!")
#%%
def window():
    '''
    Call K-fold validation GUI
    
    Initalises displays the GUI window for the class KFoldWindow.
    
    Returns
    -------
    int
        The k-value for K-fold validation (5 or 10).

    '''
    app = QApplication(sys.argv)  # start app
    mainWin = KFoldWindow()  # create main window
    mainWin.show()  # show it
    app.exec_() # close app when main window closed
    return mainWin.kFold