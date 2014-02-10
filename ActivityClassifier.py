# Includes both GUI and classifier code

import sys
from PySide import QtGui
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import numpy as np
import pylab as pl
import csv

class UserInterface(QtGui.QWidget):

    def __init__(self):
        
        super(UserInterface, self).__init__()
        self.initUI()

    def initUI(self):

        self.createWidgets()

        self.createLayout()

        self.openAccelTrainButton.clicked.connect(self.showDialog)
        self.openOrientTrainButton.clicked.connect(self.showDialog)
        self.openAccelTestButton.clicked.connect(self.showDialog)
        self.openOrientTestButton.clicked.connect(self.showDialog)

        self.trainButton.clicked.connect(self.runTraining)
        self.testButton.clicked.connect(self.classifyData)

        self.resize(650, 650)
        self.center()
        self.setWindowTitle('Human Activity Classifier')
        self.show()

    def createWidgets(self):

        self.accelTrainLabel = QtGui.QLabel('Accelerometer training set location: ')
        self.orientTrainLabel = QtGui.QLabel('Orientation training set location: ')
        self.accelTestLabel = QtGui.QLabel('Accelerometer test set location: ')
        self.orientTestLabel = QtGui.QLabel('Orientation test set location: ')

        self.accelTrainLabel.setMargin(10)
        self.orientTrainLabel.setMargin(10)
        self.accelTestLabel.setMargin(10)
        self.orientTestLabel.setMargin(10)
        
        self.openAccelTrainButton = QtGui.QPushButton("Browse")
        self.openOrientTrainButton = QtGui.QPushButton("Browse")
        self.openAccelTestButton = QtGui.QPushButton("Browse")
        self.openOrientTestButton = QtGui.QPushButton("Browse")

        self.openAccelTrainButton.setObjectName("accelTrain")
        self.openOrientTrainButton.setObjectName("orientTrain")
        self.openAccelTestButton.setObjectName("accelTest")
        self.openOrientTestButton.setObjectName("orientTest")

        self.trainButton = QtGui.QPushButton("Train")
        self.testButton = QtGui.QPushButton("Test")

        self.accelTrainText = QtGui.QLineEdit()
        self.orientTrainText = QtGui.QLineEdit()
        self.accelTestText = QtGui.QLineEdit()
        self.orientTestText = QtGui.QLineEdit()

        self.classifierLabel = QtGui.QLabel('Classifier: ')
        self.dropDownList = QtGui.QComboBox(self)
        self.dropDownList.addItems(["SVM", "KNN", "Decision Tree", "Naive Bayes", "Random Forest"])

        self.outputLabel = QtGui.QLabel('Output: ')
        self.outputTextArea = QtGui.QTextEdit()
        self.outputTextArea.setReadOnly(True)

    def createLayout(self):

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.accelTrainLabel)
        hbox.addWidget(self.accelTrainText)
        hbox.addWidget(self.openAccelTrainButton)

        hbox2 = QtGui.QHBoxLayout()
        hbox2.addWidget(self.orientTrainLabel)
        hbox2.addSpacing(15)
        hbox2.addWidget(self.orientTrainText)
        hbox2.addWidget(self.openOrientTrainButton)

        hbox3 = QtGui.QHBoxLayout()
        hbox3.addWidget(self.accelTestLabel)
        hbox3.addSpacing(16)
        hbox3.addWidget(self.accelTestText)
        hbox3.addWidget(self.openAccelTestButton)

        hbox4 = QtGui.QHBoxLayout()
        hbox4.addWidget(self.orientTestLabel)
        hbox4.addSpacing(30)
        hbox4.addWidget(self.orientTestText)
        hbox4.addWidget(self.openOrientTestButton)

        hbox5 = QtGui.QHBoxLayout()
        hbox5.addWidget(self.classifierLabel)
        hbox5.addWidget(self.dropDownList)
        hbox5.addStretch(1)
        hbox5.addWidget(self.trainButton)
        hbox5.addSpacing(10)
        hbox5.addWidget(self.testButton)
        hbox5.setContentsMargins(10,10,10,10)

        hbox6 = QtGui.QHBoxLayout()
        hbox6.addWidget(self.outputLabel)

        hbox7 = QtGui.QHBoxLayout()
        hbox7.addWidget(self.outputTextArea)

        vbox = QtGui.QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)
        vbox.addLayout(hbox4)
        vbox.addLayout(hbox5)
        vbox.addLayout(hbox6)
        vbox.addLayout(hbox7)
        
        self.setLayout(vbox)

    def showDialog(self):

        sender = self.sender()

        if (sender.objectName() == "accelTrain"):
            text = QtGui.QFileDialog.getOpenFileName()[0]
            self.accelTrainText.setText(text)
        elif (sender.objectName() == "orientTrain"):
            text = QtGui.QFileDialog.getOpenFileName()[0]
            self.orientTrainText.setText(text)
        elif (sender.objectName() == "accelTest"):
            text = QtGui.QFileDialog.getOpenFileName()[0]
            self.accelTestText.setText(text)
        elif (sender.objectName() == "orientTest"):
            text = QtGui.QFileDialog.getOpenFileName()[0]
            self.orientTestText.setText(text)

    def runTraining(self):

        accelFilePath = self.accelTrainText.text()
        orientFilePath = self.orientTrainText.text()
        self.classifierType = self.dropDownList.currentText()

        self.classifier = Classifier(accelFilePath, orientFilePath)
        self.classifier.readTrainingData()
        outputData = map(str, self.classifier.data)
        outputTarget = map(str, self.classifier.target)
        outputDataString = "Training Data: \n"
        outputTargetString = "Target Classes: \n"
        outputTargetString += "(Standing = 0, Walking = 1, Running = 2)\n\n"

        for i in outputData:
            outputDataString += i + "\n"

        for i in outputTarget:
            outputTargetString += i + ", "
        
        self.outputTextArea.append("(# samples x # features) = %s\n" % (self.classifier.data.shape,))
        self.outputTextArea.append(outputDataString)
        self.outputTextArea.append(outputTargetString)
        self.outputTextArea.append("\n\n" + self.classifier.crossValidate(self.classifierType))

        predictionValues = self.classifier.runClassificationTrain(self.classifierType)

        confusionMatrix = confusion_matrix(self.classifier.target, predictionValues)
        confusionMatrixString = "\n\nConfusion Matrix: \n\n"
        confusionMatrixString += "Standing \t Walking \t Running\n\n"
        counter = 0

        for i in confusionMatrix:
            for j in i:
                confusionMatrixString += " " + str(j) + "\t"
            if (counter == 0):
                confusionMatrixString += "Standing\n\n"
            elif (counter == 1):
                confusionMatrixString += "Walking\n\n"
            elif (counter == 2):
                confusionMatrixString += "Running\n\n"
            counter += 1
            
        self.outputTextArea.append(confusionMatrixString)

    def classifyData(self):

        self.classifier.testAccelFilePath = self.accelTestText.text()
        self.classifier.testRotationFilePath = self.orientTestText.text()
        self.classifier.readTestData()

        predictionValues = self.classifier.runClassificationTest(self.classifierType)

        testData = map(str, predictionValues)
        actualTestData = map(str, self.classifier.testData)
        outputDataString = "Test Data: \n"
        outputTestString = "Test Results: \n"
        outputTestString += "(Standing = 0, Walking = 1, Running = 2)\n\n"

        for i in testData:
            outputTestString += i + ", "

        for i in actualTestData:
            outputDataString += i + "\n"
        
        self.outputTextArea.append("\n-----------------------------------\n")
        self.outputTextArea.append(outputDataString + "\n\n")
        self.outputTextArea.append(outputTestString + "\n")

        self.plotData(testData)

    def plotData(self, data):

        ax = pl.subplot(111)
        counter = 0
        a, b, c = 0, 0, 0

        for i in data:
            if i == '0':
                a = ax.scatter(counter, i, s=40, c='b', marker='o', faceted=False)
            elif i == '1':
                b = ax.scatter(counter, i, s=40, c='r', marker='+', faceted=False)
            elif i == '2':
                c = ax.scatter(counter, i, s=40, c='g', marker='*', faceted=False)
            counter += 1

        pl.xlabel("Time (seconds)")
        pl.ylabel("State")
        pl.title("Activity Classification")
        pl.legend([a, b, c], ["Standing", "Walking", "Running"], bbox_to_anchor=(0, 0, 1, 1), bbox_transform=pl.gcf().transFigure)
        pl.grid(True)
        pl.show()

    def center(self):
        
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


class Classifier:

    accelFilePath = ""
    rotationFilePath = ""
    data = np.array([])
    target = np.array([])
    testData = np.array([])
    testAccelFilePath = ""
    testRotationFilePath = ""
    
    def __init__(self, accelFilePath, rotationFilePath):
        self.accelFilePath = accelFilePath
        self.rotationFilePath = rotationFilePath

    def readTrainingData(self):
        
        meanAccel = []
        actualClass = []
        counter, rotationCounter = 0, 0
        previousTime = 0
        xAngle, yAngle, zAngle = 0, 0, 0
        
        with open(self.accelFilePath, 'rb') as csvfile, open(self.rotationFilePath, 'rb') as csvfile2:
            accelReader = csv.reader(csvfile)
            rotationReader = csv.reader(csvfile2)
            rotationRow = rotationReader.next()
            
            for row in accelReader:
                if (counter == 0 or row[2] != previousTime) and row[2] != "":
##                    meanAccel.append([row[8], row[13], row[18]])
                    actualClass.append(row[21])
                    previousTime = row[2]
                    counter += 1
                    
                    if counter == 1:
                        meanAccel.append([row[8], row[13], row[18]])
                        rotationRow = rotationReader.next()
                        meanAccel.pop(0)
                        actualClass.pop(0)
                        
                    while rotationRow[2] <= previousTime and counter != 1:
                        #collect aggregated angle data
##                        print previousTime
##                        print rotationRow[2]
                        if rotationRow[5] != "":
                            xAngle += float(rotationRow[5])
                            yAngle += float(rotationRow[6])
                            zAngle += float(rotationRow[4])
                            rotationCounter += 1
                        try:
                            rotationRow = rotationReader.next()
                        except StopIteration:
                            break
                        
                    if rotationCounter != 0:
                        xAngle /= rotationCounter
                        yAngle /= rotationCounter
                        zAngle /= rotationCounter
                    rotationCounter = 0
                    
                    if (counter > 1):
                        x = [float(row[8]), float(row[13]), float(row[18])]
                        standardDeviation = [float(row[10]), float(row[15]), float(row[20])]
                        temp = self.rotateAccelerationVector(xAngle, yAngle, zAngle, x)
                        x = temp.tolist()
                        x.extend(standardDeviation)
                        meanAccel.append(x)
                    xAngle, yAngle, zAngle = 0, 0, 0
                    
        self.data = np.array(meanAccel)
        self.target = np.array(actualClass)


    def readTestData(self):

        meanAccel = []
        counter, rotationCounter = 0, 0
        previousTime = 0
        xAngle, yAngle, zAngle = 0, 0, 0
        
        with open(self.testAccelFilePath, 'rb') as csvfile, open(self.testRotationFilePath, 'rb') as csvfile2:
            accelReader = csv.reader(csvfile)
            rotationReader = csv.reader(csvfile2)
            rotationRow = rotationReader.next()
            
            for row in accelReader:
                if (counter == 0 or row[2] != previousTime) and row[2] != "":
                    previousTime = row[2]
                    counter += 1
                    
                    if counter == 1:
                        meanAccel.append([row[8], row[13], row[18]])
                        rotationRow = rotationReader.next()
                        meanAccel.pop(0)
                        
                    while rotationRow[2] <= previousTime and counter != 1:
                        #collect aggregated angle data
                        if rotationRow[5] != "":
                            xAngle += float(rotationRow[5])
                            yAngle += float(rotationRow[6])
                            zAngle += float(rotationRow[4])
                            rotationCounter += 1
                        try:
                            rotationRow = rotationReader.next()
                        except StopIteration:
                            break
                        
                    if rotationCounter != 0:
                        xAngle /= rotationCounter
                        yAngle /= rotationCounter
                        zAngle /= rotationCounter
                    rotationCounter = 0
                    
                    if (counter > 1):
                        x = [float(row[8]), float(row[13]), float(row[18])]
                        print x
                        standardDeviation = [float(row[10]), float(row[15]), float(row[20])]
                        temp = self.rotateAccelerationVector(xAngle, yAngle, zAngle, x)
                        x = temp.tolist()
                        x.extend(standardDeviation)
                        meanAccel.append(x)
                    xAngle, yAngle, zAngle = 0, 0, 0
                    
        self.testData = np.array(meanAccel)

    def rotateAccelerationVector(self, xAngle, yAngle, zAngle, accelerationData):
        # 1. Calculate desired x, y, z angular rotation from orientation data
        # 2. Create numpy rotation matrix
        # 3. Convert acceleration vector from list to numpy array
        # 4. Multiply rotation matrix by acceleration vector
        # 5. Return new acceleration vector

        targetX = 2
        targetY = 5
        targetZ = 199

        xRotation = np.deg2rad(targetX - xAngle)
        yRotation = np.deg2rad(targetY - yAngle)
        zRotation = np.deg2rad(targetZ - zAngle)

        rotationMatrix = np.array([[np.cos(zRotation)*np.cos(yRotation),
                                     np.cos(zRotation)*np.sin(yRotation)*np.sin(xRotation) - np.sin(zRotation)*np.cos(xRotation),
                                     np.cos(zRotation)*np.sin(yRotation)*np.cos(xRotation) + np.sin(zRotation)*np.sin(xRotation)],
                                    [np.sin(zRotation)*np.cos(yRotation),
                                     np.sin(zRotation)*np.sin(yRotation)*np.sin(xRotation) + np.cos(zRotation)*np.cos(xRotation),
                                     np.sin(zRotation)*np.sin(yRotation)*np.cos(xRotation) - np.cos(zRotation)*np.sin(xRotation)],
                                    [-1*np.sin(yRotation),
                                     np.cos(yRotation)*np.sin(xRotation),
                                     np.cos(yRotation)*np.cos(xRotation)]])
        
        accelerationVector = np.array(accelerationData)
        accelerationVector = rotationMatrix.dot(accelerationVector)
        return accelerationVector


    def runClassificationTrain(self, classifierType):

        predictionValues = 0

        if (classifierType == "SVM"):

            svmClassifier = svm.SVC(gamma=0.001, C=100.)
            svmClassifier.fit(self.data, self.target)

            predictionValues = svmClassifier.predict(self.data)

        elif (classifierType == "KNN"):

            knnClassifier = neighbors.KNeighborsClassifier(5, 'distance')
            knnClassifier.fit(self.data, self.target)

            predictionValues = knnClassifier.predict(self.data)

        elif (classifierType == "Decision Tree"):

            DTreeClassifier = tree.DecisionTreeClassifier()
            DTreeClassifier.fit(self.data, self.target)

            predictionValues = DTreeClassifier.predict(self.data)

        elif (classifierType == "Naive Bayes"):

            NaiveBayesClassifier = GaussianNB()
            NaiveBayesClassifier.fit(self.data, self.target)

            predictionValues = NaiveBayesClassifier.predict(self.data)

        elif (classifierType == "Random Forest"):

            RForestClassifier = RandomForestClassifier(n_estimators=10)
            RForestClassifier.fit(self.data, self.target)

            predictionValues = RForestClassifier.predict(self.data)

        return predictionValues

    
    def runClassificationTest(self, classifierType):

        predictionValues = 0

        if (classifierType == "SVM"):

            svmClassifier = svm.SVC(gamma=0.001, C=100.)
            svmClassifier.fit(self.data, self.target)

            predictionValues = svmClassifier.predict(self.testData)

        elif (classifierType == "KNN"):

            knnClassifier = neighbors.KNeighborsClassifier(5, 'distance')
            knnClassifier.fit(self.data, self.target)

            predictionValues = knnClassifier.predict(self.testData)

        elif (classifierType == "Decision Tree"):

            DTreeClassifier = tree.DecisionTreeClassifier()
            DTreeClassifier.fit(self.data, self.target)

            predictionValues = DTreeClassifier.predict(self.testData)

        elif (classifierType == "Naive Bayes"):

            NaiveBayesClassifier = GaussianNB()
            NaiveBayesClassifier.fit(self.data, self.target)

            predictionValues = NaiveBayesClassifier.predict(self.testData)

        elif (classifierType == "Random Forest"):

            RForestClassifier = RandomForestClassifier(n_estimators=10)
            RForestClassifier.fit(self.data, self.target)

            predictionValues = RForestClassifier.predict(self.testData)

        return predictionValues

    def crossValidate(self, classifierType):

        clf = None
        
        if (classifierType == "SVM"):
            clf = svm.SVC(gamma=0.001, C=100.)
        elif (classifierType == "KNN"):
            clf = neighbors.KNeighborsClassifier(5, 'distance')
        elif (classifierType == "Decision Tree"):
            clf = tree.DecisionTreeClassifier()
        elif (classifierType == "Naive Bayes"):
            clf = GaussianNB()
        elif (classifierType == "Random Forest"):
            clf = RandomForestClassifier(n_estimators=10)
            
        n_samples = self.data.shape[0]
        crossType = cross_validation.KFold(len(self.target), n_folds=10,
                                           indices=False)
        scores = cross_validation.cross_val_score(clf, self.data,
                                                 self.target, cv=crossType)
        return "10-fold cross validation accuracy: %0.2f" % scores.mean()

def main():

    app = QtGui.QApplication(sys.argv)
    instance = UserInterface()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
