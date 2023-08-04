from pyexpat import model
import cv2
import numpy as np
import time


class Detector:
    
    def __init__(self, vidPath, configPath, modelPath, classesPath):
        self.vidPath = vidPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        self.network = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.network.setInputSize(320, 320)
        self.network.setInputScale(1.0/127.5)
        self.network.setInputMean((127.5, 127.5, 127.5))
        self.network.setInputSwapRB(True)

        self.readClasses()

    def readClasses(self):
        with open(self.classesPath, "r") as f:
            self.classesList = f.read().splitlines()

        self.classesList.insert(0, "__Background__")
        
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))
        
        #print(self.classesList)
    
    def onVideo(self):
        capture = cv2.VideoCapture(self.vidPath)

        if not capture.isOpened():
            print("Error loading file...")
            return
        
        success, image = capture.read()
        while success:
            classLabelIDs, confs, bboxs = self.network.detect(image, confThreshold = 0.5)

            bboxs = list(bboxs)
            confs = list(np.array(confs).reshape(1,-1)[0])
            confs = list(map(float, confs))

            bboxIds = cv2.dnn.NMSBoxes(bboxs, confs, score_threshold = 0.5, nms_threshold = 0.2)

            if len(bboxIds) != 0:
                for i in range(0, len(bboxIds)):

                    bbox = bboxs[np.squeeze(bboxIds[i])]
                    classConfidence = confs[np.squeeze(bboxIds[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIds[i])])
                    classLabel = self.classesList[classLabelID]
                    classColor = [int(c) for c in self.colorList[classLabelID]]

                    displayText = "{}:{:.4f}".format(classLabel, classConfidence)

                    x, y, w, h = bbox

                    cv2.rectangle(image, (x,y), (x+w, y+h), color=classColor, thickness=2)
                    cv2.putText(image, displayText, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

            cv2.imshow("Realtime Object Detection (Press Q to Quit)", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            success, image = capture.read()
        
        cv2.destroyAllWindows()
