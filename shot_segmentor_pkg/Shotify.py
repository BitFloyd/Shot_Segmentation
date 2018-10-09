import cv2
import numpy as np
from logging_pkg.logging import message_print, debug_print
import os
import math
import matplotlib.pyplot as plt



class VideoLengthAssertionError(Exception):
    pass

class SlidingWindowLengthEvenError(Exception):
    pass

class VideoToShotConverter:

    def __init__(self,pathToVideo,pathToShots,slidingWindowLength=None):


        self.pathToVideo = pathToVideo
        self.pathToShots = pathToShots
        self.videoContainer = cv2.VideoCapture(self.pathToVideo)
        self.videoFPS = self.videoContainer.get(cv2.CAP_PROP_FPS)
        message_print("VIDEO FPS:"+str(self.videoFPS))
        self.numFrames = int(self.videoContainer.get(cv2.CAP_PROP_FRAME_COUNT))

        self.videoFrameWidth = int(self.videoContainer.get(3))
        self.videoFrameHeight = int(self.videoContainer.get(4))

        if(slidingWindowLength==None):
            if((self.videoFPS)%2==0):
                self.slidingWindowLength = int(self.videoFPS+1)
            else:
                self.slidingWindowLength = int(self.videoFPS)
        else:
            self.slidingWindowLength = slidingWindowLength

        if(self.slidingWindowLength%2==0):
            raise SlidingWindowLengthEvenError("Sliding window length must be odd")


        self.indexToCheck = (self.slidingWindowLength-1)/2
        self.listOfCurrentFrames = []
        self.listOpticalFlowMagnitudes = []
        self.listOfFramesForCurrentShot = []
        self.videoFinished = False
        self.shotId = 0
        self.logFile = os.path.join(self.pathToShots,'logfile.txt')
        self.stdMultiplierForCheck = 3.0

        self.farnBackParams = {'flow':None, 'pyr_scale':0.5, 'levels':3, 'winsize':15,'iterations':3, 'poly_n':5, 'poly_sigma':1.2,'flags': 0}

    def writeOpticalFlowDetailsToFile(self,flow):

        with open(self.logFile,'a+') as f:
            f.write(str(flow)+'\n')

        return True

    def checkShotBoundaryInCurrentFrames(self):

        arrayOpticalFlowMagnitudes=np.array(self.listOpticalFlowMagnitudes)
        medianOpticalFlow = np.median(arrayOpticalFlowMagnitudes)
        stdOpticalFlow = np.std(arrayOpticalFlowMagnitudes)

        difference = np.abs(arrayOpticalFlowMagnitudes[self.indexToCheck-1]-medianOpticalFlow)

        if(np.sum(arrayOpticalFlowMagnitudes)>0):

            if(difference>=self.stdMultiplierForCheck*stdOpticalFlow):
                return True
            else:
                return False
        else:
            return False

    def saveShotFromFramesForCurrentShot(self):

        shotFileName = os.path.join(self.pathToShots,'shot_'+str(self.shotId)+'.mp4')


        out = cv2.VideoWriter(shotFileName, cv2.VideoWriter_fourcc('F', 'M', 'P', '4'),
                              self.videoFPS, (self.videoFrameWidth, self.videoFrameHeight))

        while(len(self.listOfFramesForCurrentShot)):
            out.write(self.listOfFramesForCurrentShot.pop(0))

        out.release()

        self.shotId+=1

        return True

    def saveShotFromListOfCurrentFrames(self):

        shotFileName = os.path.join(self.pathToShots, 'shot_' + str(self.shotId) + '.mp4')

        out = cv2.VideoWriter(shotFileName, cv2.VideoWriter_fourcc('F', 'M', 'P', '4'),
                              self.videoFPS, (self.videoFrameWidth, self.videoFrameHeight))

        while (len(self.listOfCurrentFrames)):
            out.write(self.listOfCurrentFrames.pop(0))

        out.release()

        self.shotId += 1

        return True

    def populateInitialListOfCurrentFrames(self):

        ret, frame = self.videoContainer.read()
        self.videoFinished = not ret

        while (not self.videoFinished and len(self.listOfCurrentFrames) < self.slidingWindowLength):
            self.listOfCurrentFrames.append(frame)
            ret, frame = self.videoContainer.read()
            self.videoFinished = not ret

        if(len(self.listOfCurrentFrames) < self.slidingWindowLength):
            raise VideoLengthAssertionError("Video length is lower than sliding window length")

        if(self.videoFinished):
            message_print("VIDEO FINISHED")

        return True

    def prepFramesForOpticalFlows(self,f1,f2):

        f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

        f1 = cv2.resize(f1, (0,0), fx=0.25, fy=0.25)
        f2 = cv2.resize(f2, (0,0), fx=0.25, fy=0.25)

        f1 = cv2.GaussianBlur(f1,ksize=(5,5),sigmaX=1.0,sigmaY=0)
        f2 = cv2.GaussianBlur(f2,ksize=(5,5),sigmaX=1.0,sigmaY=0)

        return f1,f2

    def getOpticalFlow(self,f1,f2):

        f1,f2 = self.prepFramesForOpticalFlows(f1,f2)
        flow = cv2.calcOpticalFlowFarneback(prev=f1, next=f2, **self.farnBackParams)

        mag,_ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        mag = cv2.normalize(mag,alpha=0,beta=1.0,norm_type=cv2.NORM_MINMAX)

        mag = np.mean(mag)

        return mag

    def populateListOfOpticalFlows(self):

        self.listOpticalFlowMagnitudes=[]
        for i in range(0, len(self.listOfCurrentFrames) - 1):

            mag = self.getOpticalFlow(self.listOfCurrentFrames[i],self.listOfCurrentFrames[i + 1])

            self.listOpticalFlowMagnitudes.append(mag)
            self.writeOpticalFlowDetailsToFile(mag)

    def updateOpticalFlows(self):

        self.listOpticalFlowMagnitudes.pop(0)

        mag = self.getOpticalFlow(self.listOfCurrentFrames[-2],self.listOfCurrentFrames[-1])
        self.listOpticalFlowMagnitudes.append(mag)
        self.writeOpticalFlowDetailsToFile(mag)

        return True

    def performShotBoundaryRoutine(self):

        for i in range(0, self.indexToCheck + 1):
            self.listOfFramesForCurrentShot.append(self.listOfCurrentFrames.pop(0))

        self.saveShotFromFramesForCurrentShot()

        for i in range(0, self.indexToCheck + 1):
            ret, frame = self.videoContainer.read()
            self.videoFinished = not ret
            if (self.videoFinished):
                self.saveShotFromListOfCurrentFrames()
                break

            self.listOfCurrentFrames.append(frame)

        self.populateListOfOpticalFlows()

    def performNoShotBoundaryRoutine(self):

        self.listOfFramesForCurrentShot.append(self.listOfCurrentFrames.pop(0))
        ret, frame = self.videoContainer.read()
        self.videoFinished = not ret
        if(self.videoFinished):
            return True

        self.listOfCurrentFrames.append(frame)
        self.updateOpticalFlows()


        return True

    def segmentVideoToShots(self):

        if(os.path.exists(self.logFile)):
            os.remove(self.logFile)

        self.listOfCurrentFrames = []
        self.listOfFramesForCurrentShot = []


        self.populateInitialListOfCurrentFrames()
        self.populateListOfOpticalFlows()

        i = 0

        while (not self.videoFinished):

            shot_boundary_detected = self.checkShotBoundaryInCurrentFrames()

            if(shot_boundary_detected):
                self.performShotBoundaryRoutine()

            else:
                self.performNoShotBoundaryRoutine()

            print str(i)+'/'+str(self.numFrames)
            i+=1

        return True

    def __del__(self):
        self.videoContainer.release()


class PlotShotSegmentationParams:


    def __init__(self,vtscObject):

        self.logFile=vtscObject.logFile
        self.pathToPlot = vtscObject.pathToShots
        self.slidingWindowLength = vtscObject.slidingWindowLength
        self.indexToCheck = vtscObject.indexToCheck
        self.stdMultiplierForCheck = vtscObject.stdMultiplierForCheck

    def getOpticalFlowListFromFile(self):

        with open(self.logFile,'r') as f:
            strOfList = f.readlines()

        strOfList = [x.strip() for x in strOfList]

        ofList = [float(x) for x in strOfList ]

        return ofList

    def getSlopesOfOpticalFlow(self):

        ofList = self.getOpticalFlowListFromFile()

        slopeList = []

        for i in range(0,len(ofList)-1):

            slope = int(math.degrees(math.atan(ofList[i+1]-ofList[i])))
            slopeList.append(slope)

        return slopeList

    def processOpticalFlowsForShotBoundaryDetection(self):

        ofList = self.getOpticalFlowListFromFile()

        i = 0

        shotBoundaryTruthList = []
        differenceStdOpticalFlowRatioList = []

        while i < (self.indexToCheck-1):
            shotBoundaryTruthList.append(0)
            differenceStdOpticalFlowRatioList.append(0)
            i+=1

        j = 0
        while j+self.slidingWindowLength < len(ofList):

            arrayOpticalFlowMagnitudes = np.array(ofList[j:j+self.slidingWindowLength])
            medianOpticalFlow = np.median(arrayOpticalFlowMagnitudes)
            stdOpticalFlow = np.std(arrayOpticalFlowMagnitudes)

            difference = np.abs(arrayOpticalFlowMagnitudes[self.indexToCheck - 1] - medianOpticalFlow)

            if (np.sum(arrayOpticalFlowMagnitudes) > 0):

                if (difference >= self.stdMultiplierForCheck * stdOpticalFlow):
                    shotBoundaryTruthList.append(1)
                else:
                    shotBoundaryTruthList.append(0)

                differenceStdOpticalFlowRatioList.append(difference/stdOpticalFlow)

            else:
                shotBoundaryTruthList.append(0)
                differenceStdOpticalFlowRatioList.append(0)


            j+=1
            i+=1


        while i < len(ofList):
            shotBoundaryTruthList.append(0)
            i+=1


        return shotBoundaryTruthList, differenceStdOpticalFlowRatioList


    def plotOF(self):

        opticalFlowList = self.getOpticalFlowListFromFile()
        shotBoundaryTruthList,_ = self.processOpticalFlowsForShotBoundaryDetection()

        colors = ['green','red']

        shotBoundaryTruthList = shotBoundaryTruthList[0:len(opticalFlowList)]
        colors_points = [colors[x] for x in shotBoundaryTruthList]

        plt.figure(figsize=(100,20))
        plt.plot(opticalFlowList,'+--')
        plt.scatter(x=range(0, len(opticalFlowList)), y=opticalFlowList, c=colors_points)
        plt.xlabel('Frame Index')
        plt.ylabel('Optical Flow Value')
        plt.savefig(os.path.join(self.pathToPlot,'optical_flows_per_frame.png'))
        plt.close()

        return True

    def plotSlopes(self):

        slopeList = self.getSlopesOfOpticalFlow()
        shotBoundaryTruthList,_ = self.processOpticalFlowsForShotBoundaryDetection()
        colors = ['green','red']

        shotBoundaryTruthList = shotBoundaryTruthList[0:len(slopeList)]
        colors_points = [colors[x] for x in shotBoundaryTruthList]

        plt.figure(figsize=(100,20))
        plt.plot(slopeList,'+--')
        plt.scatter(x=range(0, len(slopeList)), y=slopeList, c=colors_points)
        plt.xlabel('Frame Index')
        plt.ylabel('Slopes from Optical Flow Value')
        plt.savefig(os.path.join(self.pathToPlot,'slopes_per_frame.png'))
        plt.close()

        return True

    def plotRatios(self):

        shotBoundaryTruthList,differenceStdOpticalFlowRatioList = self.processOpticalFlowsForShotBoundaryDetection()
        colors = ['green','red']

        shotBoundaryTruthList = shotBoundaryTruthList[0:len(differenceStdOpticalFlowRatioList)]
        colors_points = [colors[x] for x in shotBoundaryTruthList]

        plt.figure(figsize=(100,20))
        plt.plot(differenceStdOpticalFlowRatioList,'+--')
        plt.scatter(x=range(0, len(differenceStdOpticalFlowRatioList)), y=differenceStdOpticalFlowRatioList, c=colors_points)
        plt.xlabel('Frame Index')
        plt.ylabel('Ratio of Difference to StdOpticalFlow')
        plt.savefig(os.path.join(self.pathToPlot,'ratios_per_frame.png'))
        plt.close()

        return True
