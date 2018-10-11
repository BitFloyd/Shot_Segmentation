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
        self.pathToAnalysis = os.path.join(pathToShots,'Analysis')
        self.pathToShotBoundaries = os.path.join(pathToShots,'ShotBoundaries')

        if(not os.path.exists(self.pathToAnalysis)):
            os.mkdir(self.pathToAnalysis)
        if (not os.path.exists(self.pathToShotBoundaries)):
            os.mkdir(self.pathToShotBoundaries)

        self.videoContainer = cv2.VideoCapture(self.pathToVideo)
        self.videoFPS = int(self.videoContainer.get(cv2.CAP_PROP_FPS))
        message_print("VIDEO FPS:"+str(self.videoFPS))
        self.numFrames = int(self.videoContainer.get(cv2.CAP_PROP_FRAME_COUNT))

        self.videoFrameWidth = int(self.videoContainer.get(3))
        self.videoFrameHeight = int(self.videoContainer.get(4))
        self.videoFPSRatioForWindow = 1.0

        if(slidingWindowLength==None):
            if(int(self.videoFPS*self.videoFPSRatioForWindow)%2==0):
                self.slidingWindowLength = int(self.videoFPS*self.videoFPSRatioForWindow)+1
            else:
                self.slidingWindowLength = int(self.videoFPS*self.videoFPSRatioForWindow)
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
        self.logFile = os.path.join(self.pathToAnalysis,'logfile.txt')
        self.logFileShotBoundary = os.path.join(self.pathToAnalysis,'shotBoundarylog.txt')
        self.stdMultiplierForCheck = 3.5

        self.farnBackParams = {'flow':None, 'pyr_scale':0.75, 'levels':5, 'winsize':15,'iterations':3, 'poly_n':7, 'poly_sigma':1.2,'flags': 0}
        self.frameResizeParams ={'fx':1.0,'fy':1.0}

        self.ofplotter = PlotOpticalFlow()

    def writeOpticalFlowDetailsToFile(self,flow):

        with open(self.logFile,'a+') as f:
            f.write(str(flow)+'\n')

        return True

    def writeShotBoundaryDetailsToFile(self,arrayOpticalFlowMagnitudes,medianOpticalFlow,stdOpticalFlow,difference):

        arrayOpticalFlowMagnitudes = np.around(arrayOpticalFlowMagnitudes,decimals=2)
        medianOpticalFlow = np.around(medianOpticalFlow,decimals=2)
        stdOpticalFlow = np.around(stdOpticalFlow,decimals=2)
        difference = np.around(difference,decimals=2)

        with open(self.logFileShotBoundary,'a+') as f:
            f.write('-------------------------------------'+'\n')
            f.write('Shot_ID:'+str(self.shotId)+'\n')
            f.write('ArrayOF:'+str(arrayOpticalFlowMagnitudes)+'\n')
            f.write('OFofSBFrame:'+str(arrayOpticalFlowMagnitudes[self.indexToCheck])+'\n')
            f.write('MedianOF:'+str(medianOpticalFlow)+'\n')
            f.write('StdOF:'+str(stdOpticalFlow)+'\n')
            f.write('Difference:'+str(difference)+'\n')
            f.write('Threshold:'+str(self.stdMultiplierForCheck*stdOpticalFlow)+'\n')
            f.write('-------------------------------------' + '\n')

        return  True

    def checkShotBoundaryInCurrentFrames(self):

        arrayOpticalFlowMagnitudes=np.array(self.listOpticalFlowMagnitudes)
        medianOpticalFlow = np.median(arrayOpticalFlowMagnitudes)
        stdOpticalFlow = np.std(arrayOpticalFlowMagnitudes)

        difference = np.abs(arrayOpticalFlowMagnitudes[self.indexToCheck]-medianOpticalFlow)
        threshold = self.stdMultiplierForCheck*stdOpticalFlow

        if(np.sum(arrayOpticalFlowMagnitudes)>0 and arrayOpticalFlowMagnitudes[self.indexToCheck]>0):

            if(difference>=threshold and threshold>=1.0):
                self.writeShotBoundaryDetailsToFile(arrayOpticalFlowMagnitudes,medianOpticalFlow,stdOpticalFlow,difference)
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

        f1 = cv2.resize(f1, (0,0), fx=self.frameResizeParams['fx'], fy=self.frameResizeParams['fy'])
        f2 = cv2.resize(f2, (0,0), fx=self.frameResizeParams['fx'], fy=self.frameResizeParams['fy'])

        f1 = cv2.GaussianBlur(f1,ksize=(5,5),sigmaX=1.0,sigmaY=0)
        f2 = cv2.GaussianBlur(f2,ksize=(5,5),sigmaX=1.0,sigmaY=0)

        return f1,f2

    def getOpticalFlow(self,f1,f2):

        f1,f2 = self.prepFramesForOpticalFlows(f1,f2)
        flow = cv2.calcOpticalFlowFarneback(prev=f1, next=f2, **self.farnBackParams)

        mag,_ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag = cv2.medianBlur(mag,ksize=3)
        mag = np.median(mag)

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

    def saveShotBoundaryOpticalFlows(self):

        fig, axes = plt.subplots(3, 2)
        plt.axis('off')
        f1 = self.listOfCurrentFrames[self.indexToCheck]
        f2 = self.listOfCurrentFrames[self.indexToCheck+1]

        axes[0, 0].imshow(cv2.cvtColor(f1, cv2.COLOR_BGR2RGB))
        axes[0, 0].axis('off')
        axes[0, 1].imshow(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB))
        axes[0, 1].axis('off')

        f1,f2 = self.prepFramesForOpticalFlows(f1,f2)
        flow = cv2.calcOpticalFlowFarneback(prev=f1, next=f2, **self.farnBackParams)
        axes[1, 0].imshow(self.ofplotter.drawFlow(f1,flow,self.farnBackParams['winsize']))
        axes[1, 0].axis('off')

        axes[1, 1].imshow(self.ofplotter.drawFlowHsv(flow))
        axes[1, 1].axis('off')

        self.ofplotter.plotFlowHist(flow,axes[2,0])

        filename = os.path.join(self.pathToShotBoundaries,str(self.shotId)+'_shot_boundary.png')
        plt.savefig(filename,bbox_inches='tight')
        plt.close()

        return True

    def performShotBoundaryRoutine(self):

        self.saveShotBoundaryOpticalFlows()

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
        if(os.path.exists(self.logFileShotBoundary)):
            os.remove(self.logFileShotBoundary)

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

    def plotOpticalFlowSamplingWindow(self):

        plotterObj = PlotOpticalFlowSamplingWindow(self.pathToVideo, self.pathToShots,self.farnBackParams,self.frameResizeParams)
        plotterObj.createSampling()

        return True

    def __del__(self):
        self.videoContainer.release()


class PlotShotSegmentationParams:


    def __init__(self,vtscObject):

        self.logFile=vtscObject.logFile
        self.pathToPlot = vtscObject.pathToAnalysis
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


class PlotOpticalFlow:

    def __init__(self):
        self.name = 'PlotOpticalFlow'

    def drawFlow(self,img, flow, step=16):

        h, w = img.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (x2, y2) in lines:
            radius = int(np.sqrt((x1-x2)**2+(y1-y2)**2))
            cv2.circle(vis, (x1, y1), radius, (0, 255, 0), -1)

        return vis

    def drawFlowHsv(self,flow):

        h, w = flow.shape[:2]
        fx, fy = flow[:, :, 0], flow[:, :, 1]
        v, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        ang += np.pi
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[..., 0] = np.uint8(ang * (180 / np.pi / 2))
        hsv[..., 1] = 255
        hsv[..., 2] = np.uint8(np.minimum(v * 4, 255))
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return rgb

    def plotFlowHist(self,flow,axes):

        h,w = flow.shape[:2]
        fx,fy = flow[:,:,0], flow[:,:,1]
        fx = fx/h
        fy = fy/w

        mag_flow = np.sqrt(fx*fx + fy*fy)/np.sqrt(2.0)

        bins = np.linspace(0.0,1.0,101)

        axes.hist(x=mag_flow.flatten(),bins=bins)
        axes.set_xticks(ticks=np.linspace(0.0,1.0,11))

        return True


class PlotOpticalFlowSamplingWindow:

    def __init__(self,pathToVideo, pathToShots,farnBackParams,frameResizeParams):

        self.pathToVideo = pathToVideo
        self.pathToShots = pathToShots
        self.farnBackParams = farnBackParams
        self.frameResizeParams = frameResizeParams
        self.frame = self.getFrame()
        self.frameCount = 0

    def getFrame(self):

        videoContainer = cv2.VideoCapture(self.pathToVideo)

        # numFrame = int(videoContainer.get(cv2.CAP_PROP_FRAME_COUNT)/10)
        _,frame = videoContainer.read()
        videoContainer.release()

        frame = cv2.resize(frame, (0, 0), fx=self.frameResizeParams['fx'], fy=self.frameResizeParams['fy'])

        return frame

    def returnPyramid(self):

        pyramidScale = self.farnBackParams['pyr_scale']
        levels = self.farnBackParams['levels']

        pyramidImages = []
        pyramidImages.append(self.frame)

        frameResized = self.frame.copy()
        debug_print(frameResized.shape)

        for i in range(0,levels-1):
            frameResized = cv2.resize(frameResized,(0,0),fx=pyramidScale,fy=pyramidScale)
            pyramidImages.append(frameResized)

        return pyramidImages

    def slidingWindow(self,image, stepSize, windowSize):
        # slide a window across the image
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    def openVideoStream(self):

        shotFileName = os.path.join(self.pathToShots, 'slidingWindow.mp4')

        self.videoWriter = cv2.VideoWriter(shotFileName, cv2.VideoWriter_fourcc('F', 'M', 'P', '4'),
                              12, (self.frame.shape[1], self.frame.shape[0]))

        return True

    def writeFrameToVideo(self,pyramidFrame):

        zeroImage = np.zeros(shape=self.frame.shape)
        zeroImage[0:pyramidFrame.shape[0],0:pyramidFrame.shape[1],:] = pyramidFrame
        zeroImage = np.uint8(zeroImage)

        self.videoWriter.write(zeroImage)
        self.frameCount+=1

        return True

    def closeVideoStream(self):
        self.videoWriter.release()

        return True

    def createSampling(self):

        winW = self.farnBackParams['winsize']
        winH = self.farnBackParams['winsize']

        self.openVideoStream()

        pyramidImages = self.returnPyramid()
        for image in pyramidImages:
            for (x, y, window) in self.slidingWindow(image, stepSize=winH, windowSize=(winH,winW)):
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue

                clone = image.copy()
                cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)

                self.writeFrameToVideo(clone)



        self.closeVideoStream()

        message_print("THE FRAME COUNT OF THE VIDEO:"+str(self.frameCount))