from pykinect2 import PyKinectV2

from pykinect2.PyKinectV2 import *

from pykinect2 import PyKinectRuntime

import ctypes

import cv2

import math

import os
os.environ['SDL_VIDEO_WINDOW_POS'] = str(450) + "," + str(0)
os.environ['SDL_WINDOW_CENTERED'] = '0'
#                    THIS IS GROUP ONE'S WORK. feel free to look at tho

import _ctypes

import pygame

import sys

import numpy as np

if sys.hexversion >= 0x03000000:

    import _thread as thread

else:

    import thread

# colors for drawing different bodies

GestureFrameCount = [0]*100
Hands = list()
PanVector = [80,80]
PanOffsetVector = [0,0,0]
PanChangeVector = [0,0]
global PanInProgress
PanInProgress = 0
InitialZoomLength = 0
ZoomMultiplier = 1
global CumulativeScale
CumulativeScale = 0.4
global FinalScale
FinalScale = CumulativeScale * ZoomMultiplier

SKELETON_COLORS = [pygame.color.THECOLORS["red"],

                   pygame.color.THECOLORS["blue"],

                   pygame.color.THECOLORS["green"],

                   pygame.color.THECOLORS["orange"],

                   pygame.color.THECOLORS["purple"],

                   pygame.color.THECOLORS["yellow"],

                   pygame.color.THECOLORS["violet"]]


class GeneralRuntime(object):

    def __init__(self):

        pygame.init()

        # Used to manage how fast the screen updates

        self._clock = pygame.time.Clock()

        # Loop until the user clicks the close button.
        #screen = pygame.display.set_mode((600,600))
        self._done = False

        # Used to manage how fast the screen updates

        self._clock = pygame.time.Clock()

        # Kinect runtime object, we want Depth and RGB frames

        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)
        self._kinectRGB = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
        self._kinectIR = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Infrared)

        # back buffer surface for getting Kinect infrared frames, 8bit grey, width and height equal to the Kinect color frame size

        self._frame_surface = pygame.Surface( #Prepeare pygame surface to display DEPTH. (HOLDOVER)
            (self._kinect.depth_frame_desc.Width, self._kinect.depth_frame_desc.Height), 0, 24)

        self._frame_surfaceIR = pygame.Surface(  # Prepeare pygame surface to display DEPTH. (HOLDOVER)
            (self._kinectIR.infrared_frame_desc.Width, self._kinectIR.infrared_frame_desc.Height), 0, 24)

        self._frame_surfaceRGB = pygame.Surface(  # Prepeare pygame surface to display COLOR. (HOLDOVER)
            (self._kinectRGB.color_frame_desc.Width, self._kinectRGB.color_frame_desc.Height), 0, 32)

        self._GElogo = pygame.image.load('GElogo.png')
        self._back = pygame.image.load('back.jpg')



        # Set the width and height of the screen [width, height]

        self._infoObject = pygame.display.Info()

        self._screen = pygame.display.set_mode(
            (1000, 1000),

            pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE, 32)

        pygame.display.set_caption("GameScreen")



    def draw_depth_frame(self, frame, target_surface):

        if frame is None:  # some usb hub do not provide the infrared image. it works with Kinect studio though

            return

        target_surface.lock()

        f8 = np.uint8(frame.clip(1, 4000) / 16.)

        frame8bit = np.dstack((f8, f8, f8))

        address = self._kinect.surface_as_array(target_surface.get_buffer())

        ctypes.memmove(address, frame8bit.ctypes.data, frame8bit.size)

        del address

        target_surface.unlock()

    def draw_infrared_frame(self, frame, target_surface):

        if frame is None:  # some usb hub do not provide the infrared image. it works with Kinect studio though

            return

        target_surface.lock()

        f8 = np.uint8(frame.clip(1, 4000) / 16.)

        frame8bit = np.dstack((f8, f8, f8))

        address = self._kinectIR.surface_as_array(target_surface.get_buffer())

        ctypes.memmove(address, frame8bit.ctypes.data, frame8bit.size)

        del address

        target_surface.unlock()

    def draw_color_frame(self, frame, target_surface):

        target_surface.lock()

        address = self._kinectRGB.surface_as_array(target_surface.get_buffer())

        ctypes.memmove(address, frame.ctypes.data, frame.size)

        del address

        target_surface.unlock()

    def run(self):

        # -------- Main Program Loop -----------
        cv2.namedWindow("Depth", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Detection", cv2.WINDOW_AUTOSIZE)
        #cv2.namedWindow("canny", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Masked Depth", cv2.WINDOW_AUTOSIZE)
        #cv2.namedWindow("color", cv2.WINDOW_NORMAL)
        #cv2.resizeWindow("color",(int(self._kinectRGB.color_frame_desc.Width/2.0), int(self._kinectRGB.color_frame_desc.Height/2.0)))
        while not self._done:

            # --- Main event loop

            for event in pygame.event.get():  # User did something

                if event.type == pygame.QUIT:  # If user clicked close

                    self._done = True  # Flag that we are done so we exit this loop

                elif event.type == pygame.VIDEORESIZE:  # window resized

                    self._screen = pygame.display.set_mode(event.dict['size'],

                                                           pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE, 32)

            # --- Getting frames and drawing
            global CumulativeScale
            global ZoomMultiplier

            if self._kinect.has_new_depth_frame():
                frame = self._kinect.get_last_depth_frame()
                self.draw_depth_frame(frame, self._frame_surface)

                #frameRGB = self._kinectRGB.get_last_color_frame()
                #self.draw_color_frame(frameRGB, self._frame_surfaceRGB)

                frameIR = self._kinectIR.get_last_infrared_frame()
                self.draw_infrared_frame(frameIR, self._frame_surfaceIR)


                #myframe = np.zeros((self._kinect.depth_frame_desc.Width, self._kinect.depth_frame_desc.Height)) # = pygame.PixelArray(self._frame_surface)
                #pygame.pixelcopy.surface_to_array(myframe,self._frame_surface)
                #rownumber = 0
                #columnnumber = 0
                myframe = np.transpose(pygame.surfarray.array3d(self._frame_surface),[1,0,2])
                #mycolor = np.transpose(pygame.surfarray.array3d(self._frame_surfaceRGB), [1, 0, 2])
                #mycolor = np.ndarray.astype(mycolor,np.uint8)
                myframe = myframe[:,:,0]
                myinfra = np.transpose(pygame.surfarray.array3d(self._frame_surfaceIR),[1,0,2])
                myother = np.zeros(myframe.shape)
                #cv2.fastNlMeansDenoising(myframe, myother,1)

                dst ,myother = cv2.threshold(myframe,52,255,cv2.THRESH_TOZERO_INV)

                kernel = np.ones((3,3),np.uint8)
                mydilate = cv2.dilate(myother, np.ones((1,1),np.uint8))
                myerode = cv2.erode(mydilate,kernel)
                #mycanny = cv2.Canny(myerode,50,100)
                dst,mymask = cv2.threshold(myerode,10,255,cv2.THRESH_BINARY)
                #cv2.imshow("Depth", myframe)

                #mydim = ((self._kinectRGB.color_frame_desc.Width, self._kinectRGB.color_frame_desc.Height))
                #mymaskres = cv2.resize(mymask,mydim, interpolation= cv2.INTER_NEAREST)
                #mymaskres = np.ndarray.astype(mymaskres,np.uint8)

                myinfra = cv2.bitwise_and(myinfra,myinfra,mask= mymask)
                myinfra = cv2.cvtColor(myinfra, cv2.COLOR_RGB2GRAY)
                dst, myinfra = cv2.threshold(myinfra,200,255,cv2.THRESH_BINARY)
                #myinfra = np.ndarray.astype(myinfra,np.bool_)
                something ,contours, hierarchy = cv2.findContours(myinfra,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                min_area = 2000
                winning_contours = []
                for i in range(len(contours)):
                    area = cv2.contourArea(contours[i])
                    if(area > min_area):
                        winning_contours.append(i)
                a = np.zeros((myinfra.shape))
                b = a[..., np.newaxis]
                myhandimg = np.concatenate((b,b,b),axis = 2)
                #print(myhandimg.shape)
                colours= [(255,0,0),(0,255,0),(0,0,255),(0,128,128),(128,128,0),(128,0,128)]
                centers= []

                persistanceThreshold = 50;
                #print(len(winning_contours))
                usedIDs = list()
                for i in range(len(winning_contours)):

                    # identify hand, have we seen this one before?
                    ident = 100
                    M = cv2.moments(contours[winning_contours[i]])
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    if len(Hands) != 0:

                        #search known hands for match
                        for index,k in enumerate(Hands):

                            centredist = math.hypot(cX - k[0], cY - k[1])
                            if centredist < persistanceThreshold:
                                ident = index
                                Hands[index] = (cX, cY)
                                usedIDs.append(ident)
                                break

                    #if no hands are found

                    if ident == 100:
                        #check for recycled ident spaces in list
                        for index,hand in enumerate(Hands):
                            if hand == (-1000,-1000):
                                ident = index
                                Hands[ident] = (cX,cY)
                                break
                        #if no recycled idents are found, add to list
                        if ident == 100:
                            ident = len(Hands)
                            Hands.append((cX,cY))
                        usedIDs.append(ident)



                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.drawContours(myhandimg,contours,winning_contours[i],color = colours[ident],thickness = -1,)
                    hull = cv2.convexHull(contours[winning_contours[i]], returnPoints=False)
                    defects = cv2.convexityDefects(contours[winning_contours[i]], hull)
                    fingers = list()

                    #calculate fingers
                    for j in range(defects.shape[0]):
                        s, e, f, d = defects[j, 0]
                        start = tuple(contours[winning_contours[i]][s][0])
                        end = tuple(contours[winning_contours[i]][e][0])
                        far = tuple(contours[winning_contours[i]][f][0])
                        cv2.line(myhandimg,start,end,[0,128,0],1)
                        dist = math.hypot(end[0] - start[0], end[1] - start[1])
                        angle = 2 * np.arctan(dist / (2 * (d / 256.0)))
                        if (angle < 110 * (math.pi / 180)) & (far[1] > int(cY*1.2)) & (dist > 5):
                            cv2.circle(myhandimg,far,5,[0,255,255],-1)
                            cv2.circle(myhandimg, start, 5, [255, 0, 255], -1)
                            cv2.circle(myhandimg, end, 5, [255, 0, 255], -1)
                            fingers.append(start)
                            fingers.append(end)


                    if (ident == 0) | ((ident == 1)&(usedIDs == [1])):
                        #do gestures -- ZOOM
                        if (len(fingers) == 2) & (GestureFrameCount[ident] > 3):
                            if fingers[0][1] > myinfra.shape[0]*0.42:
                                cv2.line(myhandimg, fingers[0], fingers[1], [0, 255, 0], 2)
                                size = math.hypot(fingers[0][0] - fingers[1][0], fingers[0][1] - fingers[1][1])
                                cv2.putText(myhandimg, 'Zoom', fingers[0], font, size/25, (128, 255, 128), 2, cv2.LINE_AA)
                                #actuallyZoom
                                if InitialZoomLength == 0:
                                    InitialZoomLength = size
                                else:
                                    ZoomMultiplier = math.sqrt(size/InitialZoomLength)

                            else:
                                cv2.line(myhandimg, fingers[0], fingers[1], [0, 0, 255], 2)
                                cv2.putText(myhandimg, 'Too Far From Screen', fingers[0], font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                                InitialZoomLength = 0

                                CumulativeScale = CumulativeScale * ZoomMultiplier
                                ZoomMultiplier =1

                        elif len(fingers) == 2:
                            GestureFrameCount[ident] = GestureFrameCount[ident] +1
                        else:
                            GestureFrameCount[ident] = 0
                            InitialZoomLength = 0

                            CumulativeScale = CumulativeScale * ZoomMultiplier
                            ZoomMultiplier = 1


                        # -- PAN
                        global PanInProgress
                        if (len(fingers) == 4) & (GestureFrameCount[ident+20] > 3):
                            #Get Finger Height
                            #First, select regons of images
                            miniHeight = myframe[fingers[2][1] - 5:fingers[2][1] + 5,fingers[2][0] - 5:fingers[2][0] + 5]  #fingers[2]
                            minimask = myinfra[fingers[2][1] - 5:fingers[2][1] + 5,fingers[2][0] - 5:fingers[2][0] + 5]

                            # Then mask height with masked (multiply)
                            miniHM = cv2.bitwise_and(miniHeight,miniHeight,mask = minimask)

                            #Then get average brighness, discarding all black
                            activePixels = 0
                            pixelSum = 0
                            for pixelrow in miniHM:
                                for pixel in pixelrow:
                                    if pixel != 0:
                                        pixelSum = pixelSum + pixel
                                        activePixels = activePixels + 1
                            fingerHeight = int(pixelSum/activePixels)

                            if fingers[2][1] > myinfra.shape[0]*0.42:
                                cv2.circle(myhandimg, fingers[2], 20, [0, 255, 0], 2)

                                cv2.putText(myhandimg, 'Pan', fingers[2], font, 1, (128, 255, 128), 2, cv2.LINE_AA)
                                #actually pan
                                if PanInProgress == 0:
                                    PanInProgress = 1
                                    PanOffsetVector = (fingers[2][0],fingers[2][1],fingerHeight)
                                else:
                                    cv2.line(myhandimg, fingers[2], (PanOffsetVector[0],PanOffsetVector[1]), [128, 255, 128], 2)
                                    PanChangeVector[0] = 8.5*(fingers[2][0] - PanOffsetVector[0])
                                    PanChangeVector[1] = 65*(fingerHeight - PanOffsetVector[2])

                            else:
                                cv2.circle(myhandimg, fingers[2], 15, [0, 0, 255], 2)
                                cv2.putText(myhandimg, 'Too Far From Screen', fingers[2], font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                                PanInProgress = 0

                                PanVector[0] = PanVector[0] + PanChangeVector[0]
                                PanChangeVector[0] = 0

                                PanVector[1] = PanVector[1] + PanChangeVector[1]
                                PanChangeVector[1] = 0

                        elif len(fingers) == 4:
                            GestureFrameCount[ident+20] = GestureFrameCount[ident+20] +1
                        else:
                            GestureFrameCount[ident+20] = 0

                            PanInProgress = 0

                            PanVector[0] = PanVector[0] + PanChangeVector[0]
                            PanChangeVector[0] = 0

                            PanVector[1] = PanVector[1] + PanChangeVector[1]
                            PanChangeVector[1] = 0

                    cv2.putText(myhandimg, ("ID: " + str(ident)), (cX,cY), font, 0.5, (0, 255, 255), 2, cv2.LINE_AA)


                #if any IDs were not found, Recycle them.
                for index,hand in enumerate(Hands):
                    if (index in usedIDs) == False:
                        Hands[index] = ((-1000,-1000)) #impossible value signifies deletion without upsetting list order/length


                myscreenmask = np.ones(myinfra.shape)
                blackbit = np.zeros([100,myinfra.shape[1]])
                myscreenmask[myinfra.shape[0]-100:myinfra.shape[0],:] = blackbit
                myinfra = np.multiply(myinfra,myscreenmask)
                cv2.imshow("Detection", myhandimg)
                cv2.moveWindow("Detection",1400,0)
                cv2.imshow("Depth", myframe)
                cv2.moveWindow("Depth", 0, 0)
                cv2.imshow("Masked Depth", myinfra)
                cv2.moveWindow("Masked Depth", 0, 420)

                #MyImage = self._kinect.get_last_depth_frame()
                #frame = None
            global FinalScale
            OldFinalScale = FinalScale
            FinalScale = max(min(CumulativeScale*ZoomMultiplier,2),0.1)
            scaledWidth = int(self._GElogo.get_width() * (FinalScale))
            scaledHeight = int(self._GElogo.get_height() * (FinalScale))
            oldScaledWidth = int(self._GElogo.get_width() * (OldFinalScale))
            oldScaledHeight = int(self._GElogo.get_height() * (OldFinalScale))

            currentCentre = (PanVector[0] + oldScaledWidth/2,PanVector[1] + oldScaledHeight/2)
            newPosition = (currentCentre[0]-scaledWidth/2,currentCentre[1]-scaledHeight/2)
            PanVector[0] = (int(newPosition[0]))
            PanVector[1] = (int(newPosition[1]))

            scaledlogo = pygame.Surface((scaledWidth, scaledHeight))
            scaledlogo = pygame.transform.scale(self._GElogo,(scaledWidth,scaledHeight))

            #print(str(PanVector) + " " + str(PanChangeVector))
            self._screen.blit(self._back,(0,0))
            self._screen.blit(scaledlogo,(PanVector[0] + PanChangeVector[0],PanVector[1] + PanChangeVector[1]))

            #if self._kinectRGB.has_new_color_frame():



            pygame.display.update()


            # --- Go ahead and update the screen with what we've drawn.

            pygame.display.flip()

            # --- Limit to 60 frames per second

            self._clock.tick(60)

        # Close our Kinect sensor, close the window and quit.

        self._kinect.close()
        self._kinectRGB.close()

        pygame.quit()


__main__ = "Kinect v2 Depth"

game = GeneralRuntime();

game.run();