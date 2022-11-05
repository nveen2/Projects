#Object Tracking using CamShift algorithm

import cv2
import numpy as np
class ObjectTracker:
    def __init__(self, source):
        #define a video frame reading stream between the application and the source
        self.v_handle = cv2.VideoCapture(source)
        #check for the open status of the video stream
        if not self.v_handle.isOpened():
            raise Exception('Video Stream Not Established')

        #determine the fps of the video
        self.fps = self.v_handle.get(cv2.CAP_PROP_FPS)

        #current frame
        self.frame = None

        #create a window to display the video frames
        cv2.namedWindow('Video')

        #register (with cv2) a mouse event callback method
        cv2.setMouseCallback('Video', self.mouse_events, param='Sample Data')

        #selection
        self.selection = [] #x1,y1,x2,y2
        self.selection_state = 0



    def __del__(self):
        #deallocate the stream (resource)
        self.v_handle.release()
        #destroy the created windows
        cv2.destroyAllWindows()

    def mouse_events(self, event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDOWN: #mouse left button down
            self.selection.clear() #ensure that the list is empty
            self.selection.append(x) #x1
            self.selection.append(y) #y1
            self.selection_state = 1
            print('LBD ', param)
        elif event == cv2.EVENT_LBUTTONUP: #mouse left button up
            self.selection.append(x) #x2
            self.selection.append(y) #y2
            self.selection_state = 2
            print('LBU ', param)

        if self.selection_state == 2:
            #irrespective of the users selection approach (top-left to bottom-right, top-right to bottom-left, ...), define the x1,y1,x2,y2 as l,t,b,r
            if self.selection[0] > self.selection[2]:
                temp = self.selection[0]
                self.selection[0] = self.selection[2]
                self.selection[2] = temp
            if self.selection[1] > self.selection[3]:
                temp = self.selection[1]
                self.selection[1] = self.selection[3]
                self.selection[3] = temp

            self.selection_state = 3

            #ensure that the selection is within the frame size
            if self.selection[0] < 0 or self.selection[1] < 0:
                self.selection_state = 0
            h,w,_ = self.frame.shape
            if self.selection[2] > w: #w = self.frame.shape[1]
                self.selection_state = 0
            if self.selection[3] > h: #h = self.frame.shape[0]
                self.selection_state = 0



    def track(self):
        #grab a frame from the video
        flag, self.frame = self.v_handle.read() #returns (boolean, ndarray)

        while flag:
            #the grabbed frame is in BGR
            #for processing we need it in HSV
            hsv_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            #a mask to ignore/discard the dim or weakly pronounced areas of the frame
            mask = cv2.inRange(hsv_frame, np.array((0,60,32)), np.array((180,255,255)) )

            if self.selection_state == 3:
                #user has a selection (l,t,w,h)
                #grab the roi portion from the frame by slicing the ndarray
                roi_hsv_frame = hsv_frame[self.selection[1]:self.selection[3], self.selection[0]:self.selection[2]] #roi = frame[y1:y2, x1:x2]

                #know how colors are distributed over the roi
                hist_roi_hsv_frame = cv2.calcHist([roi_hsv_frame],[0],None,[180],[0,180])

                #define the track window
                track_window = []#l,t,w,h
                track_window.append(self.selection[0]) #l = x1
                track_window.append(self.selection[1]) #t = y1
                track_window.append(self.selection[2] - self.selection[0]) #w = x2 - x1
                track_window.append(self.selection[3] - self.selection[1]) #h = y2 - y1


                #termination criteria tuple (flag, iterations, movement) for the algorithm
                #algorithm (CamSHIFT) terminate either on 10 iterations or movement of 1 point
                term_criteria = (cv2.TermCriteria_COUNT | cv2.TERM_CRITERIA_EPS , 10 , 1)

                if 0 not in track_window:
                    #progress to next stage
                    self.selection_state = 4


            if self.selection_state == 4:
                #lets backproject (locate) the histogram over the  frame (image)
                back_projection = cv2.calcBackProject([hsv_frame],[0], hist_roi_hsv_frame,[0,180],1)
                back_projection &= mask

                #track now
                track_points, track_window = cv2.CamShift(back_projection, track_window, term_criteria)
                #print(track_points)

                #rendering the tracking
                #draw an ellipse using the track_points
                cv2.ellipse(self.frame, track_points, color=(0,0,255), thickness=1)

                cv2.imshow('ROI', roi_hsv_frame)
                cv2.imshow('BackProjection', back_projection)


            #render the grabbed frame
            cv2.imshow('Video', self.frame)


            #delay between 2 frames
            if cv2.waitKey(int(1/self.fps*1000)) == 27: #27 ASCII of Esc
                break

            #read the next frame
            flag, self.frame = self.v_handle.read()

def main():
    ot = ObjectTracker('/Users/naveenjhajhriya/Documents/4 Projects/Project - Object Tracking/a.mp4')
    ot.track()

main()
