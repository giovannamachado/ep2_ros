from dataclasses import dataclass

import cv2 as cv
import mediapipe as mp
import numpy as np
#task_path = "C:\\Users\\ltf\\Documents\\dev\\GIT\\workshop-python\\docs\\modulo_mediapipe\\hand_landmarker.task"
#task_path = "docs/modulo_mediapipe/hand_landmarker.task"

@dataclass
class handDetection:
    task_path :str = "mediapipe/files/hand_landmarker.task"

    def drawhandBox(self,frame,middle,box,cat,id): 
        (x,y) =(int(middle[0]),int(middle[1]))
        frame = cv.drawContours(frame,[box],contourIdx=0,color=(255,0,0),thickness=2)
        frame = cv.circle(frame,(x,y),2,color=(0,0,255),thickness=-1)
        cv.putText(frame,f"{cat}",(x,y-10),cv.FONT_HERSHEY_PLAIN,1,(0,255,255))
        cv.putText(frame,f"{(x,y)}",(x,y+10),cv.FONT_HERSHEY_PLAIN,1,(0,255,255))
        print(f"Hand {id}\n\tSide:{cat}\n\tCenter:x-{x}, y-{y}")
        return frame
    def drawHandLines(self,img,points):
        duos = [(0,1),(0,5),(0,17),(5,9),(9,13),(13,17)]
        duos +=[(v+i-1,v+i) for v in set([vl[1] for vl in duos]) for i in range(1,4) if v!=0]
        for a,b in duos:
            img = cv.line(img,points[a],points[b],color=(0,255,0))
        for p in points:
            img = cv.putText(img,f"{points.index(p)}",p,cv.FONT_HERSHEY_PLAIN,1,(255,255,0))
            img = cv.circle(img,p,2,color=(255,0,255),thickness=-1)
        return img
    def getMiddleAndBox(self,points):
        r = cv.minAreaRect(points)
        return r[0],cv.boxPoints(r)  #pixel Position
    def createDetector(self):
        
        options = mp.tasks.vision.HandLandmarkerOptions(min_hand_detection_confidence=0.4,  
            base_options=mp.tasks.BaseOptions(model_asset_path=self.task_path), min_hand_presence_confidence=0.4,  
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=2, min_tracking_confidence=0.4,)

        detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        return detector


    def main(self):
        cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        # cap.set(cv.CAP_PROP_FPS, 3)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1900)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1900)
        ret, frame = cap.read()
        y,x = frame.shape[:2]     
        center = (int(x/2),int(y/2))
        #Troca mão no video espelhado
        handSwitch = {0:'Left',1:'Right'}
        detector = self.createDetector()
        print(y,x)
        l = 15
        c = 0
        ordernames= {1:'st',2:'nd',3:'rd',10:"th"}
        while True:
            # Read frame
            ret, frame = cap.read()
            frame = cv.flip(frame,1)
            if not ret:
                continue
            frame_RGB = mp.Image(mp.ImageFormat.SRGB,cv.cvtColor(frame,cv.COLOR_BGR2RGB))
            r = detector.detect(frame_RGB)
            hands = {}
            size = len(r.hand_landmarks)
            for i in range(size):
                
                hand = r.hand_landmarks[i]
                h = r.handedness[i][0]

                hand_points = [(int(l.x*x),int(l.y*y)) for l in hand]
                middle,box = self.getMiddleAndBox(np.array([hand_points]))
                hands[i] = {"center":middle,"box":box.astype(np.int64),"side":handSwitch[h.index],"points":hand_points}
            hands_list = sorted(hands.values(),key = lambda item:item["center"][0])
            for i in range(size):
                hand = r.hand_landmarks[i]
                h = r.handedness[i][0]

                hand_points = [(int(l.x*x),int(l.y*y)) for l in hand]
                middle,box = self.getMiddleAndBox(np.array([hand_points]))
                frame = self.drawHandLines(frame,hand_points)
                frame = self.drawhandBox(frame,middle,box.astype(np.int64),
                                    handSwitch[h.index],f"{i+1}{"th" if i // 10 % 10 == 1 else ordernames.get((i+1)%10,"th")} closest hand to x")
            cv.imshow('Webcam', frame)
            if c == l:
                c=0
                '''for hand in r.hand_world_landmarks:
                    hand_points = [(int(l.x*x),int(l.y*y)) for l in hand]
                    for p in hand_points:
                        print(p)'''
                    #drawHandLines(frame,hand_lines)
                
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            c+=1
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    handDetection.main()