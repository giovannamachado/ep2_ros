from dataclasses import dataclass

import cv2 as cv
import mediapipe as mp
import numpy as np
from math import dist as pDist

@dataclass
class handDist:
    x:float
    y:float
    closed:bool
    def __str__(self):
        return f"handDist(x:{self.x*100:.1f}%,y:{self.y*100:.1f}%,closed:{self.closed})"
    
class handDetection:
    def __init__(self,#varios valores padrão
                 dead_zone_size= (160,90),#limites da zona morta, pode ser int caso o ela seja quadrada, tuple(int,int) para retangulos
                 frame_width = 1900,frame_height = 1900,#resolução desejada (no coumputador testado ele transforma em 720x1280)
                 task_path = "GIT/ep2_ros/mediapipe/files/hand_landmarker.task",#caminho para o arquivo tsak do mediapipe
                 confidence={"detection":0.5,"presence":0.5,"traking":0.5},#variaveis de confiança do modelo do mediapipe
                 limit= -200):#Quão fora do quadro o centro da mão deve estar para ser desconsiderado
        self.limit = limit if limit<0 else -limit
        self.dzone = (dead_zone_size[0]/2,dead_zone_size[1]/2) if isinstance(dead_zone_size,tuple) else (dead_zone_size/2,dead_zone_size/2)
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
        _, self.frame = self.cap.read()
        y,x = self.frame.shape[:2]
        self.rez = (x,y)  
        self.center = (int(x/2),int(y/2))
        self.hand_center = (-x,-y)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=task_path),
            min_hand_presence_confidence = confidence["presence"],
            min_hand_detection_confidence= confidence["detection"],  
            min_tracking_confidence= confidence["traking"], 
            running_mode=mp.tasks.vision.RunningMode.IMAGE, 
            num_hands=1, )

        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
    def _hand_state(self):
        pass
    def _hand_closed(self):#detecta se a mão aparenta estar fechada
        count = 0
        point0 = self.hand_points[0]
        for i in [4*j for j in range(1,6)]:
            point1 = pDist(self.hand_points[i],point0)
            point2 = pDist(self.hand_points[i-1],point0)
            print(f"\t\tp{i:>2} {point1:.2f}/ p{i-1:<2} {point2:.2f} : {point1<point2}")
            if point1<point2:#se a ponta do dedo não for o ponto mais distante do ponto 0 da mão ele é considerado fechado
                count+=1
        return count>=4#conta pelo menos 4 dedos para considerar fechada
    def _dist_center(self,p,m,z):
        dist = m/2-z
        if p<self.limit or p>m-self.limit: return 0
        if p>(z+m/2): return (p-(z+m/2))/dist
        elif p<dist: return (p-dist)/dist
        else: return 0
    @property
    def handDist(self):
        px,py = (float(self.hand_center[0]),float(self.hand_center[1]))
        x,y = self.rez
        x = self._dist_center(px,x,self.dzone[0])
        y = -self._dist_center(py,y,self.dzone[1])
        closed = self._hand_closed()
        return handDist(x,y,closed=closed)#(x,y,closed)
    
    def drawnDeadZone(self):
        zx = int(self.dzone[0])
        zy = int(self.dzone[1])
        (cx,cy) = self.center
        r1  =(cx+zx,zy+cy)
        r2  =(cx-zx,cy-zy)
        cv.circle(self.frame,(cx,cy),3,color=(255,255,255),thickness=-1)
        cv.circle(self.frame,(cx,cy),2,color=(0,0,0),thickness=-1)
        cv.rectangle(self.frame,pt1=r1,pt2=r2,color=(0,0,0),thickness=3)

    def drawhandBox(self,box,cat,id): 
        self.drawnDeadZone()
        (x,y) =(int(self.hand_center[0]),int(self.hand_center[1]))
        cv.drawContours(self.frame,[box],contourIdx=0,color=(255,0,0),thickness=2)
        cv.circle(self.frame,(x,y),2,color=(0,0,255),thickness=-1)
        cv.putText(self.frame,f"{cat}",(x,y-10),cv.FONT_HERSHEY_PLAIN,1,(0,255,255))
        cv.putText(self.frame,f"{(x,y)}",(x,y+10),cv.FONT_HERSHEY_PLAIN,1,(0,255,255))
        print(f"Hand {id}\n\tSide:{cat}\n\tCenter:x-{x}, y-{y}\n\tDist:{self.handDist}")
    
    def drawHandLines(self):
        duos = [(0,1),(0,5),(0,17),(5,9),(9,13),(13,17)]
        duos +=[(v+i-1,v+i) for v in set([vl[1] for vl in duos]) for i in range(1,4) if v!=0]+[(2,5)]
        for a,b in duos:
            self.frame = cv.line(self.frame,self.hand_points[a],self.hand_points[b],color=(0,255,0))
        for p in self.hand_points:
            self.frame = cv.putText(self.frame,f"{self.hand_points.index(p)}",p,cv.FONT_HERSHEY_PLAIN,1,(255,255,0))
            self.frame = cv.circle(self.frame,p,2,color=(255,0,255),thickness=-1)
    
    def getMiddleAndBox(self):
        r = cv.minAreaRect(np.array([self.hand_points]))
        return r[0],cv.boxPoints(r)  #pixel Position  
        
    def main(self):

        ret, frame = self.cap.read()
        #Troca mão no video espelhado
        handSwitch = {0:'Left',1:'Right'}
        detector = self.createDetector()
        x,y = self.rez
        print(self.rez)
        #limit = 15
        #c = 0
        while True:
            ret, frame = self.cap.read()
            self.frame = cv.flip(frame,1)
            if not ret: continue
            frame_RGB = mp.Image(mp.ImageFormat.SRGB,cv.cvtColor(self.frame,cv.COLOR_BGR2RGB))
            r = detector.detect(frame_RGB)
            size = len(r.hand_landmarks)
            if size>0:
                hand = r.hand_landmarks[0]
                h = r.handedness[0][0]
                self.hand_points = [(int(l.x*x),int(l.y*y)) for l in hand]
                self.hand_center,box = self.getMiddleAndBox()
                self.drawHandLines()
                self.drawhandBox(box.astype(np.int64),handSwitch[h.index],"Main Hand Stats:")
            else: self.drawnDeadZone()
            cv.imshow('Webcam', self.frame)
            #if c == limit: c=0
            if cv.waitKey(1) & 0xFF == ord('q'): break
            #c+=1
        self.cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    d = {"detection":0.4,"presence":0.4,"traking":0.6}
    det = handDetection()
    det.main()