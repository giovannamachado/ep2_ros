from dataclasses import dataclass

import cv2 as cv
import mediapipe as mp
import numpy as np
from math import dist as pDist

@dataclass
class handDist:#classe que guarda as informações
    x:float
    y:float
    closed_fingers: dict
    closed:bool
    def __str__(self):
        #return f"handDist(x:{self.x*100:.1f}%,y:{self.y*100:.1f}%,closed:{self.closed})"
        cs = [f"Finger {k}: [{",".join(f"{v if not isinstance(v,float) else v:.1f}" for v in vl)}]" for k,vl in self.closed_fingers.items()]
        return f"handDist(x:{self.x*100:.1f}%,y:{self.y*100:.1f}%,closed:{self.closed}){"".join(f"\n\t\t{v}" for v in cs)}"
    
class handDetection:
    def __init__(self,#varios valores padrão
                 dead_zone_size= (160,90),#limites da zona morta, pode ser int caso o ela seja quadrada, tuple(int,int) para retangulos
                 max_zone_size= (160,90),#limites da zona maxima, usa a distancia para borda ao invez do seu tamanho
                 frame_width = 1900,frame_height = 1900,#resolução desejada (no coumputador testado ele transforma em 720x1280)
                 task_path = "GIT/ep2_ros/mediapipe/files/hand_landmarker.task",#caminho para o arquivo tsak do mediapipe
                 confidence={"detection":0.5,"presence":0.5,"traking":0.5},#variaveis de confiança do modelo do mediapipe
                 limit= -200,#Quão fora do quadro o centro da mão deve estar para ser desconsiderado
                 cross_mode = False):#O modo de exibição das zonas da imagem
        self.limit = limit if limit<0 else -limit
        self.cross_mode = cross_mode
        self.mzone = (max_zone_size[0]/2,max_zone_size[1]/2) if isinstance(max_zone_size,tuple) else (max_zone_size/2,max_zone_size/2)
        self.dzone = (dead_zone_size[0]/2,dead_zone_size[1]/2) if isinstance(dead_zone_size,tuple) else (dead_zone_size/2,dead_zone_size/2)
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        # self.cap = cv.VideoCapture(0,cv.CAP_V4L2)
        # self.cap.set(cv.CAP_PROP_FOURCC,cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
        _, self.frame = self.cap.read()
        y,x = self.frame.shape[:2]
        self.rez = (x,y) # resolução da captura
        self.center = (int(x/2),int(y/2)) # centro da captura
        self.hand_center = (-x,-y) #centro da mão
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
        fingers = {}
        for i in [4*j for j in range(1,6)]:
            point1 = pDist(self.hand_points[i],point0)
            point2 = pDist(self.hand_points[i-1],point0)
            #print(f"\t\tp{i:>2} {point1:.2f}/ p{i-1:<2} {point2:.2f} : {point1<point2}")
            fingers[i] = [point1<point2,point1,point2]
            if point1<point2:#se a ponta do dedo não for o ponto mais distante do ponto 0 da mão ele é considerado fechado
                count+=1
        return count>=4,fingers#conta pelo menos 4 dedos para considerar fechada
    
    def _dist_center(self,p,i):#calcula a distancia da zona morta para o centro da mão
        d_zone = self.dzone[i]
        side = self.rez[i]
        m_zone =self.mzone[i]
        dist = side/2-d_zone-m_zone

        if p<self.limit or p>side-self.limit: return 0

        elif p>(d_zone+side/2): 
            return min((p-(d_zone+side/2))/dist,1.0)
        elif p<dist: 
            return max((p-(dist+m_zone))/dist,-1.0)
        else: return 0

    @property
    def handDist(self):
        px,py = (float(self.hand_center[0]),float(self.hand_center[1]))
        x = self._dist_center(px,0)
        y = -self._dist_center(py,1)
        closed,fing = self._hand_closed()
        return handDist(x,y,closed=closed,closed_fingers=fing)#(x,y,closed)
    
    def __drawnZones(self):# desenha a zona morta e zona maxima
        zx = int(self.dzone[0])
        zy = int(self.dzone[1])
        (cx,cy) = self.center#centro da tela
        rx,ry = self.rez
        mx,my = self.mzone
        
        cv.circle(self.frame,(cx,cy),3,color=(255,255,255),thickness=-1)#ponto central da tela
        cv.circle(self.frame,(cx,cy),2,color=(0,0,0),thickness=-1)
        if self.cross_mode:# Dezenha a soma das zonas para x e y
            r1  =(cx+zx,0)#pontos do retangulo
            r2  =(cx-zx,ry)
            r3  =(0,zy+cy)
            r4  =(rx,cy-zy)
            cv.rectangle(self.frame,pt1=r1,pt2=r2,color=(0,0,0),thickness=3)
            cv.rectangle(self.frame,pt1=r3,pt2=r4,color=(0,0,0),thickness=3)
            cv.rectangle(self.frame,pt1=(int(mx),int(0)),pt2=(int(rx-mx),int(ry)),color=(255,255,255),thickness=3)
            cv.rectangle(self.frame,pt1=(int(0),int(my)),pt2=(int(rx),int(ry-my)),color=(255,255,255),thickness=3)
        else:# Dezenha a intercessão das zonas para x e y
            r1  =(cx+zx,zy+cy)#pontos do retangulo
            r2  =(cx-zx,cy-zy)
            cv.rectangle(self.frame,pt1=r1,pt2=r2,color=(0,0,0),thickness=3)
            cv.rectangle(self.frame,pt1=(int(mx),int(my)),pt2=(int(rx-mx),int(ry-my)),color=(255,255,255),thickness=3)

    def drawHandAndBox(self,box,cat,id): 
        (x,y) =(int(self.hand_center[0]),int(self.hand_center[1]))#poisição do centro da mão em inteiros
        #desenha um retangulo ao redor dos pontos da mão
        dist = self.handDist
        cv.drawContours(self.frame,[box],contourIdx=0,color=(255,0,0),thickness=2)
        cv.circle(self.frame,(x,y),2,color=(0,0,255),thickness=-1)#ponto central do retangulo
        cv.putText(self.frame,f"{cat}",(x,y-10),cv.FONT_HERSHEY_PLAIN,1,(0,255,255))#Categoria(Lado) da mão
        cv.putText(self.frame,f"{dist}",(x,y+10),cv.FONT_HERSHEY_PLAIN,1,(0,255,255))#Status da mão
        
        #linha para do centro da imagem para o centro da mão
        (cx,cy) = self.center
        cv.line(self.frame,(cx,cy),(x,cy),(255,0,0),thickness=2)
        cv.line(self.frame,(x,cy),(x,y),(255,0,0),thickness=2)
        cv.line(self.frame,(cx,cy),(x,cy),(0,0,255),thickness=1)
        cv.line(self.frame,(x,cy),(x,y),(0,0,255),thickness=1)

        print(f"{id}\n\tSide:{cat}\n\tDist:{dist}")# print para as informações das mãos
    
        #desenha linhas entre os dedos da mão
        duos = [(0,1),(0,5),(0,17),(5,9),(9,13),(13,17)]
        duos +=[(v+i-1,v+i) for v in set([vl[1] for vl in duos]) for i in range(1,4) if v!=0]+[(2,5)]
        for a,b in duos:#usa as duplas de indexes dos pontos para desenhar as linhas
            self.frame = cv.line(self.frame,self.hand_points[a],self.hand_points[b],color=(0,255,0))
        for p in self.hand_points:#desenha cada ponto da mão e numera eles
            self.frame = cv.putText(self.frame,f"{self.hand_points.index(p)}",p,cv.FONT_HERSHEY_PLAIN,1,(255,255,0))
            self.frame = cv.circle(self.frame,p,2,color=(255,0,255),thickness=-1)
    
    def __getMiddleAndBox(self):# pega os pontos da caixa e centro da mão
        r = cv.minAreaRect(np.array([self.hand_points]))
        self.hand_center = r[0]
        return cv.boxPoints(r)  #pixel Position  
        
    def main(self):

        ret, frame = self.cap.read()
        
        handSwitch = {0:'Left',1:'Right'}#corrige o lado das mãos
        x,y = self.rez
        print(self.rez)
        #cv.namedWindow('Webcam', cv.WINDOW_KEEPRATIO)
        while True:
            ret, frame = self.cap.read()
            self.frame = cv.flip(frame,1)
            if not ret: continue
            frame_RGB = mp.Image(mp.ImageFormat.SRGB,cv.cvtColor(self.frame,cv.COLOR_BGR2RGB))
            r = self.detector.detect(frame_RGB)#resultado da detecção
            size = len(r.hand_landmarks)
            self.__drawnZones()#desenha a zona morta
            if size>0:#ignora se nenuma mão for detectada
                hand = r.hand_landmarks[0]
                h = r.handedness[0][0]
                self.hand_points = [(int(l.x*x),int(l.y*y)) for l in hand]
                box = self.__getMiddleAndBox()
                self.drawHandAndBox(box.astype(np.int64),handSwitch[h.index],"Main Hand Stats:")
            cv.imshow('Webcam', self.frame)#mostra a imagem capturada com as alterações feitas
            if cv.waitKey(1) & 0xFF == ord('q'): break

        self.cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    d = {"detection":0.4,"presence":0.4,"traking":0.6}
    det = handDetection(cross_mode=True)
    det.main()