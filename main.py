import cv2
import numpy as np
from util import get_parking_spots_bboxes,empty_or_not
from ultralytics import YOLO
import matplotlib.pyplot as plt
import time
import math

model=YOLO('trained.pt')
# results=model.train(data="config.yaml", epochs=1)  #train

def calc_diff(im1,im2):
    return np.abs(np.mean(im1) - np.mean(im2))

video_path = "./parking_1920_1080.mp4"
mask_path = "./mask.png"

mask= cv2.imread(mask_path,0)   # 0 is for color mode => GrayScale
cap = cv2.VideoCapture(video_path)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)
spots=sorted(spots,key=lambda x: (x[0],x[1]))

spots_status = [None for j in spots]
diffs = [None for j in spots]

prev_frame = None

ret = True
frame_nmr=0
step = 30
step2 = 15

framecrops = [[95,1080, 220,285], [95,1080, 460,540], [95,1080, 685,760], [95,990, 915,990],
              [95,990, 1140,1220], [95,990, 1375,1440], [95,990, 1595,1675], [95,990, 1850,1920],
              [990,1080, 915,1920]]

stime=time.time()
while ret:

    # if frame_nmr%30 == 0:
    #     print(time.time()-stime)
    #     stime=time.time()

    ret, frame = cap.read()

    if frame_nmr % step == 0:

        # This approach calculates the empty_or_not function for all the spots (not efficient enough)

        # for spot_ind, spot in enumerate(spots):
        #     x1, y1, w, h = spot
        #     spot_crop = frame[y1:y1 + h, x1:x1 + w]
        #     spot_status = empty_or_not(spot_crop)
        #     spots_status[spot_ind] = spot_status

        # Optimised to run empty_or_not only on spots that have significant diff from the previous frame


        for spot_ind, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w]
            if prev_frame is not None:
                diffs[spot_ind] = calc_diff(spot_crop, prev_frame[y1:y1 + h, x1:x1 + w])
            else:
                spot_status = empty_or_not(spot_crop)
                spots_status[spot_ind] = spot_status

        if prev_frame is not None:
            # print(diffs)
            for spot_ind in [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]:
                spot = spots[spot_ind]
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w]
                spot_status = empty_or_not(spot_crop)
                spots_status[spot_ind] = spot_status

        prev_frame = frame.copy()


    for spot_ind,spot in enumerate(spots):
        spot_status = spots_status[spot_ind]
        x1, y1, w, h = spot
        if spot_status:
            cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,255,0),2)
            cv2.putText(frame,str(spot_ind+1),(x1+int(w/4),y1+int(h/1.3)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        else:
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

    if frame_nmr% step2 == 0:
        prevcord=[]
        for framecrop in framecrops:
            cy1,cy2,cx1,cx2=framecrop
            framecrop=frame[cy1:cy2,cx1:cx2]
            results=model(framecrop)
            for carbox in results[0].boxes.xyxy.tolist():
                bx1,by1,bx2,by2= map(int,carbox)
                p1 = [cx1 + int((bx1+bx2)/2), cy1 + int((by1+by2)/2)]
                index=-1
                for spot_ind,spot in enumerate(spots):
                    if spots_status[spot_ind]:
                        x1, y1, w, h = spot
                        p2=[x1+int(w/2),y1+int(h/2)]
                        if index==-1:
                            mindis=math.dist(p1,p2)
                            index=spot_ind+1
                        else:
                            if(math.dist(p1,p2)<mindis):
                                mindis = math.dist(p1, p2)
                                index=spot_ind+1

                prevcord.append([cx1+bx1,cy1+by1,cx1+bx2,cy1+by2,index])
                cv2.rectangle(frame,(cx1+bx1,cy1+by1),(cx1+bx2,cy1+by2),(255,0,0),3)
                cv2.putText(frame, str(index),(cx1+bx1,cy1+by1-10),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    else:
        for carbox in prevcord:
            px1, py1, px2, py2, index = carbox
            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 3)
            cv2.putText(frame, str(index), (px1, py1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, 'Available Spots: {} / {}'.format(str(sum(spots_status)), str(len(spots_status))), (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    if frame is not None:
        cv2.imshow('frame', frame)
    if cv2.waitKey(25) == 27:
        break

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()
