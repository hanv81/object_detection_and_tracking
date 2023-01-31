import cv2
import time
import numpy as np
from helper import DamageHelper
from collections import deque
from jproperties import Properties
from tracker import track

def is_overlap(bb1, bb2):
    xx = min(bb1[2], bb2[2]) > max(bb1[0], bb2[0])  # the smaller of the largest x-coordinates is larger than the larger of the smallest x-coordinates
    yy = min(bb1[3], bb2[3]) > max(bb1[1], bb2[1])  # the smaller of the largest y-coordinates is larger than the larger of the smallest y-coordinates
    return xx and yy

def get_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def read_config():
    configs = Properties()
    with open('config.properties', 'rb') as prop:
        configs.load(prop)

    roi = list(map(int, configs.get('roi').data.split()))
    model = configs.get('model').data
    return roi, model

def tracking():
    roi, model_path = read_config()
    roi_center = np.array([(roi[2]-roi[0])//2, (roi[3]-roi[1])//2])
    model = DamageHelper(model_path)
    vid = cv2.VideoCapture(0)
    show_fps = True
    bbox_centers = deque(maxlen=15)
    while True:
        _, frame = vid.read()
        t = time.time()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0,255,0), 3)
        results = model.process(frame, 416)
        results = [p for p in results if p[-1] == 0] # person only
        if results:
            results = np.array(results, dtype=float)
            detections = track(frame, results)

            bb = {}
            bb0 = bbox_centers[0] if len(bbox_centers) > 0 else None
            for d in detections:
                bb[d.tracker_id] = d.rect.center
                x1, y1, x2, y2 = int(d.rect.x), int(d.rect.y), int(d.rect.max_x), int(d.rect.max_y)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.putText(frame, f'{d.tracker_id}', org=(x1+2, y1+15), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.5, color=(0, 255, 0), thickness=2)
                x, y = int(d.rect.center.x), int(d.rect.center.y)
                if is_overlap(roi, [x1, y1, x2, y2]):
                    # print('**** overlap ****')
                    cv2.putText(frame, 'OVERLAP', org=(x, y), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=.5, color=(0, 0, 255), thickness=2)
                elif len(bbox_centers) == bbox_centers.maxlen:
                    p = bb0.get(d.tracker_id)
                    if p is not None:
                        # print(p)
                        p0 = np.array([p.x, p.y])
                        p2 = np.array([x, y])
                        d1 = get_distance(roi_center, p0)
                        d2 = get_distance(roi_center, p2)
                        if d1 - d2 > 20:
                            # print('############## alert ##############')
                            cv2.putText(frame, 'ALERT', org=(x, y), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=.5, color=(0, 0, 255), thickness=2)
            
            bbox_centers += [bb]

        t = time.time() - t
        if show_fps and t != 0:
            # print('FPS:', int(1/t))
            cv2.putText(frame, f'FPS: {int(1/t)}', org=(0, 15), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.5, color=(0, 0, 255), thickness=2)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord('f'):
            show_fps = not show_fps

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Camera", frame)

tracking()