import cv2
import time
from helper import DamageHelper
import numpy as np
from collections import deque

model = DamageHelper('yolov5s_openvino_model/yolov5s.xml')

params = cv2.TrackerNano_Params()
params.backbone = "nanotrack_backbone_sim.onnx"
params.neckhead = "nanotrack_head_sim.onnx"

tracker = cv2.TrackerNano_create(params)

roi = [20, 20, 100, 120]
roi_center = np.array([(roi[2]-roi[0])//2, (roi[3]-roi[1])//2])

def is_overlap(bb1, bb2):
    xx = min(bb1[2], bb2[2]) > max(bb1[0], bb2[0])  # the smaller of the largest x-coordinates is larger than the larger of the smallest x-coordinates
    yy = min(bb1[3], bb2[3]) > max(bb1[1], bb2[1])  # the smaller of the largest y-coordinates is larger than the larger of the smallest y-coordinates
    return xx and yy

def get_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def tracking():
    vid = cv2.VideoCapture(0)
    bbox_centers = deque(maxlen=15)
    track_bb = None
    show_fps = True
    frame_cnt = 0
    while True:
        _, frame = vid.read()
        frame_cnt += 1
        t = time.time()
        if frame_cnt % 15 == 0:
            track_bb = None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0,255,0), 3)
        if track_bb is None:
            results = model.process(frame, 416)
            # print(results)
            if results:
                for x1, y1, x2, y2, _, c in results:
                    if c == 0:  # person class
                        track_bb = [x1, y1, x2, y2]
                        # print(f'tracking bbox: {track_bb}')
                        tracker.init(frame, (x1, y1, x2-x1, y2-y1))
                        break
            else:
                w, h, _ = frame.shape
                cv2.putText(frame, 'NO OBJECT', org=(w//2, h//2), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=1, color=(0, 0, 255), thickness=2)

        if track_bb is not None:
            success, box = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)

                bbox_centers += [(x+w//2, y+h//2)]

                if is_overlap(roi, [x, y, x+w, y+h]):
                    # print('**** overlap ****')
                    cv2.putText(frame, 'OVERLAP', org=(x+w//2, y+h//2), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=.5, color=(0, 0, 255), thickness=2)
                else:
                    if len(bbox_centers) == bbox_centers.maxlen:
                        d1 = get_distance(roi_center, bbox_centers[0])
                        d2 = get_distance(roi_center, bbox_centers[-1])
                        if d1 - d2 > 20:
                            # print('############## alert ##############')
                            cv2.putText(frame, 'ALERT', org=(x+w//2, y+h//2), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=.5, color=(0, 0, 255), thickness=2)
            else:
                track_bb = None
                w, h, _ = frame.shape
                cv2.putText(frame, 'NO OBJECT', org=(w//2, h//2), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=1, color=(0, 0, 255), thickness=2)

        t = time.time() - t
        if show_fps and t != 0:
            # print('FPS:', int(1/t))
            cv2.putText(frame, f'FPS: {int(1/t)}', org=(0, 15), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.5, color=(0, 0, 255), thickness=2)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord('s'):
            track_bb = None
        elif key == ord('f'):
            show_fps = not show_fps
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Camera", frame)

tracking()