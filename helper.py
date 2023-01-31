import cv2
import numpy as np
import openvino.runtime as ov
from typing import List, Tuple

class DamageHelper:
    def __init__(self, model_path, use_fp16=False):
        core = ov.Core()
        damage_model = core.read_model(model_path)
        self.__compiled_model = core.compile_model(damage_model, "CPU")
        self.__infer_request = self.__compiled_model.create_infer_request()
        self.__OUTPUT_NODE = self.__compiled_model.outputs[0]
        self.use_fp16 = use_fp16

    def process(self, image, size):
        try:
            ori_h, ori_w, _ = image.shape
            processed_img, _, _ = self._preprocessing(image, (size, size))
            _, _, processed_h, processed_w = processed_img.shape
            input_tensor = ov.Tensor(processed_img, shared_memory=True)

            preds = self.__infer_request.infer(inputs={"images": input_tensor})[
                self.__OUTPUT_NODE]
            preds = self._non_max_suppression(preds, conf_thres=0.5)
            if len(preds) > 0:
                preds = preds[0]
                scaled_boxes = self._scale_boxes(
                    new_shape=(processed_h, processed_w),
                    boxes=preds[:, :4],
                    original_shape=(ori_h, ori_w),
                )
                preds[:, :4] = scaled_boxes.round()
                preds[:, 4] = preds[:, 4] * 100

                preds = preds.astype(int).tolist()

            return preds
        except Exception as e:
            print(e)
            return None

    def _letterbox(
        self,
        im: np.ndarray, 
        new_size: Tuple[int, int] = (1280, 1280), 
        color: Tuple[int,int,int] = (114, 114, 114)
    ) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
        # Resize and pad image while meeting stride-multiple constraints
        
        shape = im.shape[:2]
        
        # Scale ratio (new / old)
        r = min(new_size[0] / shape[0], new_size[1] / shape[1])
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_size[1] - new_unpad[0], new_size[0] - new_unpad[1]  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)

    def _xywh2xyxy(self, x: np.ndarray) -> np.ndarray:
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def _nms(
        self,
        boxes: np.ndarray, 
        scores: np.ndarray, 
        iou_threshold: float = 0.45
    ) -> np.ndarray:
        # coordinates of bounding boxes
        start_x = boxes[:, 0]
        start_y = boxes[:, 1]
        end_x = boxes[:, 2]
        end_y = boxes[:, 3]

        # Picked bounding boxes
        picked_boxes = []

        # Compute areas of bounding boxes
        areas = (end_x - start_x + 1) * (end_y - start_y + 1)  # type: ignore

        # Sort by confidence score of bounding boxes
        order = np.argsort(scores)

        # Iterate bounding boxes
        while order.size > 0:
            # The index of largest confidence score
            index = order[-1]

            # Pick the bounding box with largest confidence score
            picked_boxes.append(index)

            # Compute ordinates of intersection-over-union(IOU)
            x1 = np.maximum(start_x[index], start_x[order[:-1]])
            x2 = np.minimum(end_x[index], end_x[order[:-1]])
            y1 = np.maximum(start_y[index], start_y[order[:-1]])
            y2 = np.minimum(end_y[index], end_y[order[:-1]])

            # Compute areas of intersection-over-union
            w = np.maximum(0.0, x2 - x1 + 1)
            h = np.maximum(0.0, y2 - y1 + 1)
            intersection = w * h

            # Compute the ratio between intersection and union
            ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

            left = np.where(ratio < iou_threshold)
            order = order[left]

        return np.array(picked_boxes)
        
    def _non_max_suppression(
        self,
        prediction: np.ndarray,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
    ) -> List[np.ndarray]:
        """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
        Arguments:
            prediction: [bach, num_preds, 7] with 7 = [x, y, w, h, conf, cls0, cls1]
        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """
        
        bs = prediction.shape[0]  # batch size
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

        output = []
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute real conf
            x[:, 5:] *= x[:, 4:5]  # real conf = obj_conf * cls_conf

            # Box/Mask
            box = self._xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        
            # Detections matrix nx6 (xyxy, conf, cls)
            ids = x[:, 5:].argmax(1, keepdims=True)
            max_vals = x[:, 5:].max(1, keepdims=True)
            x = np.concatenate((box, max_vals, ids), 1)[max_vals.flatten() > conf_thres]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n or n == 0:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[(-x[:, 4]).argsort()[:max_nms]]  # sort by confidence
            else:
                x = x[(-x[:, 4]).argsort()]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * max_wh  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = self._nms(boxes, scores, iou_thres)
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            output.append(x[i])

        return output

    def _clip_boxes(self, boxes, shape):
            # Clip boxes (xyxy) to image shape (height, width)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
            return boxes

    def _scale_boxes(
        self,
        new_shape: Tuple[int, int], 
        boxes: np.ndarray, 
        original_shape: Tuple[int, int], 
    ) -> np.ndarray:
        # Rescale boxes (xyxy) from new_shape to original_shape
        gain = min(new_shape[0] / original_shape[0], new_shape[1] / original_shape[1])  # gain  = old / new
        pad = (new_shape[1] - original_shape[1] * gain) / 2, (new_shape[0] - original_shape[0] * gain) / 2  # wh padding
        
        # x padding
        boxes[:, [0, 2]] -= pad[0]  # type: ignore
        # y padding 
        boxes[:, [1, 3]] -= pad[1]  # type: ignore 
        boxes[:, :4] /= gain
        boxes = self._clip_boxes(boxes, original_shape)
        return boxes

    def _preprocessing(
        self,
        img: np.ndarray, 
        new_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
        
        preprocessed_img, ratio, padding = self._letterbox(img, new_size=new_size)
        # batch image
        preprocessed_img = preprocessed_img[None,...]
        preprocessed_img = preprocessed_img / 255.0
        # preprocessed_img = preprocessed_img.astype(np.float16)
        preprocessed_img = preprocessed_img.astype(np.float32)
        preprocessed_img = preprocessed_img.transpose(0,3,1,2)
        preprocessed_img = np.ascontiguousarray(preprocessed_img)
        return preprocessed_img, ratio, padding


