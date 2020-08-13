import cv2
import os
import numpy as np

import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

WEIGHT_PATH = os.environ.get('WEIGHT_PATH')

class INV_layout(object):
    def __init__(self, cuda = False):

        self.thing_classes = ['DocType', 'Item', 'Payment', 'Reciever', 'Remark',
                        'Sender', 'Signature', 'Summary', 'Table']

        weight = os.path.join(WEIGHT_PATH, os.listdir(WEIGHT_PATH)[0])
        config_file = 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        cfg.MODEL.WEIGHTS = weight
        if not cuda:
            cfg.MODEL.DEVICE = 'cpu'
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.thing_classes)
        self.predictor = DefaultPredictor(cfg)

    def find_polygon(self, mask, percent = 0.005):
        mask = np.uint8(mask)
        contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        eps = percent * cv2.arcLength(contour[0], True)
        out_curve = cv2.approxPolyDP(contour[0], eps, False)
        return out_curve.reshape(-1,2).tolist()

    def predict(self, img, thresh = 0.5):
        self.predictor.model.roi_heads.box_predictor.test_score_thresh = thresh
        output = self.predictor(img)['instances'].to("cpu")
        instance = output.get_fields()
        imageHeight = output.image_size[0]
        imageWidth = output.image_size[1]
        results = []
        for pred, mask, score, bbox in zip(instance['pred_classes'].numpy(),
                                        instance['pred_masks'].numpy(),
                                        instance['scores'].numpy(),
                                        instance['pred_boxes'].tensor.cpu().numpy()):
            polygon = self.find_polygon(mask)
            dict_predict = {
                            'label':self.thing_classes[pred],
                            'polygon':polygon,
                            'bbox':bbox.tolist(),
                            'score':score.tolist()
                            }
            results.append(dict_predict)
        return results
