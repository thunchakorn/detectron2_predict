import cv2

import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

class INV_layout(object):
    def __init__(self, weight_dir = './weight/', thres = 0.5):

        self.weight = os.path.join(weight_dir, os.listdir(weight_dir)[0])
        self.thres = thres
        self.thing_classes = ['DocType', 'Item', 'Payment', 'Reciever', 'Remark',
                        'Sender', 'Signature', 'Summary', 'Table']


        cfg = get_cfg()
        config_file = 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'
        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        cfg.MODEL.WEIGHTS = self.weight
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.thres
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.thing_classes)
        self.predictor = DefaultPredictor(cfg)

    def find_polygon(self, mask, percent = 0.005):
        mask = np.uint8(mask)
        contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        eps = percent * cv2.arcLength(contour[0], True)
        out_curve = cv2.approxPolyDP(contour[0], eps, False)
        return out_curve.reshape(-1,2).tolist()

    def predict(self, img, thing_classes):

        output = self.predictor(img)['instances'].to("cpu")
        instance = output.get_fields()
        imageHeight = output.image_size[0]
        imageWidth = output.image_size[1]
        results = []
        for pred, mask in zip(instance['pred_classes'].numpy(), instance['pred_masks'].numpy()):
            polygon = find_polygon(mask)
            dict_predict = {
                            'label':self.thing_classes[pred],
                            'points':polygon
                            }
            results.append(dict_predict)
        return results
