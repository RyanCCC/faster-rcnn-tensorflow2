import colorsys
import os
import time

import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.applications.imagenet_utils import preprocess_input

import nets.fasterrcnn as frcnn
from utils.anchors import get_anchors
from tools.common import cvtColor, get_classes, get_new_img_size, resize_image
from utils.boundingbox import BBoxUtility
import config


class FRCNN(object):
    def __init__(self):
        self.model_path = config.model
        self.classes_path = config.classes_path
        self.backbone = config.backbone
        self.confidence= config.confidence
        self.nms_iou = config.nms_iou
        self.anchors_size = config.anchors
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.num_classes  = self.num_classes + 1
        self.bbox_util = BBoxUtility(self.num_classes, nms_iou = self.nms_iou, min_k = 150)
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        # 加载模型
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        self.model_rpn, self.model_classifier = frcnn.get_model(self.num_classes, self.backbone, Training=False)
        self.model_rpn.load_weights(self.model_path, by_name=True)
        self.model_classifier.load_weights(self.model_path, by_name=True)
        print('{} model, anchors, and classes loaded.'.format(model_path))
    
    def detect_image(self, image, crop = False):
        image_shape = np.array(np.shape(image)[0:2])
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        image       = cvtColor(image)
        image_data  = resize_image(image, [input_shape[1], input_shape[0]])
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
        rpn_pred        = self.model_rpn(image_data)
        rpn_pred        = [x.numpy() for x in rpn_pred]
        anchors         = get_anchors(input_shape, self.backbone, self.anchors_size)
        rpn_results     = self.bbox_util.detection_out_rpn(rpn_pred, anchors)
        classifier_pred = self.model_classifier([rpn_pred[2], rpn_results[:, :, [1, 0, 3, 2]]])
        classifier_pred = [x.numpy() for x in classifier_pred]
        results         = self.bbox_util.detection_out_classifier(classifier_pred, rpn_results, image_shape, input_shape, self.confidence)

        if len(results[0]) == 0:
            return image
            
        top_label   = np.array(results[0][:, 5], dtype = 'int32')
        top_conf    = results[0][:, 4]
        top_boxes   = results[0][:, :4]
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // input_shape[0], 1)
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0))
            del draw
        return image

if __name__ == '__main__':
    frcnn = FRCNN()
    image_path = ''
    image = Image.open(image_path)
    r_image = frcnn.detect_image(image, crop = False)
    r_image.show()
