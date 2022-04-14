from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from nets.classifier import get_resnet50_classifier, get_vgg_classifier
from nets.resnet import ResNet50
from nets.rpn import get_rpn
from nets.vgg import VGG16


def get_model(num_classes, backbone, num_anchors=9, Training=True):
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    # 获取训练模型
    if Training:
        if backbone == 'vgg':
            # 特征提取
            base_layers = VGG16(inputs)
            # rpn网络
            rpn = get_rpn(base_layers, num_anchors)
            # 分类器
            classifier = get_vgg_classifier(base_layers, roi_input, 7, num_classes)
        else:
            # resnet
            base_layers = ResNet50(inputs)
            # rpn网络
            rpn = get_rpn(base_layers, num_anchors)
            # 分类器
            classifier = get_vgg_classifier(base_layers, roi_input, 7, num_classes)
        model_rpn   = Model(inputs, rpn)
        model_all   = Model([inputs, roi_input], rpn + classifier)
        return model_rpn, model_all
    # 获取推理模型
    else:
        if backbone == 'vgg':
            feature_map_input = Input(shape=(None, None, 512))
            base_layers = VGG16(inputs)
            rpn = get_rpn(base_layers, num_anchors)
            classifier = get_vgg_classifier(feature_map_input, roi_input, 7, num_classes)
        else:
            feature_map_input = Input(shape=(None, None, 1024))
            base_layers = ResNet50(inputs)    
            rpn = get_rpn(base_layers, num_anchors)
            classifier  = get_resnet50_classifier(feature_map_input, roi_input, 14, num_classes)
    
        model_rpn   = Model(inputs, rpn + [base_layers])
        model_classifier_only = Model([feature_map_input, roi_input], classifier)
        return model_rpn, model_classifier_only

        