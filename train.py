import config
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from nets.fasterrcnn import get_model
from utils.loss import ProposalTargetCreator, classifier_cls_loss,classifier_smooth_l1, rpn_cls_loss,rpn_smooth_l1
from utils.anchors import get_anchors
from utils.dataloader import FRCNNDatasets
from tools.common import get_classes
from utils.boundingbox import BBoxUtility
from tqdm import tqdm
import numpy as np
import os


def write_log(callback, names, logs, batch_no):
    with callback.as_default():
        for name, value in zip(names, logs):
            tf.summary.scalar(name,value,step=batch_no)
            callback.flush()


def fit_one_epoch(model_rpn, model_all, callback, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, anchors, bbox_util, roi_helper):
    total_loss = 0
    rpn_loc_loss = 0
    rpn_cls_loss = 0
    roi_loc_loss = 0
    roi_cls_loss = 0

    val_loss = 0
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            X, Y, boxes = batch[0], batch[1], batch[2]
            P_rpn       = model_rpn.predict_on_batch(X)

            results = bbox_util.detection_out_rpn(P_rpn, anchors)

            roi_inputs = []
            out_classes = []
            out_regrs = []
            for i in range(len(X)):
                R = results[i]
                X2, Y1, Y2 = roi_helper.calc_iou(R, boxes[i])
                roi_inputs.append(X2)
                out_classes.append(Y1)
                out_regrs.append(Y2)

            loss_class = model_all.train_on_batch([X, np.array(roi_inputs)], [Y[0], Y[1], np.array(out_classes), np.array(out_regrs)])
            
            write_log(callback, ['total_loss','rpn_cls_loss', 'rpn_reg_loss', 'detection_cls_loss', 'detection_reg_loss'], loss_class, iteration)

            rpn_cls_loss += loss_class[1]
            rpn_loc_loss += loss_class[2]
            roi_cls_loss += loss_class[3]
            roi_loc_loss += loss_class[4]
            total_loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss

            pbar.set_postfix(**{'total'    : total_loss / (iteration + 1),  
                                'rpn_cls'  : rpn_cls_loss / (iteration + 1),   
                                'rpn_loc'  : rpn_loc_loss / (iteration + 1),  
                                'roi_cls'  : roi_cls_loss / (iteration + 1),    
                                'roi_loc'  : roi_loc_loss / (iteration + 1), 
                                'lr'       : K.get_value(model_rpn.optimizer.lr)})
            pbar.update(1)

    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            X, Y, boxes = batch[0], batch[1], batch[2]
            P_rpn       = model_rpn.predict_on_batch(X)
            
            results = bbox_util.detection_out_rpn(P_rpn, anchors)

            roi_inputs = []
            out_classes = []
            out_regrs = []
            for i in range(len(X)):
                R = results[i]
                X2, Y1, Y2 = roi_helper.calc_iou(R, boxes[i])
                roi_inputs.append(X2)
                out_classes.append(Y1)
                out_regrs.append(Y2)
                
            loss_class = model_all.test_on_batch([X, np.array(roi_inputs)], [Y[0], Y[1], np.array(out_classes), np.array(out_regrs)])

            val_loss += loss_class[0]
            pbar.set_postfix(**{'total' : val_loss / (iteration + 1)})
            pbar.update(1)

    logs = {'loss': total_loss / epoch_step, 'val_loss': val_loss / epoch_step_val}
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    model_all.save_weights('logs/ep%03d-loss%.3f-val_loss%.3f.h5' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val))



if __name__ == '__main__':
    class_path = config.classes_path
    pretrain_model = config.pretrain_model
    input_shape = config.input_shape
    backbone = config.backbone
    anchor_size = config.anchors

    freeze_epoch = config.free_epoch
    freeze_batch_size = config.freeze_batch_size
    freeze_lr = config.freeze_lr

    epoch = config.epoch
    batch_size = config.batch_size
    learning_rate = config.learning_rate

    freeze_train = config.freeze_train

    train_txt = config.train_annotation_path
    val_txt = config.val_annotation_path

    log_dir = config.log

    # 获取类别名称
    class_name, num_classes = get_classes(class_path)
    num_classes += 1
    anchors = get_anchors(input_shape, backbone, anchor_size)

    K.clear_session()
    model_rpn, model_all = get_model(num_classes, backbone = backbone)
    # 加载预训练权重
    if  os.path.exists(pretrain_model):
        print('Load weights {}.'.format(pretrain_model))
        model_rpn.load_weights(pretrain_model, by_name=True)
        model_all.load_weights(pretrain_model, by_name=True)

    # 设置训练参数
    callback        = tf.summary.create_file_writer(log_dir)
    bbox_util       = BBoxUtility(num_classes)
    roi_helper      = ProposalTargetCreator(num_classes)

    # 获取训练数据
    with open(train_txt) as f:
        train_lines = f.readlines()
    with open(val_txt) as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    freeze_layers = {'vgg' : 17, 'resnet50' : 141}[backbone]

    if freeze_train:
        for i in range(freeze_layers): 
            if type(model_all.layers[i]) != tf.keras.layers.BatchNormalization:
                model_all.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_all.layers)))

        batch_size  = freeze_batch_size
        lr= freeze_lr
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        model_rpn.compile(
            loss = {
                'classification': rpn_cls_loss(),
                'regression'    : rpn_smooth_l1()
            }, optimizer = Adam(lr=lr)
        )
        model_all.compile(
            loss = {
                'classification'                        : rpn_cls_loss(),
                'regression'                            : rpn_smooth_l1(),
                'dense_class_{}'.format(num_classes)    : classifier_cls_loss(),
                'dense_regress_{}'.format(num_classes)  : classifier_smooth_l1(num_classes - 1)
            }, optimizer = Adam(lr=lr)
        )

        gen_train  = FRCNNDatasets(train_lines, input_shape, anchors, batch_size, num_classes, train = True).generate()
        gen_val = FRCNNDatasets(val_lines, input_shape, anchors, batch_size, num_classes, train = False).generate()

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        for e in range(freeze_epoch):
            fit_one_epoch(model_rpn, model_all, callback, e, epoch_step, epoch_step_val, gen_train, gen_val, freeze_epoch,anchors, bbox_util, roi_helper)
            lr = lr*0.96
            K.set_value(model_rpn.optimizer.lr, lr)
            K.set_value(model_all.optimizer.lr, lr)
    
        # 解冻
        for i in range(freeze_layers): 
            if type(model_all.layers[i]) != tf.keras.layers.BatchNormalization:
                model_all.layers[i].trainable = True

    batch_size  = batch_size
    lr  = learning_rate

    epoch_step      = num_train // batch_size
    epoch_step_val  = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

    model_rpn.compile(
        loss = {
            'classification': rpn_cls_loss(),
            'regression'    : rpn_smooth_l1()
        }, optimizer = Adam(lr=lr)
    )
    model_all.compile(
        loss = {
            'classification'                        : rpn_cls_loss(),
            'regression'                            : rpn_smooth_l1(),
            'dense_class_{}'.format(num_classes)    : classifier_cls_loss(),
            'dense_regress_{}'.format(num_classes)  : classifier_smooth_l1(num_classes - 1)
        }, optimizer = Adam(lr=lr)
    )

    gen_train     = FRCNNDatasets(train_lines, input_shape, anchors, batch_size, num_classes, train = True).generate()
    gen_val = FRCNNDatasets(val_lines, input_shape, anchors, batch_size, num_classes, train = False).generate()

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    for e in range(epoch):
        fit_one_epoch(model_rpn, model_all, callback, e, epoch_step, epoch_step_val, gen_train, gen_val, epoch,
                    anchors, bbox_util, roi_helper)
        lr = lr*0.96
        K.set_value(model_rpn.optimizer.lr, lr)
        K.set_value(model_all.optimizer.lr, lr)
        




