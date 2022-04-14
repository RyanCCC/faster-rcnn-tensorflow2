# training
classes_path = './data/voc_classes.txt'
backbone = 'resnet50'
pretrain_model = './model/voc_weights_resnet.h5'
input_shape = [600, 600]
anchors = [128, 256, 512]
# 冻结训练
free_epoch = 50
freeze_batch_size = 16
freeze_lr = 1e-4

epoch = 100
batch_size = 16
learning_rate = 1e-5
freeze_train = True

# 数据集
train_annotation_path   = '2007_train.txt'
val_annotation_path     = '2007_val.txt'

# log
log='./log'
ckp_weight = 'random.h5'

# 预测
model = './ddd.h5'
confidence=0.5
nms_iou=0.3