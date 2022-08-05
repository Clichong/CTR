from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

import torch
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *

import os
import numpy as np
import pandas as pd

from tests.feature_process import ads_dense_features, ads_sparse_features

# 环境变量设置
SEED = 1024
os.environ['CUDA_LAUNCH_BLOCKING'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

# 数据读取
target = ["label"]
# data = pd.read_csv("ctr_data/train/train_data_ads.csv")
# test = pd.read_csv('ctr_data/test/test_data_ads.csv')
# all_data = pd.concat([data, test])

# 稀疏特征编码处理
# data, test = ads_sparse_feature_process(dataframes=[all_data, data, test], features=ads_sparse_features)
# 密集特征编码处理
# data, test = ads_dense_feature_process(dataframes=[all_data, data, test], features=ads_dense_features)

# 使用新数据集
data = pd.read_csv("train_newdata_ads.csv")
test = pd.read_csv('test_newdata_ads.csv')
all_data = pd.concat([data, test])

# 统计稀疏特征的unique(需要注意，这里是要是全部值的unique)，同时记录密集特征名称
fixlen_feature_columns = [SparseFeat(feat, all_data[feat].nunique(), embedding_dim=8) for feat in ads_sparse_features] \
                        + [DenseFeat(feat, 1, ) for feat in ads_dense_features]
# fixlen_feature_columns = [DenseFeat(feat, 1, ) for feat in ads_sparse_features] \
#                         + [DenseFeat(feat, 1, ) for feat in ads_dense_features]
dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# 数据集切分
train, val = train_test_split(data, test_size=0.2, random_state=SEED)
train_model_input = {name: train[name] for name in feature_names}
val_model_input   = {name: val[name] for name in feature_names}
test_model_input  = {name: test[name] for name in feature_names}

# 模型构建与训练
batch_size = 4096
model = DeepFM(linear_feature_columns=linear_feature_columns,
               dnn_feature_columns=dnn_feature_columns,
               task='binary',
               dnn_dropout=0.7,
               dnn_use_bn=True,
               dnn_hidden_units=(256, 128),
               device='cuda:2')
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['binary_crossentropy', 'auc'])
model.fit(train_model_input, train[target].values, batch_size=batch_size, epochs=1, verbose=1, validation_split=0.)

# 模型测试
pred_val = model.predict(val_model_input, batch_size=batch_size)
pred_val = np.around(pred_val, 6)
print("")
print("val LogLoss", round(log_loss(val[target].values, pred_val), 4))
print("val AUC", round(roc_auc_score(val[target].values, pred_val), 4))

# 模型预测
pred_test = model.predict(test_model_input, batch_size=batch_size)
pred_test = np.around(pred_test, 6)
submission = pd.DataFrame()
submission['log_id'] = test['log_id']
submission['pctr'] = pred_test.astype(np.float32)
submission.to_csv("submission.csv", index=False)
print("create submission")



