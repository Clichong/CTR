import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import DataLoader

import time
import pandas as pd
import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model import CrossDomainNet
from utils.feature_deal import target_feature_process, target_sparse_feature_names, target_varlen_sparse_feature_names


# 预测
def test_predict(model, test_loader, device):

    model.eval()

    pred_ans = []
    steps_per_epoch = len(test_loader)

    with torch.no_grad():
        for x_test in tqdm(test_loader, desc="predict label"):

            x = x_test[0].to(device).float()
            y_pred = model(x).cpu().data.numpy()  # .squeeze()
            pred_ans.append(y_pred)

    pred_ans = np.concatenate(pred_ans).astype("float64")
    pred_ans = np.around(pred_ans, 6)

    return pred_ans


if __name__ == '__main__':

    batch_size = 4096
    device_ids = '0'
    device = torch.device('cuda')

    # 导入模型： spend: 1020s
    now_times = time.time()
    model = CrossDomainNet(batch_size=batch_size, device=device)
    model.load_state_dict(torch.load('init_model.pt'))
    print("load model success - spend: {} s".format(time.time() - now_times))

    # 导入目标域测试集
    data = pd.read_csv("ctr_data/test/test_data_ads.csv")

    # 获取测试数据
    values_dick = target_feature_process(data, target_sparse_feature_names, target_varlen_sparse_feature_names,
                                         model.target_labelencoder_dict)
    x = [values_dick[feat] for feat in target_sparse_feature_names + target_varlen_sparse_feature_names]

    # 需要对数据进行扩维才能进行矩阵拼接
    for i in range(len(x)):
        if len(x[i].shape) == 1:
            x[i] = np.expand_dims(x[i], axis=1)
    X = np.concatenate(x, axis=-1)
    print('X:', X.shape)

    # 这里由于模型参数固定了batch_size，所以需要对测试数据补充，在最后再去除无用预测
    padding_nums = batch_size - len(X) % batch_size
    padding_colunms = X.shape[-1]
    padding_data = X[:padding_nums, :]      # 拿前面的数据进行补充
    test_X = np.concatenate([X, padding_data])
    print('test_X', test_X.shape)

    # 对训练数据构建dataloader
    test_tensor_data = Data.TensorDataset(
        torch.from_numpy(test_X)
    )
    test_loader = DataLoader(dataset=test_tensor_data, shuffle=False, batch_size=model.batch_size)

    # 预测(需要注意，这里是有冗余数据的，需要去除填充的数据)
    print('Start predicting...')
    y_pred = test_predict(model, test_loader, device)
    y_pred = y_pred[:-padding_nums]

    # 构建csv文件
    submission = pd.DataFrame()
    submission['log_id'] = data['log_id']
    submission['pctr'] = y_pred.astype(np.float32)
    submission.to_csv("submission.csv", index=False)
    print("create submission")
