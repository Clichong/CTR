import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

from model import CrossDomainNet
from utils.feature_deal import target_feature_process, source_feature_process,  \
    target_sparse_feature_names, target_varlen_sparse_feature_names


# 对目标域的数据添加源域数据函数 （待补充）
# 思路：训练阶段，动态根据user的用户ID来补充对应池化后的源域数据
# def target_add_source_data(x):
#     return x

best_auc = 0

# 进行一次训练的代码
def train_one_epoch(epoch, model, criterion, optimizer, train_loader, metrics, device, verbose=1):

    model.train()

    print('\nEpoch: %d' % epoch)
    loss_epoch = 0
    total_loss_epoch = 0
    aux_loss_epoch = 0

    epoch_logs = {}
    train_result = {}

    steps_per_epoch = len(train_loader)

    # 分批次训练
    for batch_idx, (x_train, y_train) in enumerate(train_loader):

        # 不需要在训练过程中进行，而是在模型中进行
        # x_train = target_add_source_data(x_train)

        x = x_train.to(device).float()
        y = y_train.to(device).float()

        # 总损失为: 交叉熵损失 + 正则化损失 + 辅助损失
        # y_pred = model(x).squeeze()
        y_pred, aux_pred, aux_label = model(x)
        y_pred = y_pred.squeeze()

        optimizer.zero_grad()

        # 损失计算
        loss = criterion(y_pred, y.squeeze(), reduction='mean')
        reg_loss = model.get_regularization_loss()

        # 三个目标联合训练
        # label_pred, clilabel_pred, pro_pred = aux_pred
        # label, clilabel, pro = aux_label[:, 0], aux_label[:, 1], aux_label[:, 2]
        # aux_loss = F.binary_cross_entropy(label_pred.squeeze(), label) + \
        #            F.binary_cross_entropy(clilabel_pred.squeeze(), clilabel) + \
        #            F.mse_loss(pro_pred.squeeze(), pro)

        # 两个目标联合训练
        label_pred, clilabel_pred = aux_pred
        label, clilabel = aux_label[:, 0], aux_label[:, 1]
        aux_loss = F.binary_cross_entropy(label_pred.squeeze(), label) + \
                   F.binary_cross_entropy(clilabel_pred.squeeze(), clilabel)

        # 总损失 = 目标域损失 + 0.5*源域辅助损失 + 正则化损失
        total_loss = loss + reg_loss + (aux_loss + model.aux_loss) * 0.5
        # total_loss = loss + reg_loss + aux_loss

        # 数值记录
        loss_epoch += loss.item()
        aux_loss_epoch += aux_loss.item()
        total_loss_epoch += total_loss.item()

        # loss.backward()
        total_loss.backward()
        optimizer.step()

        # print time = 5
        # if batch_idx % (steps_per_epoch // 5) == 0 or batch_idx == steps_per_epoch - 1:
        train_info = "epoch:{}, batch:{}/{}, loss:{}, aux_loss:{}, total_loss:{}" \
            .format(epoch + 1, batch_idx, steps_per_epoch,
                    loss_epoch / (batch_idx + 1), aux_loss_epoch / (batch_idx + 1), total_loss_epoch / (batch_idx + 1))
        print(train_info)

        if verbose > 0:
            # 遍历每种评价指标
            for name, metric_fun in metrics.items():
                # 首先分别为其创建一个列表
                if name not in train_result:
                    train_result[name] = []
                # 在对应指标的列表中添加每次的训练结果
                train_result[name].append(metric_fun(
                    y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))

    # 所有批次数据训练完成后，汇总最后结果在epoch_logs列表中
    epoch_logs["loss"] = loss_epoch / steps_per_epoch
    epoch_logs['total loss'] = total_loss_epoch / steps_per_epoch
    for name, result in train_result.items():
        epoch_logs[name] = np.sum(result) / steps_per_epoch

    if verbose > 0:
        train_info = str()
        for name in epoch_logs.keys():
            train_info += "{}: {}\n".format(name, epoch_logs[name])
        print(train_info)


# 进行一次验证的代码
def eval_one_epoch(epoch, model, val_loader, device):

    model.train()
    steps_per_epoch = len(val_loader)

    global best_auc
    pred_ans = []
    true_ans = []
    with torch.no_grad():
        for batch_idx, (x_val, y_val) in enumerate(val_loader):

            x = x_val.to(device).float()
            y = y_val.to(device).float()

            y_pred, _, _ = model(x)  # .squeeze()

            y_pred = y_pred.cpu().data.numpy()
            pred_ans.append(y_pred)
            true_ans.append(y)

            # print('finish: {} / {}'.format(batch_idx, steps_per_epoch))

    pred_ans = np.concatenate(pred_ans).astype("float64")
    true_ans = np.concatenate(true_ans).astype("float64")

    # 对预测值进行auc计算与logloss计算
    pred_ans = np.around(pred_ans, 6)
    val_logloss = round(log_loss(true_ans, pred_ans), 4)
    val_auc = round(roc_auc_score(true_ans, pred_ans), 4)
    print('-' * 30, 'Eval: epoch {}'.format(epoch), '-' * 30)
    print("val LogLoss", val_logloss)
    print("val AUC", val_auc)
    print('-' * 80)

    # 保留最高auc的模型参数
    if val_auc >= best_auc:
        print('Saving best model in Epoch: {}'.format(epoch))
        torch.save(model.state_dict(), 'best_model.pt')
        best_auc = val_auc


# 获取评价方式
def get_metrics(metrics, set_eps=False):
    metrics_ = {}
    if metrics:
        for metric in metrics:
            if metric == "binary_crossentropy" or metric == "logloss":
                metrics_[metric] = log_loss
            if metric == "auc":
                metrics_[metric] = roc_auc_score
            if metric == "mse":
                metrics_[metric] = mean_squared_error
            # if metric == "accuracy" or metric == "acc":
            #     metrics_[metric] = self._accuracy_score
            # self.metrics_names.append(metric)
    return metrics_


if __name__ == '__main__':

    epoch_size = 5
    batch_size = 4096
    SEED = 1024
    metrics = ['binary_crossentropy', 'auc']
    device = torch.device('cpu')

    # 后续操作可能会使用到模型的一些内置变量
    model = CrossDomainNet(batch_size=batch_size)
    torch.save(model.state_dict(), 'init_model.pt')

    data = pd.read_csv("ctr_data/train/train_data_ads.csv")
    # 获取标签数据
    y = data['label'].values
    # 获取训练数据（这里需要先编码 -> 再拼接，形成一个矩阵）
    values_dick = target_feature_process(data, target_sparse_feature_names, target_varlen_sparse_feature_names,
                                         model.target_labelencoder_dict)
    x = [values_dick[feat] for feat in target_sparse_feature_names + target_varlen_sparse_feature_names]
    # 需要对数据进行扩维才能进行矩阵拼接
    for i in range(len(x)):
        if len(x[i].shape) == 1:
            x[i] = np.expand_dims(x[i], axis=1)
    # 对列表x进行拼接
    X = np.concatenate(x, axis=-1)
    print('X:', X.shape)

    # 数据切分处理: 此时的数据都是还未embedding的数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # 对训练数据构建dataloader
    train_tensor_data = Data.TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train))
    train_loader = DataLoader(dataset=train_tensor_data, shuffle=True, batch_size=model.batch_size, drop_last=True)

    # 对验证数据构建dataloader
    val_tensor_data = Data.TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test))
    val_loader = DataLoader(dataset=val_tensor_data, shuffle=True, batch_size=model.batch_size, drop_last=True)

    # 设置相关优化器
    criterion = F.binary_cross_entropy
    optimizer = torch.optim.Adam(model.parameters())
    metrics = get_metrics(metrics)

    print('Start training...')
    for epoch in range(epoch_size):
        eval_one_epoch(epoch, model, val_loader, device)
        train_one_epoch(epoch, model, criterion, optimizer, train_loader, metrics, device)

    # 训练结束后保存模型
    torch.save(model.state_dict(), 'last_model.pt')

