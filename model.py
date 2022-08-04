import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
from collections import OrderedDict

from utils.feature_deal import get_embedding_mask_lbe_dict, build_feature_index, source_feature_process, \
    target_varlen_sparse_feature_names, target_sparse_feature_names, \
    user_info_feature_names, target_ad_feature_names, interact_ad_feature_names, content_info_feature_names, \
    source_sparse_feature_names, source_varlen_sparse_feature_names, \
    SPARSE_EMBEDDING_DIM, VARLEN_SPARSE_EMBEDDING_DIM, VARLEN_SPARSE_MAXLEN

from deepctr_torch.layers import FM, DNN


class Interest_Level_Attention(nn.Module):

    def __init__(self, colunms_dim_list, batch_size):
        super().__init__()

        self.b = batch_size
        self.part_nums = len(colunms_dim_list)
        self.colunms_dim_sum = np.array(colunms_dim_list).sum()

        # 最简单的做法是为每个部分赋予一个权重，但是这样就没有结合信息
        # self.interset_attention_weight = nn.Parameter(torch.ones(part_nums))
        self.V_params = nn.ParameterList([nn.Parameter(torch.Tensor(self.colunms_dim_sum, 1))
                                          for i in range(self.part_nums)])
        self.g_params = nn.ParameterList([nn.Parameter(torch.Tensor(1, self.b)) for i in range(self.part_nums)])
        self.b_params = nn.ParameterList([nn.Parameter(torch.Tensor(1)) for i in range(self.part_nums)])

        self.init_weight()

    def forward(self, X):
        # X_names：[user_info_part, target_ad_part, interact_ad_part, content_info_part]
        # X: [(b, p1_dims), (b, p2_dims), ...]
        cat_data = torch.cat(X, dim=-1)
        weight = [(torch.mm(self.g_params[i], F.relu(torch.mm(cat_data, self.V_params[i]))) + self.b_params[i]).exp()
                  for i in range(self.part_nums)]
        out = [weight[i] * X[i] for i in range(self.part_nums)]
        out = torch.cat(out, dim=-1)

        return out

    def init_weight(self, init_std=0.0001):
        for weight in self.V_params:
            torch.nn.init.normal_(weight, mean=0, std=init_std)
        for weight in self.g_params:
            torch.nn.init.normal_(weight, mean=0, std=init_std)
        for weight in self.b_params:
            torch.nn.init.normal_(weight, mean=0, std=init_std)


class PredictionLayer(nn.Module):

    def __init__(self, task='sigmoid', use_bias=True, **kwargs):

        super(PredictionLayer, self).__init__()
        self.use_bias = use_bias
        self.task = task
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1,)))

        self.init_weight()

    def forward(self, X):
        output = X
        if self.use_bias:
            output += self.bias
        if self.task == "sigmoid":
            output = torch.sigmoid(output)
        elif self.task == 'tanh':
            output = torch.tanh(output)
        elif self.task == 'relu':
            output = torch.relu(output)

        return output

    def init_weight(self, init_std=0.0001):
        if self.use_bias:
            torch.nn.init.normal_(self.bias, mean=0, std=init_std)


class Source_Linear(nn.Module):

    def __init__(self, input_dim, out_dim, activation='tanh'):
        super().__init__()

        assert activation in ['tanh', 'relu'], "No such activation choose, activation must be in ['tanh', 'relu']"

        self.ln = nn.Linear(input_dim, out_dim, bias=False)
        if activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'relu':
            self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.ln(x))


class CrossDomainNet(nn.Module):

    def __init__(self,
                 sparse_embedding_dim=SPARSE_EMBEDDING_DIM,
                 varlen_sparse_embedding_dim=VARLEN_SPARSE_EMBEDDING_DIM,
                 varlen_sparse_maxlen=VARLEN_SPARSE_MAXLEN,
                 dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001,
                 dnn_dropout=0.6, dnn_use_bn=True, dnn_activation='relu', device='cpu',
                 batch_size=4096
                 ):
        super().__init__()

        self.batch_size = batch_size

        # 编码长度与最长编码数量
        self.sparse_embedding_dim = sparse_embedding_dim
        self.varlen_sparse_embedding_dim = varlen_sparse_embedding_dim
        self.varlen_sparse_maxlen = varlen_sparse_maxlen

        # 目标域的相关特征分组
        self.target_sparse_colunms = target_sparse_feature_names
        self.target_varlen_sparse_colunms = target_varlen_sparse_feature_names
        self.target_user_info_columns = user_info_feature_names
        self.target_target_ad_columns = target_ad_feature_names
        self.target_interact_ad_colunms = interact_ad_feature_names
        self.target_content_info_colunms = content_info_feature_names

        # 源域的相关特征分组
        self.source_sparse_colunms = source_sparse_feature_names
        self.source_varlen_sparse_colunms = source_varlen_sparse_feature_names

        # 导入目标域数据以进行编码
        target_train = pd.read_csv("ctr_data/train/train_data_ads.csv")
        target_test = pd.read_csv('ctr_data/test/test_data_ads.csv')
        target_all_data = pd.concat([target_train, target_test])

        # 获取目标域各特征数据的查找表以及对应的掩码矩阵
        self.target_embedding_dict, self.target_labelencoder_dict = \
            get_embedding_mask_lbe_dict(target_all_data, self.target_sparse_colunms, self.target_varlen_sparse_colunms)
        # 获取数据的特征索引，来更好的处理数据
        self.target_feature_index = build_feature_index(self.target_sparse_colunms, self.target_varlen_sparse_colunms)

        del target_train, target_test, target_all_data

        # 导入源域诗句以进行编码
        source_train = pd.read_csv('ctr_data/train/train_data_feeds_drop.csv')
        source_test = pd.read_csv('ctr_data/test/test_data_feeds_drop.csv')
        source_all_data = pd.concat([source_train, source_test])

        # 获取源域各特征数据的查找表以及对应的掩码矩阵
        self.source_embedding_dict, self.source_labelencoder_dict = \
            get_embedding_mask_lbe_dict(source_all_data, self.source_sparse_colunms, self.source_varlen_sparse_colunms)
        self.source_labelencoder_dict['u_userId'] = self.target_labelencoder_dict['user_id']
        # 获取源域数据的特征索引
        self.source_feature_index = build_feature_index(self.source_sparse_colunms, self.source_varlen_sparse_colunms)

        # 对源域的数据集进行处理（返回的dataframe包含label标签与用户ID）
        self.source_train_dataframe = source_feature_process(data=source_train,
                                                             sparse_columns=self.source_sparse_colunms,
                                                             varlen_sparse_columns=self.source_varlen_sparse_colunms,
                                                             labelencoder_dict=self.source_labelencoder_dict)
        self.source_test_dataframe = source_feature_process(data=source_test,
                                                            sparse_columns=self.source_sparse_colunms,
                                                            varlen_sparse_columns=self.source_varlen_sparse_colunms,
                                                            labelencoder_dict=self.source_labelencoder_dict)

        del source_train, source_test, source_all_data

        # 注意力层
        self.colunms_dim_list = [110, 120, 300, 80, 440]   # 各部分编码后的特征长度
        self.interest_attention = Interest_Level_Attention(self.colunms_dim_list, self.batch_size)

        # 神经网络层: 通过compute_input_dim函数计算出全连接的第一层输入
        self.dnn = DNN(inputs_dim=np.array(self.colunms_dim_list).sum(),
                       hidden_units=dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                       init_std=init_std, device=device)
        self.dnn_linear = nn.Linear(
            dnn_hidden_units[-1], 1, bias=False).to(device)
        self.out = PredictionLayer(task='sigmoid')

        # 源域单独训练的网络设置
        self.source_dnn = DNN(inputs_dim=450,
                              hidden_units=dnn_hidden_units,
                              activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                              init_std=init_std, device=device)
        # self.source_label_clf = Source_Linear(input_dim=dnn_hidden_units[-1], out_dim=1, activation='tanh')
        # self.source_cillabel_clf = Source_Linear(input_dim=dnn_hidden_units[-1], out_dim=1, activation='tanh')
        # self.source_pro_reg = Source_Linear(input_dim=dnn_hidden_units[-1], out_dim=1, activation='relu')

        # 需要配合PredictionLayer设置，直接集成nn.Module会报错
        self.source_label_clf = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        self.source_label_out = PredictionLayer(task='sigmoid')
        self.source_cillabel_clf = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        self.source_cillabel_out = PredictionLayer(task='sigmoid')
        # self.source_pro_reg = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        # self.source_pro_out = PredictionLayer(task='relu')

        # 添加辅助损失：regularization_loss + aux_loss
        self.aux_loss = torch.zeros((1,))
        self.regularization_weight = []
        self.add_regularization_weight(self.source_embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(self.target_embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.source_dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self.add_regularization_weight(self.source_label_clf.weight, l2=l2_reg_dnn)
        self.add_regularization_weight(self.source_cillabel_clf.weight, l2=l2_reg_dnn)
        # self.add_regularization_weight(self.source_pro_reg.weight, l2=l2_reg_dnn)

    def forward(self, X):

        b = len(X)

        # 获取X的源域已编码数据（训练过程与预测过程需要分开，训练过程可以利用上源域的label标签）
        unique_sampleID = np.unique(X[:, 0].detach().numpy())
        if self.training:
            source_supple_batchdata = self.source_train_dataframe[self.source_train_dataframe[0].isin(unique_sampleID)]
        else:
            source_supple_batchdata = self.source_test_dataframe[self.source_test_dataframe[0].isin(unique_sampleID)]

        # 源域的标签信息
        aux_label = torch.tensor(source_supple_batchdata.values[:, -3:]).float()
        # source_label = source_supple_batchdata.values[:, -3]
        # source_clilabel = torch.tensor(source_supple_batchdata.values[:, -2])
        # source_pro = torch.tensor(source_supple_batchdata.values[:, -1])

        # 获取当前筛选出的源域批次数据，用来构建源域损失
        source_embedding_batchdata = self.get_source_embedding_batchdata(source_supple_batchdata)

        # 根据批数据的embedding编码表获取一个按顺序的融合矩阵
        source_fuse_batchdata = self.get_source_fuse_batchdata(target_X=X, unique_ID=unique_sampleID,
                                                               source_raw_data=source_supple_batchdata,
                                                               source_embedding_data=source_embedding_batchdata)

        # 这里需要动态的对X进行构建掩码表，然后动态的使用编码后的X数据与掩码进行相乘
        # target_mask_dict = self.get_target_mask_dick(X)
        target_mask_dict = self.get_mask_dick(X, varlen_sparse_colunms=self.target_varlen_sparse_colunms,
                                              feature_index=self.target_feature_index,
                                              embedding_dict=self.target_embedding_dict)

        # 输入的x是目标域的训练拼接特征，需要根据index进行对应的embedding来计算, 这里需要对特征数据区分为4个模块处理
        target_sparse_embedding_list = self.get_sparse_embedding_list(X, self.target_sparse_colunms,
                                                                      self.target_varlen_sparse_colunms,
                                                                      feature_index=self.target_feature_index,
                                                                      embedding_dict=self.target_embedding_dict)

        # 获取4个模块编码处理后矩阵, 根据编码表与掩码表来对输入进行编码与相关计算
        # 需要注意，如果划分的特征中包含了变长稀疏特征，需要额外进行掩码操作
        target_sparse_part_list = self.target_sparse_embedding_process(target_sparse_embedding_list,
                                                                       target_mask_dict, batch_size=b)
        target_sparse_part_list.append(source_fuse_batchdata[:, self.sparse_embedding_dim:])   # 去除user_id特征

        # 对5个模块部分进行interest_attention处理：目标域的4个部分 + 源域的补充信息
        X_interext_attention = self.interest_attention(target_sparse_part_list)

        # 目标域的预测输出
        dnn_output = self.dnn(X_interext_attention)
        dnn_logit = self.dnn_linear(dnn_output)
        y_pred = self.out(dnn_logit)

        # 验证过程
        if self.training is False:
            return y_pred

        # 训练过程
        else:
            # 源域的预测输出
            sdnn_output = self.source_dnn(source_embedding_batchdata)

            # 进行多目标预测
            source_label_pred = self.source_label_out(self.source_label_clf(sdnn_output))
            source_clilabel_pred = self.source_cillabel_out(self.source_cillabel_clf(sdnn_output))
            # source_pro_pred = self.source_pro_out(self.source_pro_reg(sdnn_output))

            # 辅助预测数据以元祖形式返回
            # aux_pred = (source_label_pred, source_clilabel_pred, source_pro_pred)
            aux_pred = (source_label_pred, source_clilabel_pred)

            return y_pred, aux_pred, aux_label

    # 首先构建ID的编码字典，然后根据目标域的ID依次从编码字典中取出对应数值，获取最终的源域数据
    def get_source_fuse_batchdata(self, target_X, unique_ID, source_raw_data, source_embedding_data):

        id_embedding_dict = OrderedDict()

        # 传入的source_raw_data是一个dataframe格式
        source_raw_data = torch.tensor(source_raw_data.values)

        # 构建一个ID索引编码字典
        for id in unique_ID:
            # 根据ID获取源域对应数据ID的全部索引
            id_index = torch.where(source_raw_data[:, 0] == id)[0]

            # 根据对应的全部索引ID数据进行融合
            embedding_array = torch.index_select(source_embedding_data, dim=0, index=id_index)
            embedding_pooling = embedding_array.mean(dim=0)     # 此处的融合步骤可以由其他attention替代

            # 将当前的ID信息存在在一个字典中
            id_embedding_dict[id] = embedding_pooling

        # 接着就是根据当前的目标域数据ID获取源域的融合特征
        # int(row_data[0]): 每行的第1个数据是用户的ID
        source_fuse_list = [id_embedding_dict[int(row_data[0])].unsqueeze(0) for row_data in target_X]
        source_fuse_data = torch.cat(source_fuse_list, dim=0)

        return source_fuse_data

    # 获取来着源域的最后拼接矩阵
    def get_source_embedding_batchdata(self, batchdata):

        batchdata_len = len(batchdata)
        batchdata = torch.tensor(batchdata.values)

        # 首先根据一个批次在dataframe筛选出来的数据获取其变长特征的掩码
        batchdata_mask_dict = self.get_mask_dick(batchdata, self.source_varlen_sparse_colunms,
                                                 feature_index=self.source_feature_index,
                                                 embedding_dict=self.source_embedding_dict)

        # 获取特征的编码表示
        batchdata_embedding_list = self.get_sparse_embedding_list(batchdata, self.source_sparse_colunms,
                                                                  self.source_varlen_sparse_colunms,
                                                                  feature_index=self.source_feature_index,
                                                                  embedding_dict=self.source_embedding_dict)

        # 变成特征进行编码与掩码相乘，获得最后的embedding表示
        for feat in self.source_varlen_sparse_colunms:
            batchdata_embedding_list[feat] = batchdata_embedding_list[feat] * batchdata_mask_dict[feat]

        # 接着对其进行维度改变，然后拼接在一起，作为一个batch的ID索引表
        for feat in self.source_sparse_colunms + self.source_varlen_sparse_colunms:
            batchdata_embedding_list[feat] = batchdata_embedding_list[feat].reshape(batchdata_len, -1)

        # 数据的拼接处理
        batchdata_embedding_part = [batchdata_embedding_list[feat] for feat in
                                    self.source_sparse_colunms + self.source_varlen_sparse_colunms]
        batchdata_embedding_part = torch.cat(batchdata_embedding_part, dim=-1)

        return batchdata_embedding_part

    # 计算Linear的输入维度
    def compute_input_dim(self, sparse_colunms, varlen_sparse_colunms):
        return len(sparse_colunms) * self.sparse_embedding_dim + \
               len(varlen_sparse_colunms) * self.varlen_sparse_maxlen * self.varlen_sparse_embedding_dim

    # 返回掩码
    def sequence_mask(self, lengths, maxlen=None, dtype=torch.bool):
        # Returns a mask tensor representing the first N positions of each cell.
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix
        mask.type(dtype)
        return mask

    # 动态根据输入X，配合faeture_index来构建掩码字典，与后续编码后的数值相乘
    def get_target_mask_dick(self, X):
        target_mask_dict = OrderedDict()
        for feat in self.target_varlen_sparse_colunms:
            # 根据feature_index找到相关的变长特征编码值
            feat_value = X[:, self.target_feature_index[feat][0]:self.target_feature_index[feat][1]]
            seq_length = feat_value.count_nonzero(dim=1)  # 统计每一行的非0个数
            seq_length = seq_length.unsqueeze(-1)  # (batch_size, ) -> (batch_size, 1)
            # 获取掩码后，需要拓展到embedding_dim的维度 (这里直接固定编码长度)
            mask = self.sequence_mask(seq_length, maxlen=self.varlen_sparse_maxlen)
            # (batch_size, 1, maxlen) -> (batch_size, maxlen, 1) -> batch_size, maxlen, embedding_dim)
            mask = torch.transpose(mask, 1, 2)
            embedding_size = self.target_embedding_dict[feat].embedding_dim
            mask = torch.repeat_interleave(mask, embedding_size, dim=2)
            target_mask_dict[feat] = mask

        return target_mask_dict

    # 获取源域的特征掩码
    def get_mask_dick(self, X, varlen_sparse_colunms, feature_index, embedding_dict):
        mask_dict = OrderedDict()
        for feat in varlen_sparse_colunms:
            # 根据feature_index找到相关的变长特征编码值
            feat_value = X[:, feature_index[feat][0]:feature_index[feat][1]]
            seq_length = feat_value.count_nonzero(dim=1)  # 统计每一行的非0个数
            seq_length = seq_length.unsqueeze(-1)  # (batch_size, ) -> (batch_size, 1)
            # 获取掩码后，需要拓展到embedding_dim的维度 (这里直接固定编码长度)
            mask = self.sequence_mask(seq_length, maxlen=self.varlen_sparse_maxlen)
            # (batch_size, 1, maxlen) -> (batch_size, maxlen, 1) -> batch_size, maxlen, embedding_dim)
            mask = torch.transpose(mask, 1, 2)
            embedding_size = embedding_dict[feat].embedding_dim
            mask = torch.repeat_interleave(mask, embedding_size, dim=2)
            mask_dict[feat] = mask

        return mask_dict

    # 构建稀疏特征的编码后的结果，方便分组计算
    def get_sparse_embedding_list(self, X, sparse_colunms, varlen_sparse_colunms, feature_index, embedding_dict):
        sparse_embedding_list = OrderedDict()
        for feat in sparse_colunms + varlen_sparse_colunms:
            sparse_embedding_list[feat] = embedding_dict[feat](
                X[:, feature_index[feat][0]:feature_index[feat][1]].long()
            )
        return sparse_embedding_list

    # 详细的编码处理
    def target_sparse_embedding_process(self, sparse_embedding_list, target_mask_dict, batch_size):

        # 获取不同模块编码后的特征列表数据：[(batch_size, 1, embedding_dim), ...] / [(batch_size, 5, embedding_dim), ...]
        user_info_part = [sparse_embedding_list[feat]
                          for feat in self.target_user_info_columns]
        target_ad_part = [sparse_embedding_list[feat]
                          for feat in self.target_target_ad_columns]
        interact_ad_part = [sparse_embedding_list[feat] * target_mask_dict[feat]
                            for feat in self.target_interact_ad_colunms]
        content_info_part = []
        for feat in self.target_content_info_colunms:
            if feat == 'u_newsCatInterestsST':
                content_info_part.append(
                    (sparse_embedding_list[feat] * target_mask_dict[feat]).reshape(batch_size, -1))
            else:
                content_info_part.append(sparse_embedding_list[feat].reshape(batch_size, -1))

        # 对数据进行concat操作：(batch_size, 1, embedding_dim_sum) -> (batch_size, embedding_dim_sum)
        user_info_part = torch.cat(user_info_part, dim=-1).reshape(batch_size, -1)
        target_ad_part = torch.cat(target_ad_part, dim=-1).reshape(batch_size, -1)
        interact_ad_part = torch.cat(interact_ad_part, dim=-1).reshape(batch_size, -1)
        content_info_part = torch.cat(content_info_part, dim=-1)

        return [user_info_part, target_ad_part, interact_ad_part, content_info_part]

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self):
        total_reg_loss = torch.zeros((1,))
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss


if __name__ == '__main__':

    model = CrossDomainNet()
    x = torch.rand([4092, 200])
    out = model(x)
    print(out.shape)