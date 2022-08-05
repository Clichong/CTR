from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn

import copy
import pandas as pd
import numpy as np

from tqdm import tqdm
from collections import OrderedDict

# 对源域数据特征进行切分为4个类别: 用户本身特征，目标广告特征，历史广告交互特征，其他特征
user_info_feature_names = ['user_id', 'age', 'gender', 'residence', 'city', 'city_rank',
                           'series_dev', 'series_group', 'emui_dev', 'device_name', 'device_size']
target_ad_feature_names = ['net_type', 'task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'inter_type_cd',
                           'slot_id', 'site_id', 'spread_app_id', 'hispace_app_tags', 'app_second_class', 'app_score']
interact_ad_feature_names = ['ad_click_list_v001', 'ad_click_list_v002', 'ad_click_list_v003',
                             'ad_close_list_v001', 'ad_close_list_v002', 'ad_close_list_v003']
content_info_feature_names = ['pt_d', 'u_newsCatInterestsST', 'u_refreshTimes',  'u_feedLifeCycle']

# 目标域的稀疏特征(sparse_feature)与变长特征(varlen_sparse_feature)
target_sparse_feature_names = ['user_id', 'age', 'gender', 'residence', 'city',
                               'city_rank', 'series_dev', 'series_group', 'emui_dev', 'device_name','device_size',
                               'net_type', 'task_id', 'adv_id', 'creat_type_cd',
                               'adv_prim_id', 'inter_type_cd', 'slot_id', 'site_id', 'spread_app_id',
                               'hispace_app_tags', 'app_second_class', 'app_score',
                               'pt_d', 'u_refreshTimes', 'u_feedLifeCycle']
target_varlen_sparse_feature_names = ['ad_click_list_v001', 'ad_click_list_v002', 'ad_click_list_v003',
                                      'ad_close_list_v001', 'ad_close_list_v002', 'ad_close_list_v003',
                                      'u_newsCatInterestsST']

# 源域的稀疏特征(sparse_feature)与变长特征(varlen_sparse_feature)
source_sparse_feature_names = ['u_userId', 'u_phonePrice', 'u_browserLifeCycle', 'u_browserMode', 'u_feedLifeCycle',
                               'u_refreshTimes', 'i_docId', 'i_s_sourceId', 'i_regionEntity', 'i_cat',
                               'i_dislikeTimes', 'i_upTimes', 'i_dtype', 'e_ch', 'e_m', 'e_po', 'e_pl',
                               'e_rn', 'e_section', 'e_et']
source_varlen_sparse_feature_names = ['u_newsCatInterests', 'u_newsCatDislike', 'u_newsCatInterestsST',
                                      'u_click_ca2_news', 'i_entities']

# 一些变量设置
SPARSE_EMBEDDING_DIM = 10
VARLEN_SPARSE_EMBEDDING_DIM = 10
VARLEN_SPARSE_MAXLEN = 5


# 对数据的第几列构建索引
# 这里默认稀疏数据的维度是1列，变长数据的维度是5列
def build_feature_index(sparse_columns, varlen_sparse_columns):

    features = OrderedDict()
    start = 0

    # 依次对稀疏数据与变长数据进行处理
    for feat in sparse_columns:
        features[feat] = (start, start + 1)
        start += 1
    for feat in varlen_sparse_columns:
        features[feat] = (start, start + VARLEN_SPARSE_MAXLEN)
        start += VARLEN_SPARSE_MAXLEN

    return features


# 对源域数据进行处理
def source_feature_process(data, sparse_columns, varlen_sparse_columns, labelencoder_dict):

    values_dick = OrderedDict()

    # 为列表数据构建子特征的dataframe
    def get_varlen_sparse_feature_dataframe(dataframe, feature):
        varlen_sparse_feature_data = dataframe[feature].values  # 获取列表数据
        varlen_sparse_feature_datalists = [values_list.split('^') for values_list in varlen_sparse_feature_data]
        # 切分后的数据构建成dataframe形式，并用0来替换填充为None的数值
        varlen_sparse_feature_dataframe = pd.DataFrame(data=varlen_sparse_feature_datalists).replace([None], ['0'])
        return varlen_sparse_feature_dataframe

    # 存储稀疏特征值
    for feat in sparse_columns:
        values_dick[feat] = labelencoder_dict[feat].fit_transform(data[feat])

    # 存储变长稀疏特征值
    for feat in varlen_sparse_columns:
        feat_dataframe = get_varlen_sparse_feature_dataframe(data, feat)
        feat_encoder_values = labelencoder_dict[feat].fit_transform(feat_dataframe.values.flatten())
        # 多值变长特征重新构建矩阵
        values_dick[feat] = np.array(feat_encoder_values).reshape(-1, 5)

    # 源域数据中含有多个监督信息，这里用于将其存储在字典中
    supervise_features = ['label', 'cillabel', 'pro']
    for feat in supervise_features:
        values_dick[feat] = data[feat].values

    # 将其进行维度扩增然后拼接成一个array矩阵
    source_datalist = [values_dick[feat] for feat in sparse_columns + varlen_sparse_columns + supervise_features]
    for i in range(len(source_datalist)):
        if len(source_datalist[i].shape) == 1:
            source_datalist[i] = np.expand_dims(source_datalist[i], axis=1)
    source_datalist = np.concatenate(source_datalist, axis=-1)

    # 构建成dataframe以便与索引
    source_dataframe = pd.DataFrame(data=source_datalist)

    return source_dataframe


# 对目标域的数据进行编码后输入：根据编码表编码数据，构建数值字典待后续拼接
# 思路：由于最后的输入是一个矩阵形式进行训练，所以需要对原始数据进行编码，同时对变长特征进行处理
# 处理完的特征构建成一个字典形式，在后续方便进行拼接操作
def target_feature_process(data, sparse_columns, varlen_sparse_columns, labelencoder_dict):

    values_dick = OrderedDict()

    # 为列表数据构建子特征的dataframe
    def get_varlen_sparse_feature_dataframe(dataframe, feature):
        varlen_sparse_feature_data = dataframe[feature].values  # 获取列表数据
        varlen_sparse_feature_datalists = [values_list.split('^') for values_list in varlen_sparse_feature_data]
        # 切分后的数据构建成dataframe形式，并用0来替换填充为None的数值
        varlen_sparse_feature_dataframe = pd.DataFrame(data=varlen_sparse_feature_datalists).replace([None], ['0'])
        return varlen_sparse_feature_dataframe

    # 存储稀疏特征值
    for feat in sparse_columns:
        values_dick[feat] = labelencoder_dict[feat].fit_transform(data[feat])

    # 存储变长稀疏特征值
    for feat in varlen_sparse_columns:
        feat_dataframe = get_varlen_sparse_feature_dataframe(data, feat)
        feat_encoder_values = labelencoder_dict[feat].fit_transform(feat_dataframe.values.flatten())
        # 多值变长特征重新构建矩阵
        #     for i in range(5):
        #         feat_dataframe[i] = feat_encoder_values[i::5]
        values_dick[feat] = np.array(feat_encoder_values).reshape(-1, 5)

    return values_dick


# 构建查找表、掩码表、编码表
def get_embedding_mask_lbe_dict(data, sparse_columns, varlen_sparse_columns, device='cpu'):

    unique_table = data.nunique()
    # mask_dick = OrderedDict()
    labelencoder_dick = OrderedDict()

    # 为列表数据构建子特征的dataframe
    def get_varlen_sparse_feature_dataframe(dataframe, feature):
        varlen_sparse_feature_data = dataframe[feature].values        # 获取列表数据
        varlen_sparse_feature_datalists = [values_list.split('^') for values_list in varlen_sparse_feature_data]
        # 切分后的数据构建成dataframe形式，并用0来替换填充为None的数值
        varlen_sparse_feature_dataframe = pd.DataFrame(data=varlen_sparse_feature_datalists).replace([None], ['0'])
        return varlen_sparse_feature_dataframe

    # 返回掩码
    # def _sequence_mask(lengths, maxlen=None, dtype=torch.bool):
    #     # Returns a mask tensor representing the first N positions of each cell.
    #     if maxlen is None:
    #         maxlen = lengths.max()
    #     row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
    #     matrix = torch.unsqueeze(lengths, dim=-1)
    #     mask = row_vector < matrix
    #     mask.type(dtype)
    #     return mask

    # 对稀疏特征进行编码处理
    for feat in sparse_columns:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
        # 保存编码表用来对数据编码
        labelencoder_dick[feat] = lbe

    # 对稀疏特征构建Embedding查找表
    embedding_dict = nn.ModuleDict(
        {feat: nn.Embedding(unique_table[feat], embedding_dim=SPARSE_EMBEDDING_DIM)
         for feat in sparse_columns}
    )

    # 对变长特征进行编码处理
    for feat in varlen_sparse_columns:
        feat_dataframe = get_varlen_sparse_feature_dataframe(data, feat)
        # 对数据现在展平编码, 补充Embedding查找表
        lbe = LabelEncoder()
        feat_encoder_values = lbe.fit_transform(feat_dataframe.values.flatten())
        unique_table[feat] = len(lbe.classes_)      # 更新唯一数值表
        embedding_dict[feat] = nn.Embedding(unique_table[feat], embedding_dim=VARLEN_SPARSE_EMBEDDING_DIM)

        # 保存编码表用来对数据编码
        labelencoder_dick[feat] = lbe

        # 对原始数据构建掩码表, 因为存在0的部分
        # seq_length = list(map(lambda x: [x.count('^') + 1], data[feat].values))
        # seq_length = torch.tensor(seq_length)
        # mask = _sequence_mask(seq_length)
        # mask = torch.transpose(mask, 1, 2)
        # embedding_size = embedding_dict[feat].embedding_dim
        # mask = torch.repeat_interleave(mask, embedding_size, dim=2)
        # mask_dick[feat] = mask

    # 对embedding进行初始化处理
    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=0.0001)

    return embedding_dict.to(device), labelencoder_dick


# 过来源域数据中在目标域没有出现过的用户ID
def source_data_filter():

    # 导入目标域的训练集与训练集
    target_train = pd.read_csv('../ctr_data/train/train_data_ads.csv')
    target_test = pd.read_csv('../ctr_data/test/test_data_ads.csv')
    # target_all_data = pd.concat([target_train, target_test])

    # 导入源域的训练集与训练集
    source_train = pd.read_csv('../ctr_data/train/train_data_feeds.csv')
    source_test = pd.read_csv('../ctr_data/test/test_data_feeds.csv')

    # 筛选源域训练集数据
    target_train_userid = target_train['user_id'].unique()
    source_train = source_train[source_train['u_userId'].isin(target_train_userid)]

    # 筛选源域测试集数据
    target_test_userid = target_test['user_id'].unique()
    source_test = source_test[source_test['u_userId'].isin(target_test_userid)]

    # 经过数据分析，发现源域中i_entities特征有缺失值，现用0来填充
    feat = 'i_entities'
    source_train[feat][source_train[feat].isna()] = 0
    source_test[feat][source_test[feat].isna()] = 0

    # 标签将-1转换为0
    label = ['label', 'cillabel']
    for feat in label:
        source_train[feat][source_train[feat] == -1] = 0
        source_test[feat][source_test[feat] == -1] = 0

    # 保存csv文件
    source_train.to_csv('../ctr_data/train/train_data_feeds_drop.csv', index=False)
    source_test.to_csv('../ctr_data/test/test_data_feeds_drop.csv', index=False)


if __name__ == '__main__':

    source_data_filter()

    # TEST_TARGET = -1
    #
    # if TEST_TARGET:
    #     target_train = pd.read_csv("../ctr_data/train/train_data_ads.csv")
    #     target_test = pd.read_csv('../ctr_data/test/test_data_ads.csv')
    #     target_all_data = pd.concat([target_train, target_test])
    #     print('load dataset success')
    #
    #     embedding_dict, labelencoder_dick = \
    #         get_embedding_mask_lbe_dict(target_all_data, target_sparse_feature_names, target_varlen_sparse_feature_names)
    #     print(embedding_dict, labelencoder_dick)
    #
    # elif TEST_TARGET == 0:
    #     source_train = pd.read_csv('../ctr_data/train/train_data_feeds_drop.csv')
    #     source_test = pd.read_csv('../ctr_data/test/test_data_feeds_drop.csv')
    #     source_all_data = pd.concat([source_train, source_test])
    #     print('load dataset success')
    #
    #     embedding_dict, labelencoder_dick = \
    #         get_embedding_mask_lbe_dict(source_all_data, source_sparse_feature_names,
    #                                     source_varlen_sparse_feature_names)
    #     print(embedding_dict, labelencoder_dick)
    #
    # else:
    #     print('just process source data')


