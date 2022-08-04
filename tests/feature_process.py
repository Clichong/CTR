from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn

import copy
import pandas as pd
import numpy as np
from tqdm import tqdm

ads_dense_features = [
    'ad_click_list_v001',   # 用户点击广告任务id列表
    'ad_click_list_v002',   # 用户点击广告对应广告主id列表
    'ad_click_list_v003',   # 用户点击广告推荐应用列表
    'ad_close_list_v001',   # 用户关闭广告任务列表
    'ad_close_list_v002',   # 用户关闭广告对应广告主列表
    'ad_close_list_v003',   # 用户关闭广告推荐应用列表
]

ads_sparse_features = [
    'user_id',
    'age',
    'gender',
    'residence',
    'city',
    'city_rank',
    'series_dev',
    'series_group',
    'emui_dev',
    'device_name',
    'device_size',
    'net_type',
    'task_id',
    'adv_id',
    'creat_type_cd',
    'adv_prim_id',
    'inter_type_cd',
    'slot_id',
    'site_id',
    'spread_app_id',
    'hispace_app_tags',
    'app_second_class',
    'app_score',
    'pt_d',
    'u_newsCatInterestsST',  # 用户短时兴趣分类偏好
    'u_refreshTimes',        # 信息流日均有效刷新次数
    'u_feedLifeCycle'        # 信息流用户活跃度
]


# 稀疏特征处理
def ads_sparse_feature_process(dataframes, features=ads_sparse_features):

    all_data, data, test = copy.deepcopy(dataframes)
    sparse_feature_embeding_process = False

    # 依次处理每个稀疏特征
    for sparse_feature in tqdm(features, desc="process sparse features"):

        # 1) LabelEncoder编码处理
        lbe = LabelEncoder().fit(all_data[sparse_feature])
        # all_data[sparse_feature] = lbe.transform(all_data[sparse_feature])
        data[sparse_feature] = lbe.transform(data[sparse_feature])
        test[sparse_feature] = lbe.transform(test[sparse_feature])

        # 2) Embedding编码处理
        if sparse_feature_embeding_process:
            lookup_tabel_nums = len(all_data[sparse_feature].unique())
            embed = nn.Embedding(num_embeddings=lookup_tabel_nums, embedding_dim=1)

            # 对data数据处理
            tensorvalue_data = torch.from_numpy(data[sparse_feature].values)
            embedding_data = embed(tensorvalue_data).detach().numpy()
            data[sparse_feature] = embedding_data

            # 对test数据处理
            tensorvalue_test = torch.from_numpy(test[sparse_feature].values)
            embedding_test = embed(tensorvalue_test).detach().numpy()
            test[sparse_feature] = embedding_test

    return data, test


# 密集特征处理
def ads_dense_feature_process(dataframes, features=ads_dense_features):

    all_data, data, test = copy.deepcopy(dataframes)
    dense_feature_embedding_process = 'dr'  # choose 'sum' or 'dr'
    dense_feature_embedding_norm = False     # if normalize the embedding

    # 为与数据构建子特征的dataframe
    def get_dense_feature_dataframe(dataframe, feature):
        dense_feature_data = dataframe[feature].values.tolist()  # 转换为列表数据
        dense_feature_datalists = [data_list.split('^') for data_list in dense_feature_data]
        dense_feature_dataframe = pd.DataFrame(data=dense_feature_datalists)  # 切分后的数据构建成dataframe形式
        dense_feature_dataframe = dense_feature_dataframe.replace([None], ['0'])  # 用0来替换填充为None的数值
        return dense_feature_dataframe

    # 依次对每一个dense特征进行处理
    for dense_feature in tqdm(features, desc="process dense features"):
        # 为每一个dense特征进行切分，构造一个dataframe格式表
        # dense_feature_data = all_data[dense_feature].values.tolist()    # 转换为列表数据
        # dense_feature_datalists = [data_list.split('^') for data_list in dense_feature_data]
        # dense_feature_dataframe = pd.DataFrame(data=dense_feature_datalists)        # 切分后的数据构建成dataframe形式
        # dense_feature_dataframe = dense_feature_dataframe.replace([None], ['0'])    # 用0来替换填充为None的数值

        # 对训练集与测试集的dense特征分别构建dataframe进行处理
        dense_feature_dataframe = get_dense_feature_dataframe(dataframe=all_data, feature=dense_feature)
        dense_feature_dataframe_data = get_dense_feature_dataframe(dataframe=data, feature=dense_feature)
        dense_feature_dataframe_test = get_dense_feature_dataframe(dataframe=test, feature=dense_feature)

        # 对当前特征的dataframe进行embedding处理
        dense_feature_result_data, dense_feature_result_test = [], []
        dense_feature_dataframe_column = dense_feature_dataframe.columns.values     # column名称为数值
        for i in dense_feature_dataframe_column:
            # 1) 先进行labelencoder (数据类型由str变成int)
            lbe = LabelEncoder().fit(dense_feature_dataframe[i])
            dense_feature_dataframe[i] = lbe.transform(dense_feature_dataframe[i])
            dense_feature_dataframe_data[i] = lbe.transform(dense_feature_dataframe_data[i])    # data处理
            dense_feature_dataframe_test[i] = lbe.transform(dense_feature_dataframe_test[i])    # test处理

            # 2) 再进行embedding处理
            lookup_tabel_nums = len(dense_feature_dataframe[i].unique())
            dense_feature_embedding_dim = 2
            embed = nn.Embedding(num_embeddings=lookup_tabel_nums, embedding_dim=dense_feature_embedding_dim) # 建立查找表
            # dense_feature_dataframe = dense_feature_dataframe.astype(int)

            # data数据处理（处理前需要转为为tensor格式）
            dense_feature_dataframe_tensorvalue_data = torch.from_numpy(dense_feature_dataframe_data[i].values)
            dense_feature_dataframe_embedding_data = embed(dense_feature_dataframe_tensorvalue_data)
            dense_feature_dataframe_embedding_data = dense_feature_dataframe_embedding_data.detach().numpy()
            dense_feature_result_data.append(dense_feature_dataframe_embedding_data)    # 对embedding进行相加融合

            # test数据处理（处理前需要转为为tensor格式）
            dense_feature_dataframe_tensorvalue_test = torch.from_numpy(dense_feature_dataframe_test[i].values)
            dense_feature_dataframe_embedding_test = embed(dense_feature_dataframe_tensorvalue_test)
            dense_feature_dataframe_embedding_test = dense_feature_dataframe_embedding_test.detach().numpy()
            dense_feature_result_test.append(dense_feature_dataframe_embedding_test)  # 对embedding进行相加融合

        # dense特征横向拼接处理
        dense_feature_result_data = np.stack(dense_feature_result_data, axis=1).reshape(len(dense_feature_result_data[0]), -1)
        dense_feature_result_test = np.stack(dense_feature_result_test, axis=1).reshape(len(dense_feature_result_test[0]), -1)

        # 对当前密集特征embed的处理方式 (1.sum方式: 直接相加融合； 2.dr方式: 拼接后降维处理)
        if dense_feature_embedding_process == 'sum':
            data[dense_feature] = dense_feature_result_data.sum(axis=1)
            test[dense_feature] = dense_feature_result_test.sum(axis=1)
        elif dense_feature_embedding_process == 'dr':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1).fit(dense_feature_result_data)
            data[dense_feature] = pca.transform(dense_feature_result_data)
            test[dense_feature] = pca.transform(dense_feature_result_test)

    # 最后进行归一化处理
    if dense_feature_embedding_norm:
        from sklearn.preprocessing import MinMaxScaler
        all_data = pd.concat([data, test])
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(all_data[ads_dense_features])
        data[ads_dense_features] = scaler.transform(data[ads_dense_features])
        test[ads_dense_features] = scaler.transform(test[ads_dense_features])

    return data, test


if __name__ == '__main__':
    data = pd.read_csv("../ctr_data/train/train_data_ads.csv")
    test = pd.read_csv('../ctr_data/test/test_data_ads.csv')
    all_data = pd.concat([data, test])

    # 构建一个新特征表
    data, test = ads_dense_feature_process(dataframes=[all_data, data, test], features=ads_dense_features)
    data, test = ads_sparse_feature_process(dataframes=[all_data, data, test], features=ads_sparse_features)
    data.to_csv("train_newdata_ads.csv", index=False)
    test.to_csv("test_newdata_ads.csv", index=False)