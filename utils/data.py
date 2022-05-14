import pandas as pd
import numpy as np
import os
import nltk
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import random
import pickle
import multiprocessing
import re
import pickle

from collections import Counter
from itertools import chain
from datetime import datetime

from torchvision import transforms
import torch
from sklearn import metrics
import re

#####################################
# log文件切分
# 处理后数据的主键为 sn + fault_time + label !!!

# sn分组后，按照最近邻+时间间隔划分日志数据
def divideLogByNearestTime(log_label_df: pd.DataFrame):
    log_correspond_label_df = pd.DataFrame(columns=['sn', 'fault_time', 'msg', 'time', 'server_model', 'label'])
    no_label_log_list = []
    # 不设置截断时间，效果最好
    cutoff = 10 * 3600

    for sn, log in log_label_df.groupby('sn'):
        if len(log[log['label'] != '']) == 0:
            no_label_log_list.append(log)
        elif len(log[log['label'] != '']) == 1:
            # 没有日志的标签，直接将标签的可用信息作为日志
            if len(log) == 1:
                msg_df = log
                msg_df['fault_time'] = log['time'].iloc[0]
                log_correspond_label_df = pd.concat([log_correspond_label_df, msg_df])
            else:
                msg_df = log[log['label'] == '']
                msg_df['label'] = log[log['label'] != '']['label'].iloc[0]
                msg_df['fault_time'] = log[log['label'] != '']['time'].iloc[0]
                log_correspond_label_df = pd.concat([log_correspond_label_df, msg_df])
        else:
            label_df = log[log['label'] != '']
            msg_df = log[log['label'] == '']
            for msg_item in msg_df.iterrows():
                previous_delta_time = 1000 * 24 * 3600
                for label_item in label_df.iterrows():
                    now_delta_time = abs(datetime.strptime(label_item[1]['time'],'%Y-%m-%d %H:%M:%S'
                        ) - datetime.strptime(msg_item[1]['time'],'%Y-%m-%d %H:%M:%S'))
                    if now_delta_time.days * 24 * 3600 + now_delta_time.seconds < previous_delta_time:
                        previous_delta_time = now_delta_time.days * 24 * 3600 + now_delta_time.seconds
                        if previous_delta_time < cutoff:
                            msg_item[1]['fault_time'] = label_item[1]['time']
                            msg_item[1]['label'] = label_item[1]['label']
            log_correspond_label_df = pd.concat([log_correspond_label_df, msg_df])
            # 没有日志的标签，直接将标签的可用信息作为日志
            for label_item in label_df.iterrows():
                if len(msg_df[(msg_df['fault_time'] == label_item[1]['time']) & (
                    msg_df['label'] == label_item[1]['label'])]) == 0:
                    label_item[1]['fault_time'] = label_item[1]['time']
            log_correspond_label_df = pd.concat([log_correspond_label_df, label_df])
    if len(log_correspond_label_df[log_correspond_label_df['label'] == '']) > 0:
        no_label_log_list.append(log_correspond_label_df[log_correspond_label_df['label'] == ''])
    log_correspond_label_df = log_correspond_label_df[log_correspond_label_df['fault_time'] != '']
    return log_correspond_label_df, no_label_log_list



# sn分组后，本次报错和上次报错之间的日志匹配到本次报错
def divideLogByFaultTime(log_label_df: pd.DataFrame):
    log_correspond_label_df = pd.DataFrame(columns=['sn', 'fault_time', 'msg', 'time', 'server_model', 'label'])
    no_label_log_list = []
    log_label_df =  log_label_df.reset_index(drop = True)
    
    for sn, log in log_label_df.groupby('sn'):
        if len(log[log['label'] != '']) == 0:
            no_label_log_list.append(log)
        elif len(log[log['label'] != '']) == 1:
            msg_df = log[log['label'] == '']
            msg_df['label'] = log[log['label'] != '']['label'].iloc[0]
            msg_df['fault_time'] = log[log['label'] != '']['time'].iloc[0]
            log_correspond_label_df = pd.concat([log_correspond_label_df, msg_df])
        else:
            # 使用index的顺序取数时，要注意index必须按所需的顺序排列
            cutoff_index = [-1] + log.loc[log['label'] != ''].index.tolist() + [log.index.tolist()[-1]+1]
            for kth in range(len(cutoff_index)-1):
                temp_log = log.loc[(log.index <= cutoff_index[kth+1]) & (log.index > cutoff_index[kth])]
                if len(temp_log) > 0:
                    if len(temp_log[temp_log['label'] != '']) == 0:
                        no_label_log_list.append(temp_log)
                    # 只有标签，没有日志的数据，把标签的部分数据直接作为日志
                    elif len(temp_log) == 1:
                        msg_df = temp_log
                        msg_df['fault_time'] = temp_log[temp_log['label'] != '']['time'].iloc[0]
                        log_correspond_label_df = pd.concat([log_correspond_label_df, msg_df])
                    else:
                        msg_df = temp_log[temp_log['label'] == '']
                        msg_df['label'] = temp_log[temp_log['label'] != '']['label'].iloc[0]
                        msg_df['fault_time'] = temp_log[temp_log['label'] != '']['time'].iloc[0]
                        log_correspond_label_df = pd.concat([log_correspond_label_df, msg_df])
    return log_correspond_label_df, no_label_log_list



# 计算统计特征
def calculateStatisticFeature(log_correspond_label_df: pd.DataFrame):
    use_log_label_df = log_correspond_label_df

    use_log_label_df['msg_hour'] = use_log_label_df['time'].apply(lambda x : datetime.strptime(x, "%Y-%m-%d %H:%M:%S").hour)
    use_log_label_df['msg_minute'] = use_log_label_df['time'].apply(lambda x : datetime.strptime(x, "%Y-%m-%d %H:%M:%S").minute)
    use_log_label_df['fault_hour'] = use_log_label_df['fault_time'].apply(lambda x : datetime.strptime(x, "%Y-%m-%d %H:%M:%S").hour)
    use_log_label_df['fault_minute'] = use_log_label_df['fault_time'].apply(lambda x : datetime.strptime(x, "%Y-%m-%d %H:%M:%S").minute)

    # 0414新增
    # 不去重msg_log
    all_msg_log_list = []
    
    # 0408新增
    # 最近一次日志时间距报错时间间隔，单位秒
    nearest_msg_fault_time_delta_list = []
    # 日志不去重时长度1,2,3,4日志数量统计
    all_msg_1_cnt_list=[]
    all_msg_2_cnt_list=[]
    all_msg_3_cnt_list=[]
    all_msg_4_cnt_list=[]
    
    fault_minute_list = []
    msg_1_cnt_list=[]
    msg_2_cnt_list=[]
    msg_3_cnt_list=[]
    msg_4_cnt_list=[]
    msg_hour_max_list=[]
    msg_hour_min_list=[]
    msg_hour_avg_list=[]
    msg_hour_median_list=[]
    msg_hour_mode_list=[]
    msg_minute_max_list=[]
    msg_minute_min_list=[]
    msg_minute_avg_list=[]
    msg_minute_median_list=[]
    msg_minute_mode_list=[]

    sn_list=[]
    day_list=[]
    server_model_list=[]
    msg_log_list=[]
    msg_cnt_list=[]
    fault_hour_list=[]
    label_list=[]
    fault_time_list=[]
    for msg_log_df in use_log_label_df.groupby(['sn','fault_time','label']):
        all_msg_log_str = ''
        msg_log_str = ''
        all_msg_1_cnt = 0
        all_msg_2_cnt = 0
        all_msg_3_cnt = 0
        all_msg_4_cnt = 0
        msg_1_cnt = 0
        msg_2_cnt = 0
        msg_3_cnt = 0
        msg_4_cnt = 0
        for info in msg_log_df[1]['msg']:
            if info == info:
                all_msg_log_str = all_msg_log_str + info.lower() + '.'
                if len(info.split('|')) == 1:
                    all_msg_1_cnt += 1
                elif len(info.split('|')) == 2:
                    all_msg_2_cnt += 1
                elif len(info.split('|')) == 3:
                    all_msg_3_cnt += 1
                else:
                    all_msg_4_cnt += 1
        for info in msg_log_df[1]['msg'].drop_duplicates():
            if info == info:
                msg_log_str=msg_log_str+info.lower()+'.'
                if len(info.split('|')) == 1:
                    msg_1_cnt += 1
                elif len(info.split('|')) == 2:
                    msg_2_cnt += 1
                elif len(info.split('|')) == 3:
                    msg_3_cnt += 1
                else:
                    msg_4_cnt += 1
        nearest_msg_fault_time_delta = abs(datetime.strptime(msg_log_df[1].iloc[-1]['time'],'%Y-%m-%d %H:%M:%S'
                        ) - datetime.strptime(msg_log_df[0][1],'%Y-%m-%d %H:%M:%S'))
        nearest_msg_fault_time_delta = nearest_msg_fault_time_delta.days * 24 * 3600 + nearest_msg_fault_time_delta.seconds
        sm=int(msg_log_df[1].iloc[0]['server_model'][2:])

        sn_list.append(msg_log_df[0][0])
        fault_time_list.append(msg_log_df[0][1])
        label_list.append(msg_log_df[0][2])

        nearest_msg_fault_time_delta_list.append(nearest_msg_fault_time_delta)
        server_model_list.append(sm)
        all_msg_log_list.append(all_msg_log_str)
        msg_log_list.append(msg_log_str)
        msg_cnt_list.append(len(msg_log_df[1]))

        fault_hour_list.append(msg_log_df[1].iloc[0]['fault_hour'])
        fault_minute_list.append(msg_log_df[1].iloc[0]['fault_minute'])

        all_msg_1_cnt_list.append(all_msg_1_cnt)
        all_msg_2_cnt_list.append(all_msg_2_cnt)
        all_msg_3_cnt_list.append(all_msg_3_cnt)
        all_msg_4_cnt_list.append(all_msg_4_cnt)    

        msg_1_cnt_list.append(msg_1_cnt)
        msg_2_cnt_list.append(msg_2_cnt)
        msg_3_cnt_list.append(msg_3_cnt)
        msg_4_cnt_list.append(msg_4_cnt)

        msg_hour_max_list.append(msg_log_df[1]['msg_hour'].max())
        msg_hour_min_list.append(msg_log_df[1]['msg_hour'].min())
        msg_hour_avg_list.append(msg_log_df[1]['msg_hour'].mean())
        msg_hour_median_list.append(msg_log_df[1]['msg_hour'].median())
        msg_hour_mode_list.append(msg_log_df[1]['msg_hour'].mode()[0])

        msg_minute_max_list.append(msg_log_df[1]['msg_minute'].max())
        msg_minute_min_list.append(msg_log_df[1]['msg_minute'].min())
        msg_minute_avg_list.append(msg_log_df[1]['msg_minute'].mean())
        msg_minute_median_list.append(msg_log_df[1]['msg_minute'].median())
        msg_minute_mode_list.append(msg_log_df[1]['msg_minute'].mode()[0])

    msg_log_label_df=pd.DataFrame(
        {
        'sn': sn_list,
        'fault_time': fault_time_list,
        'server_model': server_model_list,
        'msg_cnt': msg_cnt_list,
        'fault_hour': fault_hour_list,
        'fault_minute': fault_minute_list,
        'nearest_msg_fault_time_delta': nearest_msg_fault_time_delta_list,
        'all_msg_1_cnt': all_msg_1_cnt_list,
        'all_msg_2_cnt': all_msg_2_cnt_list,
        'all_msg_3_cnt': all_msg_3_cnt_list,
        'all_msg_4_cnt': all_msg_4_cnt_list,
        'msg_1_cnt': msg_1_cnt_list,
        'msg_2_cnt': msg_2_cnt_list,
        'msg_3_cnt': msg_3_cnt_list,
        'msg_4_cnt': msg_4_cnt_list,
        'msg_hour_max': msg_hour_max_list,
        'msg_hour_min': msg_hour_min_list,
        'msg_hour_avg': msg_hour_avg_list,
        'msg_hour_median': msg_hour_median_list,
        'msg_hour_mode': msg_hour_mode_list,
        'msg_minute_max': msg_minute_max_list,
        'msg_minute_min': msg_minute_min_list,
        'msg_minute_avg': msg_minute_avg_list,
        'msg_minute_median': msg_minute_median_list,
        'msg_minute_mode': msg_minute_mode_list,
        'msg_log': msg_log_list,
        'all_msg_log': all_msg_log_list,
        'label': label_list
        }
    )
    return msg_log_label_df

#####################################
# msg node and graph objects

class msg_node:
    msg_type = None
    attrs = []
    server_model = None
    msg_time = None
    msg_title = None
    idx = 0
    
    def __init__(self, msg, log_time, server_model):
        self.server_model = server_model
        self.attrs = [xx.strip() for xx in msg.split('|')]
        self.msg_time = log_time
        if len(self.attrs) == 1:
            self.msg_type = 0
        elif len(self.attrs) == 2:
            self.msg_type = 1
            self.msg_title = self.attrs[0]
        elif (len(self.attrs) >= 3) and (self.attrs[2] in ['Assert','Asserted','Deasserted','Deassert',
                                                            'asserted','deasserted']):
            self.msg_type = 2
            self.msg_title = self.attrs[0]
            #if len(self.attrs) > 3:
            #    self.attrs[1] = self.attrs[1] + self.attrs[3]  # 合并后面的信息？？
        elif len(self.attrs) >=3:
            self.msg_type = 3
            self.msg_title = self.attrs[0]
        else:
            print('Error in extracting msg attributes!')
    
    def set_index(self, index):
        self.idx = index
        
    def get_embedding_doc2vec(self, fault_time, gensim_model_title, gensim_model_middle,
                              dim_title=10, dim_middle=10):
        # return a 23-d numeric vector of the msg node;
        # embedding coded in 3 segments, and processed differently for each node type
        emb_feat = np.zeros(dim_title+dim_middle+3+110)
        if self.msg_type >= 1:
            emb_feat[0:dim_title] = gensim_model_title.infer_vector(word_tokenize(self.attrs[0].lower()))
        
        if self.msg_type == 0:
            emb_feat[dim_title:(dim_title+dim_middle)] = gensim_model_middle.infer_vector(word_tokenize(self.attrs[0].lower()))
        elif self.msg_type == 1:
            emb_feat[dim_title:(dim_title+dim_middle)] = gensim_model_middle.infer_vector(word_tokenize(self.attrs[1].lower()))
        elif self.msg_type == 2:
            emb_feat[dim_title:(dim_title+dim_middle)] = gensim_model_middle.infer_vector(word_tokenize(self.attrs[1].lower()))
            if self.attrs[2] in ['Assert','Asserted','asserted']:
                emb_feat[dim_title+dim_middle] = 1
        else:
            comb_msg = ' '.join(self.attrs[1:])
            emb_feat[dim_title:(dim_title+dim_middle)] = gensim_model_middle.infer_vector(word_tokenize(comb_msg.lower()))
        
        # one-hot encoding for server model
        emb_feat[(dim_title+dim_middle)+1 + int(self.server_model[2:])] = 1 

        # one-hot encoding for node type
        #emb_feat[(dim_title+dim_middle)+111 + self.msg_type] = 1

        delta_to_fault = fault_time - self.msg_time
        delta_time_to_fault = delta_to_fault.seconds + delta_to_fault.days * (24*3600)
        emb_feat[-2] = np.exp(1 - abs(delta_time_to_fault) / 1e3) * (delta_time_to_fault > 0)
        emb_feat[-1] = 1.0
        self.emb_feat = emb_feat

    def get_embedding_word2vec(self, fault_time, gensim_model_title, gensim_model_middle,
                               dim_title=100, dim_middle=100):
        # return a 23-d numeric vector of the msg node;
        # embedding coded in 3 segments, and processed differently for each node type
        emb_feat = np.zeros(dim_title+dim_middle+3+110)

        def emb_word2vec(msg_temp, word2vec_model):
            item_raw_word_list = word_tokenize(msg_temp)
            embedding_vector_weighted_sum = []
            for word_xth in item_raw_word_list:
                word_drop_xth = re.sub(r'[^\w]','',str(word_xth)).lower()
                if word_drop_xth:
                    vector=word2vec_model.wv[word_drop_xth].reshape(1,-1)[0]
                    embedding_vector_weighted_sum.append(np.array(vector))
            if len(embedding_vector_weighted_sum) > 1:
                return np.mean(np.stack(embedding_vector_weighted_sum),axis=0)
                #return embedding_vector_weighted_sum[-1]
            elif len(embedding_vector_weighted_sum) == 1:
                return embedding_vector_weighted_sum[0]
            else:
                return np.zeros(dim_title)

        # title embedding
        if self.msg_type >= 1:
            emb_feat[0:dim_title] = emb_word2vec(self.attrs[0].lower(), gensim_model_title)
        
        # content embedding 
        if self.msg_type == 0:
            emb_feat[dim_title:(dim_title+dim_middle)] = emb_word2vec(self.attrs[0].lower(), gensim_model_middle)
        elif (self.msg_type == 1) or (self.msg_type == 2):
            emb_feat[dim_title:(dim_title+dim_middle)] = emb_word2vec(self.attrs[1].lower(), gensim_model_middle)
        else:
            comb_msg = ' '.join(self.attrs[1:])
            emb_feat[dim_title:(dim_title+dim_middle)] = emb_word2vec(comb_msg.lower(), gensim_model_middle)

        if (self.msg_type == 2) and (self.attrs[2] in ['Assert','Asserted','asserted']):
            emb_feat[dim_title+dim_middle] = 1

        # one-hot encoding for server model
        emb_feat[(dim_title+dim_middle)+1 + int(self.server_model[2:])] = 1 

        # one-hot encoding for node type
        #emb_feat[(dim_title+dim_middle)+111 + self.msg_type] = 1
            
        delta_to_fault = fault_time - self.msg_time
        #emb_feat[-2] = delta_to_fault.seconds + delta_to_fault.days * (24*3600)
        delta_time_to_fault = delta_to_fault.seconds + delta_to_fault.days * (24*3600)
        emb_feat[-2] = np.exp(1 - abs(delta_time_to_fault) / 1e3) * (delta_time_to_fault > 0)
        emb_feat[-1] = 1.0
        self.emb_feat = emb_feat

    def get_embedding_wordfreq(self, fault_time, word_list):
        emb_feat = np.zeros(len(word_list)*2+3+110)

        pattern_store = [re.compile(word) for word in word_list]

        # title embedding
        if self.msg_type >= 1:
            frequency_vector = [len(re.findall(pattern,self.attrs[0])) for pattern in pattern_store]
            emb_feat[0:len(word_list)] = np.array(frequency_vector)

        # content embedding
        if self.msg_type == 0:
            frequency_vector = [len(re.findall(pattern,self.attrs[0])) for pattern in pattern_store]
            emb_feat[len(word_list):2*len(word_list)] = np.array(frequency_vector)
        elif (self.msg_type == 1) or (self.msg_type == 2):
            frequency_vector = [len(re.findall(pattern,self.attrs[1])) for pattern in pattern_store]
            emb_feat[len(word_list):2*len(word_list)] = np.array(frequency_vector)
        else:
            comb_msg = ' '.join(self.attrs[1:])
            frequency_vector = [len(re.findall(pattern,comb_msg)) for pattern in pattern_store]
            emb_feat[len(word_list):2*len(word_list)] = np.array(frequency_vector)

        if (self.msg_type == 2) and (self.attrs[2] in ['Assert','Asserted','asserted']):
            emb_feat[2*len(word_list)] = 1

        # one-hot encoding for server model
        emb_feat[2*len(word_list)+1 + int(self.server_model[2:])] = 1

        # one-hot encoding for node type
        #emb_feat[(dim_title+dim_middle)+111 + self.msg_type] = 1

        delta_to_fault = fault_time - self.msg_time
        delta_time_to_fault = delta_to_fault.seconds + delta_to_fault.days * (24*3600)
        emb_feat[-2] = np.exp(1 - abs(delta_time_to_fault) / 1e3) * (delta_time_to_fault > 0)
        emb_feat[-1] = 1.0
        self.emb_feat = emb_feat

class msg_graph:
    nodes = []  # list of msg_node
    edges = []  # list of tuples (msg_node_index1, msg_node_index2, edge_type, delta_time)
    error_label = None
    unique_edge_types = None
    server_model = None
    
    def __init__(self, nodes):
        # 注！这里默认输入的nodes是已经按照时序排好的
        self.nodes = nodes
        self.edges = []
        self.server_model = nodes[0].server_model
        
    def assign_index(self):
        for kth, node in enumerate(self.nodes):
            node.set_index(kth)
    
    def build_raw_graph(self, with_time_order = False):
        self.unique_edge_types = None
        edge_stamps = []
        # edge type 1: same msg_title
        unique_edge_types = list(set([xx.msg_title for xx in self.nodes]))
        unique_edge_lengths = np.zeros(len(unique_edge_types))
        edge_type_vector = np.array([xx.msg_title for xx in self.nodes])
        for kth in range(len(unique_edge_types)):
            temp_nodes = np.array(self.nodes)[edge_type_vector == unique_edge_types[kth]].tolist()
            edge_cnt = 1
            for pth in range(len(temp_nodes)-1):
                self.edges.append((temp_nodes[pth].idx, temp_nodes[pth+1].idx, 
                                   unique_edge_types[kth],
                                   (temp_nodes[pth+1].msg_time - temp_nodes[pth].msg_time).seconds))
                edge_stamps.append((temp_nodes[pth].idx, temp_nodes[pth+1].idx))
                edge_cnt += 1
            unique_edge_lengths[kth] = edge_cnt
        self.unique_edge_types = unique_edge_types
        self.unique_edge_lengths = unique_edge_lengths
        
        # edge type 2: order of time
        if with_time_order:
            for ith in range(len(self.nodes)-1):
                if not ((ith, ith+1) in edge_stamps):
                    self.edges.append((ith, ith+1, 'time_order', (self.nodes[ith+1].msg_time - self.nodes[ith].msg_time).seconds))
        
    def raw_network_info(self):
        node_info = [(xx.idx, {'type': xx.msg_type}) for xx in self.nodes]
        edge_info = [(xx[0], xx[1], {'type': xx[2],'weight': 10 / (xx[3]+1e-6)}) for xx in self.edges]
        return node_info, edge_info, self.error_label
    
    def representative_graph_wordfreq(self, word_list, fault_time, error_label=None):
        # node: each unique msg_title, avg feature
        # edge: all to all, delta time (in seconds)
        # output node: a psudo node (without specific time)
        
        edge_type_vector = np.array([xx.msg_title for xx in self.nodes])
        rep_nodes_feat = []
        rep_nodes_time = []
        for kth, edge_type in enumerate(self.unique_edge_types):          
            temp_nodes = np.array(self.nodes)[edge_type_vector == self.unique_edge_types[kth]].tolist()
            avg_feat = []
            for each_node in temp_nodes:
                each_node.get_embedding_wordfreq(fault_time, word_list)
                avg_feat.append(each_node.emb_feat)
            # caculate avg embedding
            avg_vector = np.mean(np.array(avg_feat),axis=0)
            
            # manual normalization
            avg_vector[-1] = len(temp_nodes) / 5
            #avg_vector[-2] = avg_feat[-1][-2] / 1e4
            
            rep_nodes_feat.append(avg_vector)
            rep_nodes_time.append(temp_nodes[-1].msg_time)
        
        x = np.array(rep_nodes_feat)
        
        # assign edges
        edge_index = []
        edge_attr = []
        for kth in range(len(rep_nodes_feat)-1):
            for mth in range(kth+1, len(rep_nodes_feat)):
                edge_index.append([kth, mth])
                edge_attr.append((rep_nodes_time[mth] - rep_nodes_time[kth]).seconds + \
                                 (rep_nodes_time[mth] - rep_nodes_time[kth]).days*24*3600)
        
        # assign self interations
        for kth in range(len(rep_nodes_feat)):
            edge_index.append([kth, kth])
            edge_attr.append(0)
        
        # assign error type to the graph
        self.error_label = error_label
        
        return x, edge_index, edge_attr, self.server_model

    def representative_graph_word2vec(self, gensim_model_title, gensim_model_middle, fault_time, 
                                      error_label=None, dim_title=10, dim_middle=10):
        # node: each unique msg_title, avg feature
        # edge: all to all, delta time (in seconds)
        # output node: a psudo node (without specific time)
        
        edge_type_vector = np.array([xx.msg_title for xx in self.nodes])
        rep_nodes_feat = []
        rep_nodes_time = []
        for kth, edge_type in enumerate(self.unique_edge_types):          
            temp_nodes = np.array(self.nodes)[edge_type_vector == self.unique_edge_types[kth]].tolist()
            avg_feat = []
            for each_node in temp_nodes:
                each_node.get_embedding_word2vec(fault_time, gensim_model_title, gensim_model_middle, 
                                                 dim_title, dim_middle)
                avg_feat.append(each_node.emb_feat)
            # caculate avg embedding
            avg_vector = np.mean(np.array(avg_feat),axis=0)
            
            # manual normalization
            avg_vector[-1] = len(temp_nodes) / 5
            #avg_vector[-2] = avg_feat[-1][-2] / 1e4
            
            rep_nodes_feat.append(avg_vector)
            rep_nodes_time.append(temp_nodes[-1].msg_time)
        
        x = np.array(rep_nodes_feat)
        
        # assign edges
        edge_index = []
        edge_attr = []
        for kth in range(len(rep_nodes_feat)-1):
            for mth in range(kth+1, len(rep_nodes_feat)):
                edge_index.append([kth, mth])
                edge_attr.append((rep_nodes_time[mth] - rep_nodes_time[kth]).seconds + \
                                 (rep_nodes_time[mth] - rep_nodes_time[kth]).days*24*3600)
        
        # assign self interations
        for kth in range(len(rep_nodes_feat)):
            edge_index.append([kth, kth])
            edge_attr.append(0)
        
        # assign error type to the graph
        self.error_label = error_label
        
        return x, edge_index, edge_attr, self.server_model

    def category_graph_word2vec(self, gensim_model_title, gensim_model_middle, fault_time, 
                                error_label=None, dim_title=10, dim_middle=10):
        # node: each unique msg_title, avg feature
        # edge: all to all, delta time (in seconds)
        # output node: a psudo node (without specific time)
        
        edge_type_vector = np.array([xx.msg_title for xx in self.nodes])
        rep_nodes_feat = []
        category_last_node_idx = []
        rep_nodes_time = []
        edge_index = []
        edge_attr = []

        for each_node in self.nodes:
            each_node.get_embedding_word2vec(fault_time, gensim_model_title, gensim_model_middle, 
                                             dim_title, dim_middle)
            rep_nodes_feat.append(each_node.emb_feat)
        node_feat = np.array(rep_nodes_feat)
        #node_feat[:,-2] = node_feat[:,-2] / 1e3   # manual normalization
        node_feat = node_feat[:,0:-1]             # get rid of the last node number

        # assign within-category edges
        for kth, edge_type in enumerate(self.unique_edge_types):          
            temp_nodes = np.array(self.nodes)[edge_type_vector == self.unique_edge_types[kth]].tolist()
            for lth in range(len(temp_nodes)-1):
                edge_index.append([temp_nodes[lth].idx, temp_nodes[lth+1].idx])
                delta_time = (temp_nodes[lth+1].msg_time - temp_nodes[lth].msg_time).seconds + \
                             (temp_nodes[lth+1].msg_time - temp_nodes[lth].msg_time).days*24*3600
                edge_attr.append(np.exp(1 - abs(delta_time) / 1e3) * (delta_time > 0))
            
            rep_nodes_time.append(temp_nodes[-1].msg_time)
            category_last_node_idx.append(temp_nodes[-1].idx)
        
        x = node_feat
        
        # assign edges
        for rth in range(len(category_last_node_idx)-1):
            for tth in range(rth+1, len(category_last_node_idx)):
                edge_index.append([category_last_node_idx[rth], category_last_node_idx[tth]])
                delta_time = (rep_nodes_time[tth] - rep_nodes_time[rth]).seconds + \
                             (rep_nodes_time[tth] - rep_nodes_time[rth]).days*24*3600
                edge_attr.append(np.exp(1 - abs(delta_time) / 1e3) * (delta_time > 0))
        
        # assign self interations
        for kth in range(len(category_last_node_idx)):
            edge_index.append([category_last_node_idx[kth], category_last_node_idx[kth]])
            edge_attr.append(np.exp(1))
        
        # assign error type to the graph
        self.error_label = error_label
        
        return x, edge_index, edge_attr, self.server_model


from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import pdb

class prepare_repGraph_wordfreq(object):
    
    #def __init__(self, gensim_model_title, gensim_model_content, 
    #             dim_title=10, dim_middle=10):
    def __init__(self, word_list):
        self.word_list = word_list
    
    def __call__(self, grp):
        nodes = []
        for ith in range(len(grp)):
            nodes.append(msg_node(grp['msg'].iloc[ith], 
                                  grp['log_time'].iloc[ith], 
                                  grp['server_model'].iloc[ith]))
        test_graph = msg_graph(nodes)
        test_graph.assign_index()
        test_graph.build_raw_graph(grp['label'].iloc[-1])

        x, edge_index, edge_attr, server_model =test_graph.representative_graph_wordfreq(self.word_list,
                                                                                         pd.to_datetime(grp['fault_time']).iloc[0],
                                                                                         error_label=grp['label'].iloc[0])

        d = Data(x=torch.tensor(x).type(torch.float32))
        d.label = torch.tensor(test_graph.error_label).type(torch.long)
        d.num_nodes = x.shape[0]
        
        d.edge_index = torch.transpose(torch.tensor(edge_index).type(torch.long),0,1)
        d.edge_attr = torch.tensor(edge_attr).type(torch.float32)
                
        d.id = grp['sample_id'].iloc[0]
        d.server_model = server_model

        return d

class prepare_repGraph_word2vec(object):
    
    def __init__(self, gensim_model_title, gensim_model_content, 
                 dim_title=10, dim_middle=10):
        self.gensim_model_title = gensim_model_title
        self.gensim_model_content = gensim_model_content
        self.dim_title = dim_title
        self.dim_middle = dim_middle
    
    def __call__(self, grp):
        nodes = []
        for ith in range(len(grp)):
            nodes.append(msg_node(grp['msg'].iloc[ith], 
                                  grp['log_time'].iloc[ith], 
                                  grp['server_model'].iloc[ith]))
        test_graph = msg_graph(nodes)
        test_graph.assign_index()
        test_graph.build_raw_graph(grp['label'].iloc[-1])

        x, edge_index, edge_attr, server_model =test_graph.representative_graph_word2vec(self.gensim_model_title, 
                                                                                         self.gensim_model_content,
                                                                                         pd.to_datetime(grp['fault_time']).iloc[0],
                                                                                         error_label=grp['label'].iloc[0],
                                                                                         dim_title=self.dim_title, 
                                                                                         dim_middle=self.dim_middle)


        d = Data(x=torch.tensor(x).type(torch.float32))
        d.label = torch.tensor(test_graph.error_label).type(torch.long)
        d.num_nodes = x.shape[0]
        
        d.edge_index = torch.transpose(torch.tensor(edge_index).type(torch.long),0,1)
        d.edge_attr = torch.tensor(edge_attr).type(torch.float32)
                
        d.id = grp['sample_id'].iloc[0]
        d.server_model = server_model

        return d

class prepare_catGraph_word2vec(object):
    
    def __init__(self, gensim_model_title, gensim_model_content, 
                 dim_title=10, dim_middle=10):
        self.gensim_model_title = gensim_model_title
        self.gensim_model_content = gensim_model_content
        self.dim_title = dim_title
        self.dim_middle = dim_middle
    
    def __call__(self, grp):
        nodes = []
        for ith in range(len(grp)):
            nodes.append(msg_node(grp['msg'].iloc[ith], 
                                  grp['log_time'].iloc[ith], 
                                  grp['server_model'].iloc[ith]))
        test_graph = msg_graph(nodes)
        test_graph.assign_index()
        test_graph.build_raw_graph(grp['label'].iloc[-1])

        x, edge_index, edge_attr, server_model =test_graph.category_graph_word2vec(self.gensim_model_title, 
                                                                                   self.gensim_model_content,
                                                                                   pd.to_datetime(grp['fault_time']).iloc[0],
                                                                                   error_label=grp['label'].iloc[0],
                                                                                   dim_title=self.dim_title, 
                                                                                   dim_middle=self.dim_middle)


        d = Data(x=torch.tensor(x).type(torch.float32))
        d.label = torch.tensor(test_graph.error_label).type(torch.long)
        d.num_nodes = x.shape[0]
        
        d.edge_index = torch.transpose(torch.tensor(edge_index).type(torch.long),0,1)
        d.edge_attr = torch.tensor(edge_attr).type(torch.float32)
                
        d.id = grp['sample_id'].iloc[0]
        d.server_model = server_model

        return d

class RepGraphDataset(Dataset):
    
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.transform(self.data[idx])
