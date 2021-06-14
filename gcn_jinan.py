'''
Descripttion:
version:
Author: ShaojieDai
Date: 2021-03-18 22:22:58
LastEditors: sueRimn
LastEditTime: 2021-05-20 11:02:38
'''
import random
from sklearn.model_selection import train_test_split
import itertools

import matplotlib.pyplot as plt
'''
优化多个模型的参数 使用tertools.chain将参数链接起来
self.optimizer = optim.Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
'''

from utils import *
from args import *
from models import *
from metrics import *
from walk import RWGraph
from extract_city_volume_info import *
from attention import Multi_Head_SelfAttention


def objective_volume_current(train_ways_segment_volume_dict, train_ways_segment_vec_dict, topk, negk):
    '''
    @name:
    @Date: 2021-04-27 15:49:53
    @msg: 只考虑当前时间片的流量信息影响
    @param {*}
    @return {*}
    '''
    pre_volume  = []
    true_volume = []
    loss_term = 0.

    for k1,v1 in train_ways_segment_vec_dict.items():
        num_slice = v1.shape[0]
        for i in range(num_slice):
            curr_score_dict, recent_score_dict, period_score_dict = {}, {}, {}
            # curr_score = torch.cosine_similarity(v1, v2, dim=0) # 和点乘不同
            # curr_score = torch.mm(v1.unsqueeze(0),v2.unsqueeze(1))
            for k2, v2 in train_ways_segment_vec_dict.items():
                if(k1 != k2):
                    curr_score   = torch.cosine_similarity(v1[i], v2[i], dim=-1)
                    curr_score_dict[k2]   = curr_score
            # sorted_score_dict_min = sorted(score_dict.items(), key=lambda item:item[1], reverse = True)[-negk:] 本来是计算负样本的
            cur_sum_volume_max, cur_sum_sim_score        =  0, .0
            cur_sorted_score_dict_max    = sorted(curr_score_dict.items(),   key=lambda item:item[1], reverse = True)[:topk]

            for truple in cur_sorted_score_dict_max:
                cur_sum_volume_max = cur_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i]*truple[1] # train_ways_segment_volume_dict[truple[0]] (288,1)  TODO: 不能硬编码
                cur_sum_sim_score = cur_sum_sim_score + truple[1]
            cur_pre_volume = cur_sum_volume_max / cur_sum_sim_score
            loss_term = abs((train_ways_segment_volume_dict[k1][i] - cur_pre_volume))**2 + loss_term
    return loss_term


def objective_volume_recent(train_ways_segment_volume_dict, train_ways_segment_vec_dict, topk, negk):
    '''
    @name:
    @Date: 2021-04-27 15:48:30
    @msg: 只考虑当前时间流量时间片的前两个时间片的影响作用
    @param {*}
    @return {*}
    '''
    pre_volume  = []
    true_volume = []
    loss_term = 0.

    for k1,v1 in train_ways_segment_vec_dict.items():
        num_slice = v1.shape[0]

        for i in range(num_slice): # (744)
            # 查看当第i时间片上，和k1最相近的前5个
            recent_score_dict, period_score_dict = {}, {}
            for k2, v2 in train_ways_segment_vec_dict.items():
                if(k1 != k2):
                    if i>=2:
                        recent_score = torch.cosine_similarity(v1[i-1], v2[i-1], dim=-1)
                        period_score = torch.cosine_similarity(v1[i-2], v2[i-2], dim=-1)
                        recent_score_dict[k2] = recent_score
                        period_score_dict[k2] = period_score

                    elif i==1:
                        recent_score = torch.cosine_similarity(v1[i-1], v2[i-1], dim=-1)
                        recent_score_dict[k2] = recent_score

            # sorted_score_dict_min = sorted(score_dict.items(), key=lambda item:item[1], reverse = True)[-negk:] 本来是计算负样本的
            recent_sum_volume_max, recent_sum_sim_score  =  0, .0
            period_sum_volume_max, period_sum_sim_score  =  0, .0

            if i == 1:
                recent_sorted_score_dict_max = sorted(recent_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]
                for truple in recent_sorted_score_dict_max:
                    recent_sum_volume_max = recent_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-1]*truple[1] # train_ways_segment_volume_dict[truple[0]] (288,1)  TODO: 不能硬编码
                    recent_sum_sim_score = recent_sum_sim_score + truple[1]
                recent_pre_volume = recent_sum_volume_max / recent_sum_sim_score
                loss_term = abs((train_ways_segment_volume_dict[k1][i] - recent_pre_volume))**2 + loss_term

            if i >= 2:
                recent_sorted_score_dict_max = sorted(recent_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]
                period_sorted_score_dict_max = sorted(period_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]

                for truple in recent_sorted_score_dict_max:
                    recent_sum_volume_max = recent_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-1]*truple[1] # train_ways_segment_volume_dict[truple[0]] (288,1)  TODO: 不能硬编码
                    recent_sum_sim_score = recent_sum_sim_score + truple[1]
                recent_pre_volume = recent_sum_volume_max / recent_sum_sim_score
                loss_term = abs((train_ways_segment_volume_dict[k1][i] - recent_pre_volume))**2 + loss_term

                for truple in period_sorted_score_dict_max:
                    period_sum_volume_max = period_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-2]*truple[1] # train_ways_segment_volume_dict[truple[0]] (288,1)  TODO: 不能硬编码
                    period_sum_sim_score = period_sum_sim_score + truple[1]
                period_pre_volume = period_sum_volume_max/period_sum_sim_score
                loss_term = abs((train_ways_segment_volume_dict[k1][i] - period_pre_volume)) + loss_term

    return loss_term

def objective_volume_daily(train_ways_segment_volume_dict, train_ways_segment_vec_dict, topk, negk):
    pre_volume  = []
    true_volume = []
    loss_term = 0.

    for k1,v1 in train_ways_segment_vec_dict.items():
        num_slice = v1.shape[0]
        daily1_score_dict, daily2_score_dict = {}, {}
        for i in range(num_slice):
            # curr_score = torch.cosine_similarity(v1, v2, dim=0) # 和点乘不同
            # curr_score = torch.mm(v1.unsqueeze(0),v2.unsqueeze(1))
            for k2, v2 in train_ways_segment_vec_dict.items():
                if(k1 != k2):
                    if i < 24:
                        break
                    elif i < 48:
                        daily1_score   = torch.cosine_similarity(v1[i-24], v2[i-24], dim=-1)
                        daily1_score_dict[k2]   = daily1_score
                    else:
                        daily1_score   = torch.cosine_similarity(v1[i-24], v2[i-24], dim=-1)
                        daily2_score   = torch.cosine_similarity(v1[i-48], v2[i-48], dim=-1)
                        daily1_score_dict[k2]  = daily1_score
                        daily2_score_dict[k2]  = daily2_score

            daily1_sum_volume_max, daily1_sum_sim_score  =  0, .0
            daily2_sum_volume_max, daily2_sum_sim_score  =  0, .0

            if i < 24:
                continue
            elif i < 48:
                daily1_sorted_score_dict_max = sorted(daily1_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]

                for truple in daily1_sorted_score_dict_max:
                    daily1_sum_volume_max = daily1_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-24]*truple[1] # train_ways_segment_volume_dict[truple[0]] (288,1)  TODO: 不能硬编码
                    daily1_sum_sim_score = daily1_sum_sim_score + truple[1]
                daily1_pre_volume = daily1_sum_volume_max / daily1_sum_sim_score
                loss_term = abs(train_ways_segment_volume_dict[k1][i] - daily1_pre_volume)**2 + loss_term

            else:
                daily1_sorted_score_dict_max = sorted(daily1_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]
                daily2_sorted_score_dict_max = sorted(daily2_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]

                for truple in daily1_sorted_score_dict_max:
                    daily1_sum_volume_max = daily1_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-24]*truple[1] # train_ways_segment_volume_dict[truple[0]] (288,1)  TODO: 不能硬编码
                    daily1_sum_sim_score = daily1_sum_sim_score + truple[1]
                daily1_pre_volume = daily1_sum_volume_max / daily1_sum_sim_score
                loss_term = abs(train_ways_segment_volume_dict[k1][i] - daily1_pre_volume)**2 + loss_term

                for truple in daily2_sorted_score_dict_max:
                    daily2_sum_volume_max = daily2_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-48]*truple[1] # train_ways_segment_volume_dict[truple[0]] (288,1)  TODO: 不能硬编码
                    daily2_sum_sim_score = daily2_sum_sim_score + truple[1]
                daily2_pre_volume = daily2_sum_volume_max / daily2_sum_sim_score
                loss_term = abs((train_ways_segment_volume_dict[k1][i] - daily2_pre_volume))**2 + loss_term
    return loss_term


def objective_volume_weekly(train_ways_segment_volume_dict, train_ways_segment_vec_dict, topk, negk):
    pre_volume  = []
    true_volume = []
    loss_term   = 0.

    for k1,v1 in train_ways_segment_vec_dict.items():
        num_slice = v1.shape[0]
        weekly1_score_dict, weekly2_score_dict = {}, {}
        for i in range(num_slice):
            # curr_score = torch.cosine_similarity(v1, v2, dim=0) # 和点乘不同
            # curr_score = torch.mm(v1.unsqueeze(0),v2.unsqueeze(1)) weekly
            for k2, v2 in train_ways_segment_vec_dict.items():
                if(k1 != k2):
                    if i < 24*7:
                        break
                    elif i < 24*14:
                        weekly1_score   = torch.cosine_similarity(v1[i-24*7], v2[i-24*7], dim=-1)
                        weekly1_score_dict[k2]   = weekly1_score
                    else:
                        weekly1_score   = torch.cosine_similarity(v1[i-24*7], v2[i-24*7], dim=-1)
                        weekly2_score   = torch.cosine_similarity(v1[i-24*14], v2[i-24*14], dim=-1)
                        weekly1_score_dict[k2]  = weekly1_score
                        weekly2_score_dict[k2]  = weekly2_score

            weekly1_sum_volume_max, weekly1_sum_sim_score  =  0, .0
            weekly2_sum_volume_max, weekly2_sum_sim_score  =  0, .0

            if i < 24*7:
                continue
            elif i < 24*14:
                weekly1_sorted_score_dict_max = sorted(weekly1_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]

                for truple in weekly1_sorted_score_dict_max:
                    weekly1_sum_volume_max = weekly1_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-24*7]*truple[1] # train_ways_segment_volume_dict[truple[0]] (288,1)  TODO: 不能硬编码
                    weekly1_sum_sim_score = weekly1_sum_sim_score + truple[1]
                weekly1_pre_volume = weekly1_sum_volume_max / weekly1_sum_sim_score
                loss_term = abs(train_ways_segment_volume_dict[k1][i] - weekly1_pre_volume)**2 + loss_term

            else:
                weekly1_sorted_score_dict_max = sorted(weekly1_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]
                weekly2_sorted_score_dict_max = sorted(weekly2_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]

                for truple in weekly1_sorted_score_dict_max:
                    weekly1_sum_volume_max = weekly1_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-24*7]*truple[1] # train_ways_segment_volume_dict[truple[0]] (288,1)  TODO: 不能硬编码
                    weekly1_sum_sim_score = weekly1_sum_sim_score + truple[1]
                weekly1_pre_volume = weekly1_sum_volume_max / weekly1_sum_sim_score
                loss_term = abs(train_ways_segment_volume_dict[k1][i] - weekly1_pre_volume)**2 + loss_term

                for truple in weekly2_sorted_score_dict_max:
                    weekly2_sum_volume_max = weekly2_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-24*14]*truple[1] # train_ways_segment_volume_dict[truple[0]] (288,1)  TODO: 不能硬编码
                    weekly2_sum_sim_score = weekly2_sum_sim_score + truple[1]
                weekly2_pre_volume = weekly2_sum_volume_max / weekly2_sum_sim_score
                loss_term = abs((train_ways_segment_volume_dict[k1][i] - weekly2_pre_volume))**2 + loss_ter
    return loss_term

def objective_rw( train_ways_segment_vec_dict, negk, adj_weight_dict, output, vocab_list, word_freqs):
    # 参考师哥www那篇的目标函数学习
    loss_term = torch.tensor(0.,dtype=torch.float32)
    for k1,v1 in train_ways_segment_vec_dict.items():
        negative_list = []
        cur_adj = adj_weight_dict[k1] # {    0: {1: {'weight': 0.7310585786300049}, 131: {'weight': 0.7310585786300049}}     }
        while( len(set(negative_list) - set(cur_adj.keys())) < negk ): # TODO: 可能出现死循环，几率很小
            negative_list = random.choices(population=vocab_list, weights=word_freqs, k=negk)
        positive_embedding = output[:, list(cur_adj.keys())]
        weight_positive_tensor = torch.tensor([item["weight"] for item in list(cur_adj.values())],dtype=torch.float32).unsqueeze(0)
        cur_embedding = output[:,k1]
        negative_embedding = output[:, negative_list]
        # cur_loss_term =  - sum(F.logsigmoid( positive_embedding.mm(cur_embedding.unsqueeze(1)) )) - sum(torch.log( 1. - torch.sigmoid(negative_embedding.mm(cur_embedding.unsqueeze(1))) ))
        # cur_loss_term =  - sum(F.logsigmoid(weight_positive_tensor.mm( positive_embedding.mm(cur_embedding.unsqueeze(1)) ) )) - sum(F.logsigmoid( 1. - negative_embedding.mm(cur_embedding.unsqueeze(1)) ))
        # 下面这个带权重wij                          (1,8)                                             (8,128)             (1,128)
        # cur_loss_term =  - sum(F.logsigmoid(weight_positive_tensor.mm( torch.cosine_similarity( positive_embedding, cur_embedding.unsqueeze(0), dim=1).unsqueeze(1) ))) \
        #                  - sum(torch.log( 1. - torch.sigmoid(torch.cosine_similarity(negative_embedding, cur_embedding.unsqueeze(0), dim=1 ) ) ))
        # 下面这个不带权重wij
        cur_loss_term =  - torch.sum(F.logsigmoid(torch.cosine_similarity( positive_embedding, cur_embedding.unsqueeze(1), dim=-1).unsqueeze(1) )) \
                         - torch.sum(torch.log( 1. - torch.sigmoid(torch.cosine_similarity(negative_embedding, cur_embedding.unsqueeze(1), dim=-1 ) ) ))

        # cur_loss_term_test =  - sum(sum(sum(F.logsigmoid(torch.cosine_similarity( positive_embedding, cur_embedding.unsqueeze(1), dim=-1).unsqueeze(1) )))) \
        #                  - sum(sum(torch.log( 1. - torch.sigmoid(torch.cosine_similarity(negative_embedding, cur_embedding.unsqueeze(1), dim=-1 ) ) )))

        loss_term = loss_term + cur_loss_term
    return loss_term


def train_regression(model, weight_adj_list, train_features, train_ways_segment_volume_dict,
                     test_ways_segment_volume_dict, unnormed_ways_segment_volume_dict, volume_sqrt_var, volume_mean, G, adj,
                     epochs, weight_decay=1e-2,
                     lr=0.001, dropout=0.1):

    '''准备objective_rw的需要数据'''
    walker = RWGraph(G)
    walks_list = walker.simulate_walks(args.num_walks, args.walk_length, schema=None, isweighted=args.isweighted) # TODO: 有136个节点是没有边的，是单独的

    walks_list = [col for row in walks_list for col in row]
    vocab_list = Counter(walks_list).most_common() # 每个元素是一个元组[(539,347), (457,333)...]

    word_counts = np.array([count[1] for count in vocab_list], dtype=np.float32) #
    word_freqs = word_counts / np.sum(word_counts)
    word_freqs = word_freqs ** (3. / 4.)
    adj_weight_dict = find_positive_samples(G)
    vocab_list = [item[0] for item in vocab_list]

    criterion = nn.MSELoss()
    train_ways_segment_list = list(train_ways_segment_volume_dict.keys())


    params_list = []
    for i in range(args.num_slice):
        params_list.append({"params":model.model_list[i].parameters()})
    params_list.append({"params":model.attention.at_block1.parameters()})
    params_list.append({"params":model.attention.at_block2.parameters()})
    params_list.append({"params":model.attention.at_block3.parameters()})
    optimizer = optim.Adam( params_list, lr=lr, weight_decay=weight_decay) # L2惩罚项 TODO: 硬编码
    # optimizer = optim.Adam(itertools.chain(model.model_list[0].parameters(), model.model_list[1].parameters(), model.model_list[2].parameters(),model.model_list[3].parameters(),\
    #                                        model.model_list[4].parameters(), model.model_list[5].parameters(), model.model_list[6].parameters(), model.model_list[7].parameters(),\
    #                                        model.model_list[8].parameters(), model.model_list[9].parameters(), model.model_list[10].parameters(), model.model_list[11].parameters(),
    #                                        model.attention.at_block1.parameters(), model.attention.at_block2.parameters(), model.attention.at_block3.parameters() ), lr=lr,
    #                                        weight_decay=weight_decay) # L2惩罚项 TODO: 硬编码
    # optimizer = optim.SGD(model.parameters(), lr=lr) # L2惩罚项
    t = perf_counter()

    for epoch in range(epochs):
        train_ways_segment_vec_dict = {} # key= ways_segment_id  v=tensor.vec(128)
        model.train()
        optimizer.zero_grad() # 将梯度归零
        if args.model == "SGC":
            output = model(train_features, weight_adj_list) # TODO: 疑问：没有打乱顺序，没有分batch更新 是140整体作为一个batch (553,128)
        if args.model == "GCN":
            output = model(train_features, adj)

        for i, item in enumerate(train_ways_segment_list):
            train_ways_segment_vec_dict[item] = output[:, item, :]
        loss_train_volume_current = objective_volume_current(train_ways_segment_volume_dict, train_ways_segment_vec_dict, args.topk, args.negk)
        loss_train_volume_recent  = objective_volume_recent(train_ways_segment_volume_dict, train_ways_segment_vec_dict, args.topk, args.negk)
        loss_train_volume_daily   = objective_volume_daily(train_ways_segment_volume_dict, train_ways_segment_vec_dict, args.topk, args.negk)
        loss_train_volume_weekly  = objective_volume_weekly(train_ways_segment_volume_dict, train_ways_segment_vec_dict, args.topk, args.negk)

        # (train_ways_segment_vec_dict, negk, adj_weight_dict, output, vocab_list, word_freqs)
        loss_train_rw = objective_rw(train_ways_segment_vec_dict, args.negk, adj_weight_dict, output, vocab_list, word_freqs)
        loss = args.hy_volume_current*loss_train_volume_current + args.hy_volume_recent*loss_train_volume_recent + \
               args.hy_RW*loss_train_rw +  \
               args.hy_volume_daily*loss_train_volume_daily + args.hy_volume_weekly*loss_train_volume_weekly # TODO: loss_train_volume 30左右，loss_train_rw1500左右


        loss.backward() # 反向传播计算得到每个参数的梯度值
        optimizer.step() # 梯度下降执行一步参数更新

        if (epoch) % 2 == 0:
            with torch.no_grad(): # 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度(记录)
                train_ways_segment_vec_dict = {}
                model.eval()
                if args.model == "SGC":
                    output = model(weight_adj_list, train_features)
                if args.model == "GCN": # 暂未处理bug
                    output = model(features, adj)
                for i, item in enumerate(train_ways_segment_list):
                    train_ways_segment_vec_dict[item] = output[:, item]

                leida_pre_MAPE_info_y, leida_pre_MAPE_info_y_head,leida_pre_RMSE_info = evaluate_metric(epoch, output, train_ways_segment_volume_dict, train_ways_segment_vec_dict, test_ways_segment_volume_dict, unnormed_ways_segment_volume_dict, args.topk, volume_sqrt_var, volume_mean)
                print("epoch:{}\tloss:{}".format(epoch, loss))
                print("MAPE_y: {}".format(leida_pre_MAPE_info_y))
                print("MAPE_y_head: {}".format(leida_pre_MAPE_info_y_head))
                print("RMSE: {}".format(leida_pre_RMSE_info))

                print("mean MAPE_y: {}".format(calc_avg_dict_value(leida_pre_MAPE_info_y)))
                print("mean MAPE_y_head: {}".format(calc_avg_dict_value(leida_pre_MAPE_info_y_head)))
                print("mean RMSE: {}".format(calc_avg_dict_value(leida_pre_RMSE_info)))

    train_time = perf_counter()-t
    return leida_pre_MAPE_info_y, leida_pre_MAPE_info_y_head,leida_pre_RMSE_info


if __name__=="__main__":
    args = get_args()
    t = perf_counter()
    '''1.读取数据'''
    all_info = extract_jinan_volume_info()
    jinan_5min_slice_volume_dict, \
    cams_way_segments_dict, way_segments_cams_dict, cams_attr_dict,\
    G_edge_list, G_edge_list_attr = all_info



    '''2.数据indexing化'''
    ways_segment2id, id2ways_segment = indexing(G_edge_list_attr,ifpad=False) # len=433
    # ways_segment2id = read_pkl("jinan/ways_segment2id.pkl")
    # id2ways_segment = read_pkl("jinan/id2ways_segment.pkl")
    way_segments_cams_dict = change_dict_key(way_segments_cams_dict, ways_segment2id)
    cams_way_segments_dict = change_dict_value(cams_way_segments_dict, ways_segment2id)
    G_edge_list = change_tuple_elem(G_edge_list, ways_segment2id) # 1184
    G_edge_list_attr = change_dict_key(G_edge_list_attr, ways_segment2id)
    # TODO: 将多个时间片的信息嵌入G中,硬编码，具体时间需要根据情况来改写内部代码 matched_leida_id_set=("12_0")
    new_jinan_5min_slice_volume_dict, matched_road_id_list = change_jinan_5min_slice_volume_dict_edge2id(ways_segment2id, jinan_5min_slice_volume_dict)


    '''3.利用边构图'''
    G_0, isolated_way_segments = get_G_from_edges_jinan(G_edge_list, G_edge_list_attr)
    # 更新图的属性信息: 起点终点属性、边的流量信息
    G_1 = update_G_with_attr_jinan(G_0, G_edge_list_attr) # edge_volume_dict
    # 利用车道数目更新边的weight num_of_lanes
    G_2 = update_G_with_lanes(G_1)


    '''组织多个时间片数据'''
    # 形成ways_segment_volume={key=way_id  :  value=[12*2*31=744]}
    ways_segment_volume_dict = find_jinan_volume_slice_by_leida(matched_road_id_list, new_jinan_5min_slice_volume_dict)


    ''''处理features adj '''
    features = np.zeros((len(G_edge_list_attr),7),dtype=np.float32)
    adj = nx.adjacency_matrix(G_2)
    if args.cuda:
        features = feature_process_jinan(features, G_edge_list_attr).to(device='cuda') # FloatTensor
        adj = preprocess_adj(adj, normalization=args.normalization).to('cuda') # TODO: 按照FAME的那种方式计算adj 返回sp.tensor
    else:
        features = feature_process_jinan(features, G_edge_list_attr).to(device='cpu')
        adj = preprocess_adj(adj, normalization=args.normalization).to('cpu')


    '''流量标准化'''
    normed_ways_segment_volume_dict, volume_sqrt_var_list, volume_mean_list, unnormed_ways_segment_volume_dict = norm_volume(ways_segment_volume_dict)
    volume_sqrt_var_list = mask_list(volume_sqrt_var_list, 1.)
    volume_mean_list = mask_list(volume_mean_list, 0.)


    '''数据集划分：去除某些低流量路段'''
    unnormed_ways_segment_volume_dict.pop(109)
    unnormed_ways_segment_volume_dict.pop(71)
    data_feature, data_target = preprocess_split_data(unnormed_ways_segment_volume_dict) # TODO: 划分数据集总是随机，不可复现,normed_ways_segment_volume_dict->unnormed_ways_segment_volume_dict
    train_volume_arr, test_volume_arr, train_leida_id_arr, test_leida_id_arr = \
            train_test_split(data_feature, data_target, test_size=args.percent, random_state=args.seed)
    train_ways_segment_volume_dict = combine_ways_segment_volume_dict(train_leida_id_arr, train_volume_arr)
    test_ways_segment_volume_dict  = combine_ways_segment_volume_dict(test_leida_id_arr, test_volume_arr)
    set_seed(args.seed, args.cuda)



    '''设置模型、训练模型'''
    model = JINAN_model(model_type=args.model, num_head=args.num_head ,num_slice=args.num_slice, nfeat=features.shape[1], nhid=args.hidden, nclass=args.output_dim, dropout=args.dropout, degree=args.degree)
    if args.isweighted_adj and args.model == "SGC":
        weight_adj_list, precompute_time = weight_sgc_precompute(adj, args.degree) # args.degree=2 degree of the approximation.
        map_mean, map_list, pre_volume, true_volume, corresponding_way_list = train_regression(model, weight_adj_list, features, train_ways_segment_volume_dict, test_ways_segment_volume_dict, volume_sqrt_var_list, volume_mean_list, G_2, adj, args.epochs, lr=args.lr)
    elif not(args.isweighted_adj) and args.model == "SGC":
        weight_adj_list, precompute_time = sgc_precompute(features, adj, args.degree)
        map_mean, map_list, pre_volume, true_volume, corresponding_way_list = train_regression(model, weight_adj_list, features, train_ways_segment_volume_dict, test_ways_segment_volume_dict, volume_sqrt_var_list, volume_mean_list, G_2, adj, args.epochs, lr=args.lr)
    elif args.model == "GCN":
        # 不初始化报错
        weight_adj_list = 0.
        leida_pre_MAPE_info_y, leida_pre_MAPE_info_y_head,leida_pre_RMSE_info = train_regression(model, weight_adj_list, features, train_ways_segment_volume_dict, test_ways_segment_volume_dict, unnormed_ways_segment_volume_dict, volume_sqrt_var_list, volume_mean_list, G_2, adj, args.epochs, lr=args.lr)

    # 训练; 修改
    # map_mean, map_list, pre_volume, true_volume, corresponding_way_list = train_regression(model, weight_adj_list, features, train_ways_segment_volume_dict, test_ways_segment_volume_dict, volume_sqrt_var, volume_mean, G_3, adj, args.epochs, lr=args.lr)
    show_pre_info(leida_pre_MAPE_info_y, leida_pre_RMSE_info, way_segments_cams_dict)
    print("over!")

