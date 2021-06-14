import random
from sklearn.model_selection import train_test_split
import itertools

import matplotlib.pyplot as plt
import optuna
import time
'''

self.optimizer = optim.Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
'''

from utils import *
from args import *
from models import *
from metrics import *
from walk import RWGraph
from extract_city_volume_info import *
from attention import Multi_Head_SelfAttention

def matched_cams_plot(way_segments_cams_dict, matched_road_id_list):
    matched_cams = []
    for road_id in matched_road_id_list:
        matched_cams.append(way_segments_cams_dict[road_id])
    with open('matched_cams.txt','w', encoding='utf-8') as f:
        for item in matched_cams:
            f.write(str(item)+'\n')

def matual_split_data(normed_ways_segment_volume_dict):
    test_ways_list = [465, 4, 331, 121, 69, 65, 50, 139]
    train_ways_segment_volume_dict = {}
    test_ways_segment_volume_dict  = {}
    train_ways_set = set(normed_ways_segment_volume_dict.keys()) - set(test_ways_list)
    for test_way in test_ways_list:
        test_ways_segment_volume_dict[test_way] = normed_ways_segment_volume_dict[test_way]
    for train_way in train_ways_set:
        train_ways_segment_volume_dict[train_way] = normed_ways_segment_volume_dict[train_way]
    return train_ways_segment_volume_dict, test_ways_segment_volume_dict

def objective(trial):
    '''0.optuna hyper-parameters'''
    config = {
        "epochs" : trial.suggest_int('epochs',  10, 10),
        "hy_RW"  : trial.suggest_uniform('hy_RW' , 0.1, 10), # 7.229532451922468, 7.229532451922468
        "hy_volume_current" : trial.suggest_uniform('hy_volume_current' , 0.1, 10), # 8.000282368653435, 8.000282368653435
        "hy_volume_recent"  : trial.suggest_uniform('hy_volume_recent' , 0.1, 10), #  0.3680750182157313,  0.3680750182157313
        "hy_volume_daily"   : trial.suggest_uniform('hy_volume_daily' ,  0.1, 10), #   3.268632890232799, 3.268632890232799
        "hy_volume_weekly"  : trial.suggest_uniform('hy_volume_weekly' ,  0.1, 10), #  6.459585633045093, 6.459585633045093
        'hy_unvolume_recent' : trial.suggest_uniform('hy_unvolume_recent' , 0.1, 10),
        'hy_unvolume_daily'  : trial.suggest_uniform('hy_unvolume_daily' , 0.1, 10),
        'hy_unvolume_weekly' : trial.suggest_uniform('hy_unvolume_weekly' , 0.1, 10),
    }

    args = get_args()

    t = perf_counter()
    '''1.read data'''
    all_info = extract_jinan_volume_info()
    jinan_5min_slice_volume_dict, \
    cams_way_segments_dict, way_segments_cams_dict, cams_attr_dict,\
    G_edge_list, G_edge_list_attr = all_info



    '''2.indexing'''
    ways_segment2id, id2ways_segment = indexing(G_edge_list_attr,ifpad=False)
    way_segments_cams_dict = change_dict_key(way_segments_cams_dict, ways_segment2id)
    cams_way_segments_dict = change_dict_value(cams_way_segments_dict, ways_segment2id)
    G_edge_list = change_tuple_elem(G_edge_list, ways_segment2id)
    G_edge_list_attr = change_dict_key(G_edge_list_attr, ways_segment2id)
    new_jinan_5min_slice_volume_dict, matched_road_id_list = change_jinan_5min_slice_volume_dict_edge2id(ways_segment2id, jinan_5min_slice_volume_dict)


    '''3.construct graph'''
    G_0, isolated_way_segments = get_G_from_edges_jinan(G_edge_list, G_edge_list_attr)
    G_1 = update_G_with_attr_jinan(G_0, G_edge_list_attr) # edge_volume_dict
    G_2 = update_G_with_lanes(G_1)




    ways_segment_volume_dict = find_jinan_volume_slice_by_leida(matched_road_id_list, new_jinan_5min_slice_volume_dict)


    ''''process features adj '''
    features = np.zeros((len(G_edge_list_attr),7),dtype=np.float32)
    adj = nx.adjacency_matrix(G_2)
    if args.cuda:
        features = feature_process_jinan(features, G_edge_list_attr).to(device='cuda') # FloatTensor
        adj = preprocess_adj(adj, normalization=args.normalization).to('cuda')
    else:
        features = feature_process_jinan(features, G_edge_list_attr).to(device='cpu')
        adj = preprocess_adj(adj, normalization=args.normalization).to('cpu')


    '''normallize'''
    normed_ways_segment_volume_dict, volume_sqrt_var_list, volume_mean_list, unnormed_ways_segment_volume_dict = norm_volume(ways_segment_volume_dict)
    volume_sqrt_var_list = mask_list(volume_sqrt_var_list, 1.)
    volume_mean_list = mask_list(volume_mean_list, 0.)


    '''delet abnormal sensors & splite data'''
    unnormed_ways_segment_volume_dict.pop(109) # 85
    unnormed_ways_segment_volume_dict.pop(71)
    # unnormed_ways_segment_volume_dict.pop(85)
    if args.matual_split:
        train_ways_segment_volume_dict, test_ways_segment_volume_dict = matual_split_data(unnormed_ways_segment_volume_dict)
    else:
        data_feature, data_target = preprocess_split_data(unnormed_ways_segment_volume_dict)
        train_volume_arr, test_volume_arr, train_leida_id_arr, test_leida_id_arr = \
                train_test_split(data_feature, data_target, test_size=args.percent, random_state=args.seed)
        train_ways_segment_volume_dict = combine_ways_segment_volume_dict(train_leida_id_arr, train_volume_arr)
        test_ways_segment_volume_dict  = combine_ways_segment_volume_dict(test_leida_id_arr, test_volume_arr)


    set_seed(args.seed, args.cuda)


    '''train & evaluate model'''
    model = JINAN_model(model_type=args.model, num_head=args.num_head ,num_slice=args.num_slice, nfeat=features.shape[1], nhid=args.hidden, nclass=args.output_dim, dropout=args.dropout, degree=args.degree)

    if args.model == "GCN":
        weight_adj_list = 0.
        leida_pre_MAPE_info_y, leida_pre_MAPE_info_y_head, leida_pre_RMSE_info, mean_mape = train_regression(model, weight_adj_list, features, train_ways_segment_volume_dict, test_ways_segment_volume_dict, unnormed_ways_segment_volume_dict, volume_sqrt_var_list, volume_mean_list, G_2, adj, args.weight_decay, args.lr, args.dropout, config)

    show_pre_info(leida_pre_MAPE_info_y, leida_pre_RMSE_info, way_segments_cams_dict)
    print("over!")

    return mean_mape


def objective_volume_current(train_ways_segment_volume_dict, train_ways_segment_vec_dict, topk, negk):
    pre_volume  = []
    true_volume = []
    loss_term = 0.
    for k1,v1 in train_ways_segment_vec_dict.items():
        num_slice = v1.shape[0]
        for i in range(num_slice):
            curr_score_dict, recent_score_dict, period_score_dict = {}, {}, {}
            for k2, v2 in train_ways_segment_vec_dict.items():
                if(k1 != k2):
                    curr_score   = torch.cosine_similarity(v1[i], v2[i], dim=-1)
                    curr_score_dict[k2]   = curr_score
            cur_sum_volume_max, cur_sum_sim_score        =  0, .0
            cur_sorted_score_dict_max    = sorted(curr_score_dict.items(),   key=lambda item:item[1], reverse = True)[:topk]
            for truple in cur_sorted_score_dict_max:
                cur_sum_volume_max = cur_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i]*truple[1]
                cur_sum_sim_score = cur_sum_sim_score + truple[1]
            cur_pre_volume = cur_sum_volume_max / cur_sum_sim_score
            loss_term = abs((train_ways_segment_volume_dict[k1][i] - cur_pre_volume))**3 + loss_term
    return loss_term


def objective_volume_recent(train_ways_segment_volume_dict, train_ways_segment_vec_dict, topk, negk):
    pre_volume  = []
    true_volume = []
    loss_term = 0.

    for k1,v1 in train_ways_segment_vec_dict.items():
        num_slice = v1.shape[0]

        for i in range(num_slice):
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
            recent_sum_volume_max, recent_sum_sim_score  =  0, .0
            period_sum_volume_max, period_sum_sim_score  =  0, .0

            if i == 1:
                recent_sorted_score_dict_max = sorted(recent_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]
                for truple in recent_sorted_score_dict_max:
                    recent_sum_volume_max = recent_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-1]*truple[1]
                    recent_sum_sim_score = recent_sum_sim_score + truple[1]
                recent_pre_volume = recent_sum_volume_max / recent_sum_sim_score
                loss_term = abs((train_ways_segment_volume_dict[k1][i] - recent_pre_volume))**3 + loss_term

            if i >= 2:
                recent_sorted_score_dict_max = sorted(recent_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]
                period_sorted_score_dict_max = sorted(period_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]

                for truple in recent_sorted_score_dict_max:
                    recent_sum_volume_max = recent_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-1]*truple[1]
                    recent_sum_sim_score = recent_sum_sim_score + truple[1]
                recent_pre_volume = recent_sum_volume_max / recent_sum_sim_score
                loss_term = abs((train_ways_segment_volume_dict[k1][i] - recent_pre_volume))**3 + loss_term

                for truple in period_sorted_score_dict_max:
                    period_sum_volume_max = period_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-2]*truple[1]
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
            for k2, v2 in train_ways_segment_vec_dict.items():
                if(k1 != k2):
                    if i < 12:
                        break
                    elif i < 12*2:
                        daily1_score   = torch.cosine_similarity(v1[i-12], v2[i-12], dim=-1)
                        daily1_score_dict[k2]   = daily1_score
                    else:
                        daily1_score   = torch.cosine_similarity(v1[i-12], v2[i-12], dim=-1)
                        daily2_score   = torch.cosine_similarity(v1[i-12*2], v2[i-12*2], dim=-1)
                        daily1_score_dict[k2]  = daily1_score
                        daily2_score_dict[k2]  = daily2_score

            daily1_sum_volume_max, daily1_sum_sim_score  =  0, .0
            daily2_sum_volume_max, daily2_sum_sim_score  =  0, .0

            if i < 12:
                continue
            elif i < 12*2:
                daily1_sorted_score_dict_max = sorted(daily1_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]

                for truple in daily1_sorted_score_dict_max:
                    daily1_sum_volume_max = daily1_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-12]*truple[1]
                    daily1_sum_sim_score = daily1_sum_sim_score + truple[1]
                daily1_pre_volume = daily1_sum_volume_max / daily1_sum_sim_score
                loss_term = abs(train_ways_segment_volume_dict[k1][i] - daily1_pre_volume)**3 + loss_term

            else:
                daily1_sorted_score_dict_max = sorted(daily1_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]
                daily2_sorted_score_dict_max = sorted(daily2_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]

                for truple in daily1_sorted_score_dict_max:
                    daily1_sum_volume_max = daily1_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-12]*truple[1]
                    daily1_sum_sim_score = daily1_sum_sim_score + truple[1]
                daily1_pre_volume = daily1_sum_volume_max / daily1_sum_sim_score
                loss_term = abs(train_ways_segment_volume_dict[k1][i] - daily1_pre_volume)**3 + loss_term

                for truple in daily2_sorted_score_dict_max:
                    daily2_sum_volume_max = daily2_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-48]*truple[1]
                    daily2_sum_sim_score = daily2_sum_sim_score + truple[1]
                daily2_pre_volume = daily2_sum_volume_max / daily2_sum_sim_score
                loss_term = abs((train_ways_segment_volume_dict[k1][i] - daily2_pre_volume))**3 + loss_term
    return loss_term


def objective_volume_weekly(train_ways_segment_volume_dict, train_ways_segment_vec_dict, topk, negk):
    pre_volume  = []
    true_volume = []
    loss_term   = 0.

    for k1,v1 in train_ways_segment_vec_dict.items():
        num_slice = v1.shape[0]
        weekly1_score_dict, weekly2_score_dict = {}, {}
        for i in range(num_slice):
            for k2, v2 in train_ways_segment_vec_dict.items():
                if(k1 != k2):
                    if i < 12*7:
                        break
                    elif i < 12*14:
                        weekly1_score   = torch.cosine_similarity(v1[i-12*7], v2[i-12*7], dim=-1)
                        weekly1_score_dict[k2]   = weekly1_score
                    else:
                        weekly1_score   = torch.cosine_similarity(v1[i-12*7], v2[i-12*7], dim=-1)
                        weekly2_score   = torch.cosine_similarity(v1[i-12*14], v2[i-12*14], dim=-1)
                        weekly1_score_dict[k2]  = weekly1_score
                        weekly2_score_dict[k2]  = weekly2_score

            weekly1_sum_volume_max, weekly1_sum_sim_score  =  0, .0
            weekly2_sum_volume_max, weekly2_sum_sim_score  =  0, .0

            if i < 12*7:
                break
            elif i < 12*14:
                weekly1_sorted_score_dict_max = sorted(weekly1_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]

                for truple in weekly1_sorted_score_dict_max:
                    weekly1_sum_volume_max = weekly1_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-12*7]*truple[1]
                    weekly1_sum_sim_score = weekly1_sum_sim_score + truple[1]
                weekly1_pre_volume = weekly1_sum_volume_max / weekly1_sum_sim_score
                loss_term = abs(train_ways_segment_volume_dict[k1][i] - weekly1_pre_volume)**3 + loss_term

            else:
                weekly1_sorted_score_dict_max = sorted(weekly1_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]
                weekly2_sorted_score_dict_max = sorted(weekly2_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]

                for truple in weekly1_sorted_score_dict_max:
                    weekly1_sum_volume_max = weekly1_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-12*7]*truple[1]
                    weekly1_sum_sim_score = weekly1_sum_sim_score + truple[1]
                weekly1_pre_volume = weekly1_sum_volume_max / weekly1_sum_sim_score
                loss_term = abs(train_ways_segment_volume_dict[k1][i] - weekly1_pre_volume)**3 + loss_term

                for truple in weekly2_sorted_score_dict_max:
                    weekly2_sum_volume_max = weekly2_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-12*14]*truple[1]
                    weekly2_sum_sim_score = weekly2_sum_sim_score + truple[1]
                weekly2_pre_volume = weekly2_sum_volume_max / weekly2_sum_sim_score
                loss_term = abs((train_ways_segment_volume_dict[k1][i] - weekly2_pre_volume))**3 + loss_term
    return loss_term

def objective_volume_recent_unmonitored(output_embedding, train_ways_segment_volume_dict, train_ways_segment_vec_dict, topk, negk):

    pre_volume  = []
    true_volume = []
    loss_term = 0.
    all_road_segments_set = set([i for i in range(output_embedding.shape[1])])
    monitored_set= set(train_ways_segment_volume_dict.keys())
    unmonitored_set = all_road_segments_set - monitored_set

    for unmonitored_road in unmonitored_set:
        num_slice = output_embedding.shape[0]
        cur_score_dict, recent_score_dict, period_score_dict = {}, {}, {}
        for i in range(num_slice):
            for monitored_road_target in monitored_set:

                cur_score = torch.cosine_similarity(output_embedding[i][unmonitored_road], output_embedding[i][monitored_road_target], dim=-1)
                cur_score_dict[monitored_road_target] = cur_score

                if i>=2:
                    recent_score = torch.cosine_similarity(output_embedding[i-1][unmonitored_road], output_embedding[i-1][monitored_road_target], dim=-1)
                    period_score = torch.cosine_similarity(output_embedding[i-2][unmonitored_road], output_embedding[i-2][monitored_road_target], dim=-1)
                    recent_score_dict[monitored_road_target] = recent_score
                    period_score_dict[monitored_road_target] = period_score
                elif i==1:
                    recent_score = torch.cosine_similarity(output_embedding[i-1][unmonitored_road], output_embedding[i-1][monitored_road_target], dim=-1)
                    recent_score_dict[monitored_road_target] = recent_score

            cur_sum_volume_max, cur_sum_sim_score = 0, 0
            cur_sorted_score_dict_max = sorted(cur_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]
            for truple in cur_sorted_score_dict_max:
                cur_sum_volume_max = cur_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i]*truple[1]
                cur_sum_sim_score = cur_sum_sim_score + truple[1]
            cur_true_volume = cur_sum_volume_max / cur_sum_sim_score


            recent_sum_volume_max, recent_sum_sim_score  =  0, .0
            period_sum_volume_max, period_sum_sim_score  =  0, .0

            if i == 1:
                recent_sorted_score_dict_max = sorted(recent_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]
                for truple in recent_sorted_score_dict_max:
                    recent_sum_volume_max = recent_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-1]*truple[1]
                    recent_sum_sim_score = recent_sum_sim_score + truple[1]
                recent_pre_volume = recent_sum_volume_max / recent_sum_sim_score
                loss_term = abs((cur_true_volume - recent_pre_volume)) + loss_term
            if i >= 2:
                recent_sorted_score_dict_max = sorted(recent_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]
                period_sorted_score_dict_max = sorted(period_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]

                for truple in recent_sorted_score_dict_max:
                    recent_sum_volume_max = recent_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-1]*truple[1]
                    recent_sum_sim_score = recent_sum_sim_score + truple[1]
                recent_pre_volume = recent_sum_volume_max / recent_sum_sim_score
                loss_term = abs((cur_true_volume - recent_pre_volume)) + loss_term

                for truple in period_sorted_score_dict_max:
                    period_sum_volume_max = period_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-2]*truple[1]
                    period_sum_sim_score = period_sum_sim_score + truple[1]
                period_pre_volume = period_sum_volume_max/period_sum_sim_score
                loss_term = abs((cur_true_volume - period_pre_volume)) + loss_term

    return loss_term

def objective_volume_daily_unmonitored(output_embedding, train_ways_segment_volume_dict, train_ways_segment_vec_dict, topk, negk):

    pre_volume  = []
    true_volume = []
    loss_term = 0.
    all_road_segments_set = set([i for i in range(output_embedding.shape[1])])
    monitored_set= set(train_ways_segment_volume_dict.keys())
    unmonitored_set = all_road_segments_set - monitored_set

    for unmonitored_road in unmonitored_set:
        num_slice = output_embedding.shape[0]
        cur_score_dict, daily1_score_dict, daily2_score_dict = {}, {}, {}
        for i in range(num_slice):
            for monitored_road_target in monitored_set:
                cur_score = torch.cosine_similarity(output_embedding[i][unmonitored_road], output_embedding[i][monitored_road_target], dim=-1)
                cur_score_dict[monitored_road_target] = cur_score


                if i < 12: # TODO:12slice
                    break
                elif i < 12*2:
                    daily1_score   = torch.cosine_similarity(output_embedding[i-12][unmonitored_road], output_embedding[i-12][monitored_road_target], dim=-1)
                    daily1_score_dict[monitored_road_target]   = daily1_score
                else:
                    daily1_score   = torch.cosine_similarity(output_embedding[i-12][unmonitored_road], output_embedding[i-12][monitored_road_target], dim=-1)
                    daily2_score   = torch.cosine_similarity(output_embedding[i-12*2][unmonitored_road], output_embedding[i-12*2][monitored_road_target], dim=-1)
                    daily1_score_dict[monitored_road_target]  = daily1_score
                    daily2_score_dict[monitored_road_target]  = daily2_score

            cur_sum_volume_max, cur_sum_sim_score = 0, 0
            cur_sorted_score_dict_max = sorted(cur_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]
            for truple in cur_sorted_score_dict_max:
                cur_sum_volume_max = cur_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i]*truple[1]
                cur_sum_sim_score = cur_sum_sim_score + truple[1]
            cur_true_volume = cur_sum_volume_max / cur_sum_sim_score

            daily1_sum_volume_max, daily1_sum_sim_score  =  0, .0
            daily2_sum_volume_max, daily2_sum_sim_score  =  0, .0


            if i < 12:
                continue
            elif i < 12*2:
                daily1_sorted_score_dict_max = sorted(daily1_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]
                for truple in daily1_sorted_score_dict_max:
                    recent_sum_volume_max = daily1_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-12]*truple[1]
                    daily1_sum_sim_score = daily1_sum_sim_score + truple[1]
                daily1_pre_volume = daily1_sum_volume_max / daily1_sum_sim_score
                loss_term = abs((cur_true_volume - daily1_pre_volume))**3 + loss_term
            else:
                daily1_sorted_score_dict_max = sorted(daily1_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]
                daily2_sorted_score_dict_max = sorted(daily2_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]

                for truple in daily1_sorted_score_dict_max:
                    daily1_sum_volume_max = daily1_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-12]*truple[1]
                    daily1_sum_sim_score = daily1_sum_sim_score + truple[1]
                daily1_pre_volume = daily1_sum_volume_max / daily1_sum_sim_score
                loss_term = abs(cur_true_volume - daily1_pre_volume)**3 + loss_term

                for truple in daily2_sorted_score_dict_max:
                    daily2_sum_volume_max = daily2_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-48]*truple[1]
                    daily2_sum_sim_score = daily2_sum_sim_score + truple[1]
                daily2_pre_volume = daily2_sum_volume_max / daily2_sum_sim_score
                loss_term = abs((cur_true_volume - daily2_pre_volume))**3 + loss_term
    return loss_term


def objective_volume_weekly_unmonitored(output_embedding, train_ways_segment_volume_dict, train_ways_segment_vec_dict, topk, negk):
    pre_volume  = []
    true_volume = []
    loss_term = 0.
    all_road_segments_set = set([i for i in range(output_embedding.shape[1])])
    monitored_set= set(train_ways_segment_volume_dict.keys())
    unmonitored_set = all_road_segments_set - monitored_set

    for unmonitored_road in unmonitored_set:
        num_slice = output_embedding.shape[0]
        cur_score_dict, weekly1_score_dict, weekly2_score_dict = {}, {}, {}
        for i in range(num_slice):
            for monitored_road_target in monitored_set:
                cur_score = torch.cosine_similarity(output_embedding[i][unmonitored_road], output_embedding[i][monitored_road_target], dim=-1)
                cur_score_dict[monitored_road_target] = cur_score


                if i < 12*7:
                    break
                elif i < 12*14:
                    weekly1_score   = torch.cosine_similarity(output_embedding[i-12*7][unmonitored_road], output_embedding[i-12*7][monitored_road_target], dim=-1)  # TODO:12slice
                    weekly1_score_dict[monitored_road_target]   = weekly1_score
                else:
                    weekly1_score   = torch.cosine_similarity(output_embedding[i-12*7][unmonitored_road], output_embedding[i-12*7][monitored_road_target], dim=-1)  # TODO:12slice
                    weekly2_score   = torch.cosine_similarity(output_embedding[i-12*14][unmonitored_road], output_embedding[i-12*14][monitored_road_target], dim=-1)  # TODO:12slice
                    weekly1_score_dict[monitored_road_target]   = weekly1_score
                    weekly2_score_dict[monitored_road_target]   = weekly2_score

            cur_sum_volume_max, cur_sum_sim_score = 0, 0
            cur_sorted_score_dict_max = sorted(cur_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]
            for truple in cur_sorted_score_dict_max:
                cur_sum_volume_max = cur_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i]*truple[1]
                cur_sum_sim_score = cur_sum_sim_score + truple[1]
            cur_true_volume = cur_sum_volume_max / cur_sum_sim_score

            weekly1_sum_volume_max, weekly1_sum_sim_score  =  0, .0
            weekly2_sum_volume_max, weekly2_sum_sim_score  =  0, .0


            if i < 12*7:
                continue
            elif i < 12*14:
                weekly1_sorted_score_dict_max = sorted(weekly1_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]

                for truple in weekly1_sorted_score_dict_max:
                    weekly1_sum_volume_max = weekly1_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-12*7]*truple[1]
                    weekly1_sum_sim_score = weekly1_sum_sim_score + truple[1]
                weekly1_pre_volume = weekly1_sum_volume_max / weekly1_sum_sim_score
                loss_term = abs(cur_true_volume - weekly1_pre_volume)**3 + loss_term


            else:
                weekly1_sorted_score_dict_max = sorted(weekly1_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]
                weekly2_sorted_score_dict_max = sorted(weekly2_score_dict.items(), key=lambda item:item[1], reverse = True)[:topk]

                for truple in weekly1_sorted_score_dict_max:
                    weekly1_sum_volume_max = weekly1_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-12*7]*truple[1]
                    weekly1_sum_sim_score = weekly1_sum_sim_score + truple[1]
                weekly1_pre_volume = weekly1_sum_volume_max / weekly1_sum_sim_score
                loss_term = abs(cur_true_volume - weekly1_pre_volume)**3 + loss_term

                for truple in weekly2_sorted_score_dict_max:
                    weekly2_sum_volume_max = weekly2_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i-12*14]*truple[1]
                    weekly2_sum_sim_score = weekly2_sum_sim_score + truple[1]
                weekly2_pre_volume = weekly2_sum_volume_max / weekly2_sum_sim_score
                loss_term = abs((cur_true_volume - weekly2_pre_volume))**3 + loss_term
    return loss_term


def objective_rw( train_ways_segment_vec_dict, negk, adj_weight_dict, output, vocab_list, word_freqs):

    loss_term = torch.tensor(0.,dtype=torch.float32)
    for k1,v1 in train_ways_segment_vec_dict.items():
        negative_list = []
        cur_adj = adj_weight_dict[k1] # {    0: {1: {'weight': 0.7310585786300049}, 131: {'weight': 0.7310585786300049}}     }
        while( len(set(negative_list) - set(cur_adj.keys())) < negk ):
            negative_list = random.choices(population=vocab_list, weights=word_freqs, k=negk)
        positive_embedding = output[:, list(cur_adj.keys())]
        weight_positive_tensor = torch.tensor([item["weight"] for item in list(cur_adj.values())],dtype=torch.float32).unsqueeze(0)
        cur_embedding = output[:,k1]
        negative_embedding = output[:, negative_list]

        cur_loss_term =  - torch.sum(F.logsigmoid(torch.cosine_similarity( positive_embedding, cur_embedding.unsqueeze(1), dim=-1).unsqueeze(1) )) \
                         - torch.sum(torch.log( 1. - torch.sigmoid(torch.cosine_similarity(negative_embedding, cur_embedding.unsqueeze(1), dim=-1 ) ) ))
        loss_term = loss_term + cur_loss_term
    return loss_term


def train_regression(model, weight_adj_list, train_features, train_ways_segment_volume_dict,
                     test_ways_segment_volume_dict, unnormed_ways_segment_volume_dict, volume_sqrt_var, volume_mean, G, adj,
                     weight_decay, lr, dropout, config ):
    epochs = config['epochs']
    hy_RW = config['hy_RW']
    hy_volume_current = config['hy_volume_current']
    hy_volume_recent = config['hy_volume_recent']
    hy_volume_daily = config['hy_volume_daily']
    hy_volume_weekly = config['hy_volume_weekly']
    hy_unvolume_recent = config['hy_unvolume_recent']
    hy_unvolume_daily = config['hy_unvolume_daily']
    hy_unvolume_weekly = config['hy_unvolume_weekly']


    '''objective_rw'''
    walker = RWGraph(G)
    walks_list = walker.simulate_walks(args.num_walks, args.walk_length, schema=None, isweighted=args.isweighted)

    walks_list = [col for row in walks_list for col in row]
    vocab_list = Counter(walks_list).most_common() # 每个元素是一个元组[(539,347), (457,333)...]

    word_counts = np.array([count[1] for count in vocab_list], dtype=np.float32) #
    word_freqs = word_counts / np.sum(word_counts)
    word_freqs = word_freqs ** (3. / 4.)
    adj_weight_dict = find_positive_samples(G)
    vocab_list = [item[0] for item in vocab_list]

    criterion = nn.MSELoss()
    train_ways_segment_list = list(train_ways_segment_volume_dict.keys())

    all_epoch_mape_y,all_epoch_mape_y_head, all_epoch_RMSE = [], [], []
    params_list = []
    for i in range(args.num_slice):
        params_list.append({"params":model.model_list[i].parameters()})

    for i in range(args.num_head):
        params_list.append({"params":model.attention.at_block_list[i].parameters()})


    optimizer = optim.Adam( params_list, lr=lr, weight_decay=weight_decay)
    t = perf_counter()

    per_list = []
    for epoch in range(epochs):
        train_ways_segment_vec_dict = {}
        model.train()
        optimizer.zero_grad()
        if args.model == "SGC":
            output = model(train_features, weight_adj_list)
        if args.model == "GCN":
            output = model(train_features, adj)

        for i, item in enumerate(train_ways_segment_list):
            train_ways_segment_vec_dict[item] = output[:, item, :]
        loss_train_volume_current = objective_volume_current(train_ways_segment_volume_dict, train_ways_segment_vec_dict, args.topk, args.negk)
        loss_train_volume_recent  = objective_volume_recent(train_ways_segment_volume_dict, train_ways_segment_vec_dict, args.topk, args.negk)
        loss_train_volume_daily   = objective_volume_daily(train_ways_segment_volume_dict, train_ways_segment_vec_dict, args.topk, args.negk)
        loss_train_volume_weekly  = objective_volume_weekly(train_ways_segment_volume_dict, train_ways_segment_vec_dict, args.topk, args.negk)

        loss_train_volume_recent_unmonitored  = objective_volume_recent_unmonitored(output, train_ways_segment_volume_dict, train_ways_segment_vec_dict, args.topk, args.negk)
        loss_train_volume_daily_unmonitored   = objective_volume_daily_unmonitored(output, train_ways_segment_volume_dict, train_ways_segment_vec_dict, args.topk, args.negk)
        loss_train_volume_weekly_unmonitored  = objective_volume_weekly_unmonitored(output, train_ways_segment_volume_dict, train_ways_segment_vec_dict, args.topk, args.negk)


        loss_train_rw = objective_rw(train_ways_segment_vec_dict, args.negk, adj_weight_dict, output, vocab_list, word_freqs)
        loss = hy_volume_current*loss_train_volume_current + hy_volume_recent*loss_train_volume_recent + \
               hy_RW*loss_train_rw +  \
               hy_volume_daily*loss_train_volume_daily + hy_volume_weekly*loss_train_volume_weekly + \
               hy_unvolume_recent*loss_train_volume_recent_unmonitored + \
               hy_unvolume_daily*loss_train_volume_daily_unmonitored + \
               hy_unvolume_weekly* loss_train_volume_weekly_unmonitored


        loss.backward()
        optimizer.step()

        if (epoch) % 2 == 0:
            with open('jinan/train_log.txt', 'a', encoding='utf-8') as f:
                with torch.no_grad():
                    train_ways_segment_vec_dict = {}
                    model.eval()
                    if args.model == "SGC":
                        output = model(weight_adj_list, train_features)
                    if args.model == "GCN":
                        output = model(train_features, adj)
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

                    per_list.append(calc_avg_dict_value(leida_pre_MAPE_info_y))

                    f.write("epoch: {}\n".format(epoch))
                    f.write("mean MAPE_y: {}\n".format(calc_avg_dict_value(leida_pre_MAPE_info_y)))
                    f.write("mean MAPE_y_head: {}\n".format(calc_avg_dict_value(leida_pre_MAPE_info_y_head)))
                    f.write("mean RMSE: {}\n".format(calc_avg_dict_value(leida_pre_RMSE_info)))
                    all_epoch_mape_y.append(calc_avg_dict_value(leida_pre_MAPE_info_y))
                    all_epoch_mape_y_head.append(calc_avg_dict_value(leida_pre_MAPE_info_y_head))
                    all_epoch_RMSE.append(calc_avg_dict_value(leida_pre_RMSE_info))
    train_time = perf_counter()-t
    print('In this trial, avg_mape_y:{},  avg_mape_y_head:{},   avg_RMSE：{}'.format(np.mean(all_epoch_mape_y), np.mean(all_epoch_mape_y_head), np.mean(all_epoch_RMSE)))
    return leida_pre_MAPE_info_y, leida_pre_MAPE_info_y_head, leida_pre_RMSE_info, min(per_list)


if __name__=="__main__":
    args = get_args()

    '''0. optuna train'''
    study_name = 'jinan_study'
    args = get_args()
    study = optuna.create_study(study_name=study_name, storage='sqlite:///jinan/log/jinan_study.db', load_if_exists=True)
    study.optimize(objective, n_trials=args.n_trials)
    print('over!')

    # '''1. read db file, find the best parameters'''
    # study = optuna.create_study(study_name='jinan_study', storage='sqlite:///jinan/log/jinan_study_100_seed2.db', load_if_exists=True)
    # df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    # print(df)
    # df.to_csv('jinan\log\{}_jinan_study.csv'.format(time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))))
    # print('over!')


    # '''2. visualization'''
    # import plotly as py
    # study = optuna.create_study(study_name='jinan_study', storage='sqlite:///jinan/log/jinan_study_100_sed2.db', load_if_exists=True)
    # fig = optuna.visualization.plot_parallel_coordinate(study, params=['hy_RW', 'hy_volume_current', 'hy_volume_recent', 'hy_volume_daily', 'hy_volume_weekly'])
    # py.offline.plot(fig,auto_open=True) # filename="iris1.html"
    # print('over!')

