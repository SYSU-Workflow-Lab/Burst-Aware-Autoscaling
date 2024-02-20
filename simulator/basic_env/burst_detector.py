from util.constdef import TEST_MODE, VALI_MODE, TRAIN_MODE, BURSTY_CACHE_DIR, create_folder_if_not_exist, LONGTERM_CONF_RATIO
from util.metric import QuantileLoss
import numpy as np
from util.metric import SMAPE
import os
import logging
import math
from tqdm import trange
"""
职能：使用给定的流量数据进行评估。给定流量值
问题：
* is_burst该如何与训练数据/测试数据进行对应
    * 目前采取的方法是，不同段直接给予不同的数据
"""
import matplotlib.pyplot as plt

def calculate_longterm_accuracy(longterm_prediction_result, real_result, train_step=10000,ratio=0.9):
    """
    longterm_prediction_result[i]对应的是real_result[i:i+seq_len]
    返回P90精确度
    """
    quantile = LONGTERM_CONF_RATIO
    qrisk = QuantileLoss(quantiles=[1-quantile,0.5,quantile])
    seq_len = longterm_prediction_result.shape[1]
    acc_list = []
    for i in range(train_step):
        for j in range(seq_len):
            true = real_result[i+j]
            pred = longterm_prediction_result[i,j]
            acc_list.append(qrisk.numpy_normalised_quantile_loss(pred,true))
    acc_list = np.array(acc_list)
    acc_list = np.sort(acc_list)
    # # DEBUG
    # acc_list[acc_list>0.2] = 0
    # plt.hist(acc_list,bins=100)
    # plt.show()
    acc_threshold = acc_list[int(len(acc_list) * ratio)]
    mu = np.mean(acc_list)
    std = np.std(acc_list)
    return mu + 3 * std

class BurstDetector:
    def __init__(self, preds, real_data, one_step_pred, workload_name, train_step=10000, vali_step=3000, test_step=3000,is_retrain=False,max_reward_count=100):
        """
            读取流量数据信息，设定超参数，并对流量的质量进行评估
        """
        self.is_retrain=is_retrain
        self.max_reward_count=max_reward_count

        self.workload_name = workload_name
        self.train_step = train_step
        self.vali_step = vali_step
        self.test_step = test_step
        self.__analyze_prediction(preds,real_data,one_step_pred)
        self.reset()
        

    def __analyze_prediction(self, longterm_result_total, real_result,one_step_pred):
        """
            longterm_result: (ts,24,3)
            real_result: (ts+seq_len)
        """
        seq_len = longterm_result_total.shape[1]

        # NOTE 以下为不可预期的判断，判断是否为burst状态
        score_threshold = 0.01 # 取消偏移点的问题，仅考虑综合违约程度
        total_steps = self.train_step + self.vali_step + self.test_step
        is_burst = np.zeros(total_steps)
        # 数据校验：
        # 第一个点的数据，所使用的预测数据->对应的真实数据的最后一个点应该是真实环境的第一个数据
        # * 因此，为了保持对齐，其他数据会向后偏移24个点，但是burst_detection中会对此进行判断，所以最终的结果是一样的。
        # ? 不需要纠结对齐问题，这里的数据是被手动平移过的
        # * 可以保证，在当前点出发进行的burst判断，其所使用的真实数据没有超过这个点的数据（但是一定会包括这个点）
        logging.info("start to calculate burst array")
        cache_directory_name = f"{self.workload_name}_{self.train_step}_{self.vali_step}_{self.test_step}"
        folder_path = os.path.join(BURSTY_CACHE_DIR,cache_directory_name)
        file_name = "is_burst.npy"
        # DEBUG 强制设为True
        # self.is_retrain = True
        if self.is_retrain or not os.path.exists(folder_path):
            # NOTE origin_burst计算
            # 1. 遍历所有的训练集数据（最佳容器数据），收集k窗口内的最大值
            # k=10
            # sigma_up_thre = 5
            # # 3. 对当前的所有的数据进行求解，得到最终的结果
            # all_workload_data = real_result[seq_len-1:seq_len-1+self.train_step+self.vali_step+self.test_step]
            # all_optimal_ins = np.zeros_like(all_workload_data)
            # for i in range(len(all_workload_data)):
            #     all_optimal_ins[i] = math.ceil(all_workload_data[i] / self.max_reward_count)
            
            # is_origin_burst = np.zeros_like(all_optimal_ins)
            # for i in trange(len(is_origin_burst)):
            #     if i-k<0:
            #         back_k_array = all_optimal_ins[:i+1]
            #     else:
            #         back_k_array = all_optimal_ins[i-k:i+1] # 必须包含i位置，总共应该有k+1个。第i个位置此时对应着未来的预测值

            #     sigma_max = 0
            #     for j in range(1,k+1):
            #         arr = back_k_array[-j-1:]
            #         if len(arr) <=1 or len(back_k_array) < j+1:
            #             break
            #         sigma = np.std(arr)
            #         if sigma > sigma_max:
            #             sigma_max = sigma
            #     if sigma_max > sigma_up_thre:
            #         is_origin_burst[i] = 1

            # is_burst = is_origin_burst
            # NOTE burst计算
            onestep_acc = np.abs(one_step_pred.ravel() - real_result.ravel())[:self.train_step]
            onestep_thre = np.mean(onestep_acc) + 3 * np.std(onestep_acc)
            # * 使用训练段的全体预测结果进行计算，并将每一个点都单独地分离出来进行存储，然后进行误差分析
            acc_threshold = calculate_longterm_accuracy(longterm_result_total, real_result, self.train_step)

            # * 计算出均值和方差，传入burst判断
            # ? burst中实时计算所涉及到对应的部分（pred,true）的预测精度，如果超过对应的上界，则认为当前预测失败
            logging.info(f"train {self.workload_name} to get is_burst array. (is_retrain={self.is_retrain})") 
            for i in trange(total_steps):
            # if i<seq_len-1: # 如果数量过小，则直接忽略
            #     continue
                is_burst[i] = burst_detection(is_burst[i-24:i],
                                        preds=longterm_result_total[i:i+seq_len],
                                        trues=np.array([real_result[j:j+seq_len] for j in range(i,i+seq_len)]),
                                        onestep=np.array([one_step_pred[j:j+seq_len] for j in range(i,i+seq_len)]),
                                        status = is_burst[i-1], 
                                        onestep_threshold = onestep_thre,
                                        score_threshold=score_threshold,
                                        accuracy_threshold = acc_threshold,
                                        avail_horizon=24) # 默认使用全部的horizon

            # 记录下来
            if not os.path.exists(folder_path):
                create_folder_if_not_exist(folder_path)
            np.save(os.path.join(folder_path, file_name), is_burst)
        else:
            # 读取
            is_burst = np.load(os.path.join(folder_path, file_name))
        # 顺序扫描得到burst的起止点

        self.burst_array = is_burst
        self.train_burst_array, self.train_non_burst_array = self.generate_burst_point_list(is_burst[:self.train_step])
        self.vali_burst_array, self.vali_non_burst_array = self.generate_burst_point_list(is_burst[self.train_step:self.train_step+self.vali_step])
        self.test_burst_array, self.test_non_burst_array = self.generate_burst_point_list(is_burst[self.train_step+self.vali_step:self.train_step+self.vali_step+self.test_step])

    def reset(self):
        """
            重置，不管是non-burst或是burst模式
        """
        self.current_point = 0 # 指向下一个的指针
    
    def get_next_burst(self, mode=TRAIN_MODE):
        if mode == TRAIN_MODE:
            result = self.train_burst_array[self.current_point,0]
            self.current_point += 1
            if self.current_point >= len(self.train_burst_array):
                self.current_point = 0
        elif mode == VALI_MODE:
            result = self.vali_burst_array[self.current_point,0]
            self.current_point += 1
            if self.current_point >= len(self.vali_burst_array):
                self.current_point = 0
        elif mode == TEST_MODE:
            result = self.test_burst_array[self.current_point,0]
            self.current_point += 1
            if self.current_point >= len(self.test_burst_array):
                self.current_point = 0
        else:
            raise NotImplementedError
        return result

    def get_next_non_burst_without_change(self, mode=TRAIN_MODE):
        if mode == TRAIN_MODE:
            result = self.train_non_burst_array[self.current_point,0]
        elif mode == VALI_MODE:
            result = self.vali_non_burst_array[self.current_point,0]
        elif mode == TEST_MODE:
            result = self.test_non_burst_array[self.current_point,0]
        else:
            raise NotImplementedError
        return result

    def get_next_non_burst(self, mode=TRAIN_MODE):
        if mode == TRAIN_MODE:
            result = self.train_non_burst_array[self.current_point,0]
            self.current_point += 1
            if self.current_point >= len(self.train_non_burst_array):
                self.current_point = 0
        elif mode == VALI_MODE:
            result = self.vali_non_burst_array[self.current_point,0]
            self.current_point += 1
            if self.current_point >= len(self.vali_non_burst_array):
                self.current_point = 0
        elif mode == TEST_MODE:
            result = self.test_non_burst_array[self.current_point,0]
            self.current_point += 1
            if self.current_point >= len(self.test_non_burst_array):
                self.current_point = 0
        else:
            raise NotImplementedError
        return result

    def generate_burst_point_list(self,burst_array):
        """
            产生对应的序列搜索结果
        """
        burst_list = []
        non_burst_list = []
        pivot = 0
        if burst_array[pivot] == 1:
            aim_list = burst_list
        else:
            aim_list = non_burst_list
        for i in range(1, len(burst_array)):
            if burst_array[pivot] == burst_array[i]:
                continue
            # 发生切换，进行记录
            aim_list.append([pivot,i-1])
            pivot = i
            if burst_array[pivot] == 1:
                aim_list = burst_list
            else:
                aim_list = non_burst_list
        # 处理最后一个
        if burst_array[pivot] == burst_array[-1]:
            aim_list.append([pivot,len(burst_array)-1])
        return np.array(burst_list), np.array(non_burst_list)

    def is_burst(self, start_point, mode=TRAIN_MODE):
        """
            返回某个点是否为burst
        """
        if mode == TRAIN_MODE:
            return self.burst_array[start_point]
        elif mode == VALI_MODE:
            start_point += self.train_step
            return self.burst_array[start_point]
        elif mode == TEST_MODE:
            start_point += self.train_step + self.vali_step
            return self.burst_array[start_point]
        

# validator
# import matplotlib.pyplot as plt
# i = 23
# plt.plot(preds[i,:,1],color='b')
# plt.plot(preds[i,:,0],color='r')
# plt.plot(preds[i,:,2],color='r')
# plt.plot(trues[i],color='g')
# plt.show()

# ! 原则上avail_horizon不改变，变了的话长期预测不太好说明过去
# * 退出比例原则上采用6个点，如果有连续6个点的话则需要提高警戒水平
def burst_detection(is_burst_segment,preds, trues, onestep, status = False, score_threshold = 0.1, accuracy_threshold = 0.1, onestep_threshold = 200, num_score=0.1, avail_horizon=24,near_count=3, down_ratio=1./2.):
    """
        给定所有包含这一个点的预测器的预测结果（应该为24个），以及这一段的真实值打包的结果
        status表示上一个点是否为burst状态
            对于一开始的24个点，默认为non-burst状态。对于之后的部分，如果没有找到，则使用true值进行替代
        Returns:
            返回当前点是否为burst状态
        注意：
            划定可用数据界。函数只能判断当前状态是否为burst状态，以i=23为例，所有超出23的真实值都应该予以忽略
        * 具体来说，第一个预测结果有24个点可用，第二个点有23个点可用，...最后一个预测结果只有一个点是可用的。不能将未来的情况纳入到现在考虑
        ? 有一个问题，如果长期预测器的表现是处于有问题的状态，这时候应该如何去处理？
        ? 具体来说，比如长期预测在4998左右会出现SMAPE的突变，突变的原因是异常值远远大于实际情况，击穿了localMinmax，干扰了正常的预测长达168个点。
        解决方案：引入avail_horizon有效区间。即我们只考虑一段那时间内的违约情况，如果超出了这个区间则不进行考虑。
            区间长度由经验决定，一般来说，从non-burst到burst是不需要这个的，而更多与从burst状态恢复相关。6个点一般来说是足够的，4个点其实也可以。
            长期预测的质量也得到了保证，正则化方法被对应的修改了。
        Args:
            avail_horizon：计算违约情况的范围
            near_count：认为对当前状态有影响的范围（超出范围则认为对当前情况已经没有影响了）。当前点是一个确定点
            down_ratio：退出burst采用的是投票方法，当比例小于down_ratio的时候才允许退出
            score_threshold：认定当前预测结果为突发情况的分数的阈值上界
            num_score：一个越界点在突发情况中的所占的分数
    """
    quantile = LONGTERM_CONF_RATIO
    qrisk = QuantileLoss(quantiles=[1-quantile,0.5,quantile])

    ts_length, pred_len, quantile_count = preds.shape
    assert ts_length == pred_len # 这两个值必须要相等
    assert ts_length == trues.shape[0] 
    assert pred_len == trues.shape[1]
    score_list = []
    num_list = []
    prediction_acc_list = []
    onestep_acc_list = []
    # 根据给定的这部分数据，计算每一个点的outlier值和距离，并给每个预测结果计算分数
    # 根据给定的分数进行判断
    # 如果上一个点处于non-burst状态，且有2个以上点违约，总分数大于0.3，则当前处于burst；反之则为non-burst
    # 如果上一个点处于burst状态，且所有的点都小于0.3，则为non-burst；反之则继续维持在burst上。
    for i in range(len(preds)):
        # if len(preds)-i>avail_horizon:
        #     continue
        pred = preds[i,:pred_len-i,:]
        true = trues[i,:pred_len-i]
        onestep_pred = onestep[i,:pred_len-i,0]
        onestep_acc = np.abs(onestep_pred - true)
        if len(true) >= near_count:
            pred = pred[-near_count:,:]
            true = true[-near_count:]
            onestep_acc = onestep_acc[-near_count:]
        # else:
        #     break
        acc_value = qrisk.numpy_normalised_quantile_loss(pred,true)
        prediction_acc_list.append(acc_value)

        up_thre, down_thre = pred[...,2], pred[...,0]
        outlier_num = np.sum(np.logical_or(up_thre < true, down_thre > true))
        num_list.append(outlier_num)
        onestep_acc_list.append(np.mean(onestep_acc))

        up_outlier_dist = true - up_thre
        down_outlier_dist = down_thre - true
        up_outlier_dist[up_outlier_dist<0] = 0
        down_outlier_dist[down_outlier_dist<0] = 0
        outlier_dist_array = (up_outlier_dist + down_outlier_dist) / pred[...,1]
        # outlier_score = outlier_num*num_score + np.sum(outlier_dist_array)
        outlier_score = np.mean(outlier_dist_array)
        score_list.append(outlier_score)

    assert len(num_list)==avail_horizon
    is_burst = np.zeros(len(num_list))
    for i in range(len(num_list)):
        # 对于数量上大幅偏移的情况
        # ! 如果预测精确度严重偏离，比如说IQF，则不能认为是正常的i情况。
        # * 对于每个预测结果，如果说它的视界内有两个违约点且违约范围足够大（防止异常点影响），则认定为违约
        if (num_list[i] >=2 and score_list[i] > score_threshold) or prediction_acc_list[i]>accuracy_threshold:
            is_burst[i] = 1

    if not status: # non-burst状态.
        # 如果最近的数个调度器检测到异常，且最近仍然在异常中，则选择进入异常
        # 原因：non-burst情况下，大部分的数据都是正常的，因此应该尽量采用可信度最高的数据，也就是最近的阶段的数据
        if np.sum(is_burst[-near_count:])>0 and num_list[-1]>0:
            return True
        else:
            return False
    else: # burst状态下，如果burst的点小于指定比例，且最近的一个点没有违约，则允许退出。
        # 前一个条件是为了更有效地应对长期异常的情况。长期异常可能会影响到长期预测的精度，此时只能依赖于没有解除到长期预测部分的数据进行对应的预测。
        #      * burst状态下，受到异常状态的影响，异常检测可能会被影响。因此采用少数服从多数的方法进行判断。
        # 后一个条件是防止短期内再次进入异常，最近的预测判断器具备一票否决的权力
        #      * 采用np.sum(is_burst)，不对near_count进行限制，则可能需要等突发情况离开流量的预测窗口才行。在实践中，这会导致比较严重的资源浪费。
        # 最后采用最近的一个点，这样的设计是为了防止波动和反复进入的情况。
        # if np.sum(is_burst[-near_count:])==0 and np.sum(is_burst)/len(is_burst) < down_ratio and num_list[-1]==0 and np.sum(onestep_acc_list[-near_count:] < onestep_threshold):
        if np.sum(is_burst[-near_count:])==0 and num_list[-1]==0:
            # 进一步判断，之所以不直接使用最近部分判断，是因为这一部分可能受到了burst部分的影响，而且它们的预测结果也没有得到充分的验证
            # 所以需要更加严格的判断
            non_burst_part = is_burst[is_burst_segment == 0]
            if np.sum(non_burst_part) < 2:
                return False
            else:
                return True
        else:
            return True

def calculate_outlier(preds, trues):
    """
        preds:(24,3)
        trues:(24)
        计算outlier的数量，以及偏离距离
        尝试用了下中文变量，说实话看着眼睛有点累。如果需要比较复杂的算法确实是更自由一些。（算了，还是换了回去，测起来太麻烦了）
    """
    up_thre, down_thre = preds[...,2], preds[...,0]
    outlier_num = np.sum(np.logical_or(up_thre < trues, down_thre > trues))
    up_outlier_dist = trues - up_thre
    down_outlier_dist = down_thre - trues
    up_outlier_dist[up_outlier_dist<0] = 0
    down_outlier_dist[down_outlier_dist<0] = 0
    outlier_dist_array = (up_outlier_dist + down_outlier_dist) / preds[...,1]
    assert np.sum(outlier_dist_array == 0) == len(trues) - outlier_num
    if len(outlier_dist_array[outlier_dist_array>0]) == 0:
        return outlier_num, np.sum(outlier_dist_array)
    else:
        return outlier_num, np.sum(outlier_dist_array)