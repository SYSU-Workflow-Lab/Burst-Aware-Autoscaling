import time
from baseline_method.dqn.agent import *
from ppo.common import PreprocessEnv
from util.constdef import TOTAL_STEP, TRAIN_MODE, VALI_MODE, TEST_MODE
from util.constdef import save_actor_model, get_folder_name, create_folder_if_not_exist, MODEL_PREDICTOR_DIR
from ppo.validator import validate_actor
from basic_env.predict_env import APPOEnv
import logging
class Arguments:
    def __init__(self, agent=None, env=None, if_off_policy=True):
        self.agent = agent  # DRL algorithm
        self.env = env  # env for training

        self.cwd = None  # current work directory. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None)
        self.break_step = 2 ** 20  # terminate training after 'total_step > break_step'
        self.if_allow_break = True  # terminate training when reaching a target reward

        self.visible_gpu = '0'  # e.g., os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2,'
        self.worker_num = 2  # #rollout workers per GPU
        self.num_threads = 8  # cpu_num to evaluate model, torch.set_num_threads(self.num_threads)

        '''Arguments for training'''
        self.gamma = 0.99  # discount factor
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.learning_rate = 2 ** -14  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3

        if if_off_policy:  # (off-policy)
            self.net_dim = 2 ** 8  # the network width  w
            self.batch_size = self.net_dim  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
            self.target_step = 2 ** 10  # collect target_step, then update network
            self.max_memo = 2 ** 20  # capacity of replay buffer
            self.if_per_or_gae = False  # PER for off-policy sparse reward: Prioritized Experience Replay.
        else:
            self.net_dim = 2 ** 9  # the network width
            self.batch_size = self.net_dim * 2  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 3  # collect target_step, then update network
            self.target_step = 2 ** 12  # repeatedly update network to keep critic's loss small
            self.max_memo = self.target_step  # capacity of replay buffer
            self.if_per_or_gae = False  # GAE for on-policy sparse reward: Generalized Advantage Estimation.

        '''Arguments for evaluate'''
        self.eval_env = None  # the environment for evaluating. None means set automatically.
        self.eval_gap = 2 ** 6  # evaluate the agent per eval_gap seconds
        self.eval_times = 2  # number of times that get episode return in first
        self.random_seed = 0  # initialize random seed in self.init_before_training()

    def init_before_training(self,if_main=True):
        # 该值参考 constdef/save_actor_model中的对应值
        # if self.cwd is None:
        #     folder_name = get_folder_name(workload_name,env_name,use_burst=use_burst)
        #     self.cwd = os.path.join(MODEL_PREDICTOR_DIR, folder_name)
        #     create_folder_if_not_exist(self.cwd)

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.visible_gpu)


def train_and_evaluate_dqn(args, use_burst=False, agent_id=0,use_two_step=False):
    """
        兼容burst-aware
    """
    args.init_before_training(if_main=True)

    # 需要在这里设置一下args.cwd
    '''init: Agent'''
    env = args.env
    agent = args.agent
    agent.init(args.net_dim, env.state_dim, env.action_dim, args.learning_rate, args.if_per_or_gae)
    # agent.save_or_load_agent(args.cwd, if_save=False) # 尝试读取，这里选择关闭，因为读取的话可能导致不公平
    if use_two_step:
        args.break_step = int(args.env.max_step)+1 # 一个较小的量，可以设到5e4
        agent.is_action_mask = True
        agent.if_use_multistep = True
    '''init Evaluator'''
    eval_env = deepcopy(env) if args.eval_env is None else args.eval_env
    evaluator = Evaluator(args.cwd, agent_id, agent.device, eval_env,
                          args.eval_times, args.eval_gap, use_burst=use_burst)

    '''init ReplayBuffer'''
    buffer = ReplayBuffer(max_len=args.max_memo, state_dim=env.state_dim,
                          action_dim=1 if env.if_discrete else env.action_dim)
    buffer.save_or_load_history(args.cwd, if_save=False)

    def update_buffer(_trajectory):
        ten_state = torch.as_tensor([item[0] for item in _trajectory], dtype=torch.float32)
        ary_other = torch.as_tensor([item[1] for item in _trajectory])
        ary_other[:, 0] = ary_other[:, 0] * reward_scale  # ten_reward
        ary_other[:, 1] = (1.0 - ary_other[:, 1]) * gamma  # ten_mask = (1.0 - ary_done) * gamma

        buffer.extend_buffer(ten_state, ary_other)

        _steps = ten_state.shape[0]
        _r_exp = ary_other[:, 0].mean()  # other = (reward, mask, action)
        return _steps, _r_exp

    '''start training'''
    cwd = args.cwd
    gamma = args.gamma
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    reward_scale = args.reward_scale
    repeat_times = args.repeat_times
    if_allow_break = args.if_allow_break
    soft_update_tau = args.soft_update_tau

    agent.state = env.reset()
    if agent.if_off_policy:
        trajectory = agent.explore_env(env, target_step)
        update_buffer(trajectory)

    if_train = True
    while if_train:
        with torch.no_grad():
            trajectory = agent.explore_env(env, target_step)
            steps, r_exp = update_buffer(trajectory)

        logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)

        with torch.no_grad():
            if_reach_goal = evaluator.evaluate_and_save(agent.act, steps, r_exp, logging_tuple, cur_mode=TRAIN_MODE)
            if_train = not ((if_allow_break and if_reach_goal)
                            or evaluator.total_step > break_step
                            or os.path.exists(f'{cwd}/stop'))
            if not if_train and use_two_step and agent.is_action_mask:
                break_step = TOTAL_STEP
                agent.is_action_mask = False
                if_train = True
                past_max_r = evaluator.r_max
                evaluator = Evaluator(args.cwd, agent_id, agent.device, eval_env,
                          args.eval_times, args.eval_gap, use_burst=use_burst)
                evaluator.r_max = past_max_r
    print(f'| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}')
    # agent.save_or_load_agent(cwd, if_save=True)
    # buffer.save_or_load_history(cwd, if_save=True) if agent.if_off_policy else None

class Evaluator:
    def __init__(self, cwd, agent_id, device, env, eval_times, eval_gap, use_burst=False):
        self.recorder = list()  # total_step, r_avg, r_std, obj_c, ...
        self.recorder_path = f'{cwd}/recorder.npy'
        self.r_max = -np.inf
        self.total_step = 0
        self.use_burst = use_burst

        self.env = env
        self.cwd = cwd
        self.device = device
        self.agent_id = agent_id
        self.eval_gap = eval_gap
        # self.eval_times = eval_times
        self.eval_times = 1
        self.target_return = env.target_return
        self.workload_name = env.workload_name

        self.used_time = None
        self.start_time = time.time()
        self.eval_time = 0
        # early stop设置相关
        self.earlystop_counter = 0
        self.earlystop_limit = 5
        print(f"{'#' * 80}\n"
              f"{'ID':<3}{'Step':>8}{'maxR':>8} |"
              f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
              f"{'expR':>8}{'objC':>7}{'etc.':>7}")

    def evaluate_and_save(self, act, steps, r_exp, log_tuple, cur_mode=TRAIN_MODE) -> bool:
        """
            判断当前是否是最佳的模型，如果是的话就进行保存
        """
        self.total_step += steps  # update total training steps

        if time.time() - self.eval_time < self.eval_gap:
            return False  # if_reach_goal

        self.eval_time = time.time()
        rewards_steps_list = [get_episode_return_and_step(self.env, act, self.device, cur_mode=TRAIN_MODE, use_burst=self.use_burst) for _ in
                              range(self.eval_times)]
        r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)

        if r_avg > self.r_max:  # save checkpoint with highest episode return
            self.earlystop_counter = 0
            self.r_max = r_avg  # update max reward (episode return)

            save_actor_model(self.workload_name, self.env.env_name, act,self.use_burst)
            # act_save_path = f'{self.cwd}/actor.pth'
            # torch.save(act.state_dict(), act_save_path)  # save policy network in *.pth
            print(f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |")  # save policy and print
        else:
            self.earlystop_counter+=1
        self.recorder.append((self.total_step, r_avg, r_std, r_exp, *log_tuple))  # update recorder

        if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
        if self.earlystop_counter > self.earlystop_limit:
            logging.info("Early stopping")
            if_reach_goal = True
        if if_reach_goal and self.used_time is None:
            self.used_time = int(time.time() - self.start_time)
            print(f"{'ID':<3}{'Step':>8}{'TargetR':>8} |"
                  f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
                  f"{'UsedTime':>8}  ########\n"
                  f"{self.agent_id:<3}{self.total_step:8.2e}{self.target_return:8.2f} |"
                  f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
                  f"{self.used_time:>8}  ########")

        print(f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |"
              f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
              f"{r_exp:8.2f}{''.join(f'{n:7.2f}' for n in log_tuple)}")
        return if_reach_goal

    @staticmethod
    def get_r_avg_std_s_avg_std(rewards_steps_list):
        rewards_steps_ary = np.array(rewards_steps_list, dtype=np.float32)
        r_avg, s_avg = rewards_steps_ary.mean(axis=0)  # average of episode return and episode step
        r_std, s_std = rewards_steps_ary.std(axis=0)  # standard dev. of episode return and episode step
        return r_avg, r_std, s_avg, s_std


def get_episode_return_and_step(env, act, device, cur_mode=TRAIN_MODE, use_burst=False):
    episode_return = 0.0  # sum of rewards in an episode
    episode_step = 1
    max_step = env.max_step
    if_discrete = env.if_discrete

    state = env.reset(cur_mode=cur_mode)
    for episode_step in range(max_step):
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor = act(s_tensor)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
        state, reward, done, _ = env.step(action)
        episode_return += reward
        if done:
            if not use_burst: # 如果不是使用burst，则此时应该退出
                break
            else: # 如果使用burst，则应该继续进行枚举
                point1 = env.get_current_point()
                state = env.reset(cur_mode=cur_mode,restart=False)
                point2 = env.get_current_point()
                if point2 < point1: # 如果到达末尾，新的点会出现在最前面
                    break
    episode_return = getattr(env, 'episode_return', episode_return)
    return episode_return, episode_step


# 训练接口
def train_sarima_model(workload_name, action_max=5,
    train_step=10000, vali_step=3000, test_step=3000,
    use_burst=False, is_burst=False,sla=8.,des_mean=10000, des_std=3000):
    """
        use_burst=False是关闭burst-aware
            如果开启，则is_burst=False时会提供non-burst数据，is_burst=True时会提供burst数据
    """
    args = Arguments(if_off_policy=True)
    args.agent = AgentDQN()  # AgentDoubleDQN AgentDQN
    args.visible_gpu = '1'

    # 在此处设置超参数
    args.env = PreprocessEnv(env=APPOEnv(workload_name = workload_name, action_max=action_max,
        train_step=train_step, vali_step=vali_step, test_step=test_step,
        use_burst=use_burst,is_burst_bool = is_burst, sla=sla, des_mean=des_mean, des_std=des_std))
    args.env.target_return = 50000 # 目标期望，达到后训练停止
    args.break_step = TOTAL_STEP
    args.target_step = args.env.max_step 
    args.if_per_or_gae = True
    args.gamma = 0.5
    args.workload_name = workload_name

    # args.reward_scale = 2 ** -1
    # args.target_step = args.env.max_step * 8
    # args.eval_times = 2 ** 5  # evaluate times of the average episode return
    train_and_evaluate_dqn(args)

def validate_asarima_model(dataStore, workload_name,cur_mode=TRAIN_MODE,action_max=1,
    train_step=10000, vali_step=3000, test_step=3000,
    in_burst=False,train_burst=0,sla=8., des_mean=10000, des_std=3000):
    """
    in_burst表示读取的模型来自哪个文件夹(True时表示用non-burst训练，False时表示使用原来的模型)
    use_burst表示是否使用burst训练的模型
    train_burst表示是否使用burst去训练，0表示不用，1表示使用non-burst数据，2表示使用burst数据
    """
    args = Arguments()  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentDQN() # 使用离散版本的Agent，如果连续动作需要对此项进行修改
    args.visible_gpu = '1'
    # 传统
    # args.env = PreprocessEnv(env=APPOEnv(workload_name = workload_name, action_max=action_max))
    remark = ""
    is_burst = False
    if train_burst == 0: # 使用全体数据测试
        use_burst = False
        args.env = PreprocessEnv(env=APPOEnv(workload_name = workload_name, action_max=action_max,
                train_step=train_step, vali_step=vali_step, test_step=test_step,
                use_burst=use_burst,is_burst_bool = is_burst, sla=sla, des_mean=des_mean, des_std=des_std))
        remark = "baseline_dqn/non-burst-aware"
    elif train_burst == 1: # 使用non-burst数据测试
        use_burst = True
        args.env = PreprocessEnv(env=APPOEnv(workload_name = workload_name, action_max=action_max,
                train_step=train_step, vali_step=vali_step, test_step=test_step,
                use_burst=use_burst,is_burst_bool = is_burst, sla=sla, des_mean=des_mean, des_std=des_std))
        remark = "baseline_dqn/burst-aware/non-burst"
    elif train_burst == 2: # 使用burst数据测试
        use_burst = True
        is_burst = True
        args.env = PreprocessEnv(env=APPOEnv(workload_name = workload_name, action_max=action_max,
                train_step=train_step, vali_step=vali_step, test_step=test_step,
                use_burst=use_burst,is_burst_bool = is_burst, sla=sla, des_mean=des_mean, des_std=des_std))
        remark = "baseline_dqn/burst-aware/burst"
    args.env.target_return = 50000 # 目标期望，达到后训练停止
    args.target_step = args.env.max_step 
    args.if_per_or_gae = True
    args.gamma = 0.5
    args.workload_name = workload_name
    workload_name = args.workload_name
    env_name = args.env.env_name
    args.init_before_training()
    dataStore, _, _ = validate_actor(dataStore, args=args, workload_name=workload_name,cur_mode=cur_mode, res_sla=sla,
        use_burst=use_burst,in_burst=in_burst,train_burst=train_burst,remark=remark) # 使用基础动作和
    return dataStore