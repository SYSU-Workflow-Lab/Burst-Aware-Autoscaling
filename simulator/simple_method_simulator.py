from basic_env.basic_env import BasicEnv
def threshold_simulate(env):
    """
        使用阈值法对指定环境进行模拟，收集全部奖励信息并进行输出
    """
    state = env.reset()
    episode_return = 0.0  # sum of rewards in an episode
    episode_step = 1
    max_step = env.max_step

    state = env.reset()
    response_up_threshold = 1. / 8.
    cpu_down_threshold = 0.2
    ins_list = []
    wok_list = []
    rew_list = []
    for episode_step in range(max_step):
        workload, cpu, response_time = state
        wok_list.append(workload * env.instance)
        instance = env.instance
        if response_time > response_up_threshold:
            instance += 1
        elif cpu < cpu_down_threshold:
            instance -= 1
        action = instance - env.instance + 2
        state, reward, done, _ = env.step(action)
        ins_list.append(env.instance)
        rew_list.append(reward)
        episode_return += reward
        if done:
            break
    episode_return = getattr(env, 'episode_return', episode_return)
    return episode_return, episode_step

def simulate_threshold_scaler(workload_name='North_Korea'):
    env = BasicEnv(workload_name=workload_name)
    r,t = threshold_simulate(env)
    print(r,t)

if __name__ == '__main__':
    simulate_threshold_scaler()