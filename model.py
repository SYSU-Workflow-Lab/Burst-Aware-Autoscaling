# 负责实现各种模型的接口与具体实现
import torch
from argument_manager import DQN_MODEL, PPO_MODEL, BURST_MODEL, THRE_MODEL
from simulator.util.constdef import load_actor_model
from simulator.ppo.agent import AgentDiscretePPO
from simulator.baseline_method.dqn.agent import AgentDQN
import simulator.baseline_method.dqn.run as DQNRun
import simulator.ppo.common as PPORun
class ScalerModel:
    def __init__(self, argument_manager):
        # 读取已有的模型（agent）
        self.argument_manager = argument_manager
        self.action_max = self.argument_manager.config['hyperparameter'][0]['action_max']
        self.sla = self.argument_manager.config['hyperparameter'][0]['sla']
        self.pred_action_max = self.action_max
        gpu_id = 0
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.is_threshold = False
        if argument_manager.scaler_model == PPO_MODEL or argument_manager.scaler_model == DQN_MODEL:
            if argument_manager.scaler_model == PPO_MODEL:
                is_burst = True
                args = PPORun.Arguments()
                hyper_env = PPORun.PreprocessEnv(env=self.argument_manager.env)
                self.agent = AgentDiscretePPO()
                self.agent.init(args.net_dim, hyper_env.state_dim, hyper_env.action_dim,
                    args.learning_rate, args.if_per_or_gae)
                self.act = self.agent.act
                self.act.load_state_dict(load_actor_model(workload_name=self.argument_manager.workload_name, 
                                        env_name=self.argument_manager.env.env_name, use_continuous=False,
                                        use_burst=is_burst,if_outside=True))
            else:
                is_burst = False
                args = DQNRun.Arguments()
                hyper_env = PPORun.PreprocessEnv(env=self.argument_manager.env)
                self.agent = AgentDQN()
                self.agent.init(args.net_dim, hyper_env.state_dim, hyper_env.action_dim,
                    args.learning_rate, args.if_per_or_gae)
                self.act = self.agent.act
                self.act.load_state_dict(load_actor_model(workload_name=self.argument_manager.workload_name, 
                                        env_name=self.argument_manager.env.env_name, 
                                        use_burst=is_burst,if_outside=True))
        elif argument_manager.scaler_model == THRE_MODEL:
            self.is_threshold = True
            

    def decide_action_from_state(self, state, ins_num, cpu=80, res=16):
        # 如果开启简单阈值法
        if self.is_threshold:
            cur_ins = ins_num
            if res > self.sla:
                cur_ins += 1
            elif cpu < 50:
                cur_ins -= 1
            return cur_ins

        # 其他方法
        if self.argument_manager.if_replay:
            return self.argument_manager.get_action()
        else:
            # 1. 将数据送入到加载的模型中，得到动作数组
            s_tensor = torch.as_tensor((state,),dtype=torch.float32,device=self.device)

            a_tensor = self.act(s_tensor) # 耗时间1Ms
            a_tensor = a_tensor.argmax(dim=1)
            action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside

            # 2. 根据动作数组以及当前的实例数获取最终的结果
            cur_ins = ins_num
            if self.argument_manager.scaler_model == DQN_MODEL:
                cur_ins += action - self.action_max
            elif self.argument_manager.scaler_model == PPO_MODEL:
                if self.argument_manager.get_is_burst():
                    # TODO 验证
                    cur_ins = self.argument_manager.get_burst_ins()
                else:
                    if action < 2*self.action_max+1:
                        cur_ins += action - self.action_max
                    elif action < (2*self.action_max+1) + (2*self.pred_action_max+1): # 使用单步预测结果
                        delta_instance = action - (2*self.action_max+1) - self.pred_action_max
                        predict_workload = self.argument_manager.get_latest_onestep_prediction_result()
                        cur_ins = self.argument_manager.env._judge_instance_num(predict_workload) + delta_instance
                    else: # 使用多步预测的上阈值
                        delta_instance = action - (2*self.action_max+1) - (2*self.pred_action_max+1) - self.pred_action_max
                        predict_workload = self.argument_manager.get_latest_longterm_prediction_result()
                        cur_ins = self.argument_manager.env._judge_instance_num(predict_workload) + delta_instance
            else:
                raise NotImplementedError
            return cur_ins