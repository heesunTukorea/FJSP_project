# 11개 디스패칭룰 64 32
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
import collections
import random
from simulator_DFJSP import *



class Qnet(nn.Module):
    def __init__(self, input_layer, output_layer):
        super(Qnet, self).__init__()
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.fc1 = nn.Linear(self.input_layer, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.output_layer)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
    def select_action(self, obs, epsilon):
        out = self.forward(obs)
        return out.argmax().item(),out
        
def dqn_params(param_name,param,r_param):
    params1 = torch.load(param_name)
    q = Qnet(r_param["input_layer"], r_param["output_layer"])
    q_target = Qnet(r_param["input_layer"], r_param["output_layer"])
    q_target.load_state_dict(q.state_dict())
    q.eval()
    #env = FJSP_simulator('C:/Users/user/main_pro/duedate_DQN/data/FJSP_SIM7.csv','C:/Users/user/main_pro/duedate_DQN/data/FJSP_SETUP_SIM.csv',"C:/Users/user/main_pro/duedate_DQN/data/FJSP_Fab.csv",1)
    env = FJSP_simulator(param.data["p_data"], param.data["s_data"],param.data["q_data"], param.data["rd_data"],param)
    s = env.reset()
    done = False
    score = 0.0
    epsilon = max(0.01 , 0.08 - 0.02*(20/200))
    while not done:
        a, a_list = q.select_action(torch.from_numpy(s). float(), epsilon)
        #print(a_list)
        #print(a)
        s_prime, r, done = env.step(a)
        #print(r)
        s = s_prime
        score += r
        if done:
            break
    Flow_time, machine_util, util, makespan, Tardiness_time, Lateness_time, T_max, q_time_true, q_time_false, q_job_t, q_job_f, q_time = env.performance_measure()
    gantt = GanttChart(env.plotlydf, env.plotlydf_arrival_and_due,env.params)
   
    fig,fig2,fig3,fig4,fig5,fig6,fig8=gantt.play_gantt()
    print("FlowTime:" , Flow_time)
    print("machine_util:" , machine_util)
    print("util:" , util)
    print("makespan:" , makespan)
    print("Score" , score)    #683 sim7 fab
    return fig,fig2,fig3,fig4,fig5,fig6,fig8,Flow_time, machine_util, util, makespan, Tardiness_time, Lateness_time, T_max,q_time_true,q_time_false,q_job_t, q_job_f, q_time


# data_dict = {
#     "p_time_data" :  "C:/Users/user/main_pro/duedate_DQN/data/FJSP_Sim_10_zero.csv",
#     "setup_data" : 'C:/Users/user/main_pro/duedate_DQN/data/FJSP_Set_10.csv',
#     "Q_data" : "C:/Users/user/main_pro/duedate_DQN/data/FJSP_Q_time_10_0.4.csv",
#     "rd_data" : "C:/Users/user/main_pro/duedate_DQN/data/FJSP_rd_time_10_10,60.csv"
#     }
# if "__main__":
#     dqn_params("nomorspt.pt", data_dict)
    