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
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(12,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,11)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
    def select_action(self, obs, epsilon):
        out = self.forward(obs)
        return out.argmax().item(),out
        
def dqn_params(param_name, sim_file_name, setup_file_name, q_time_file_name, rddata_file_name):
    params = torch.load(param_name)
    q = Qnet()
    q.load_state_dict(params)
    q.eval()
    #env = FJSP_simulator('C:/Users/user/main_pro/duedate_DQN/data/FJSP_SIM7.csv','C:/Users/user/main_pro/duedate_DQN/data/FJSP_SETUP_SIM.csv',"C:/Users/user/main_pro/duedate_DQN/data/FJSP_Fab.csv",1)
    env = FJSP_simulator(sim_file_name, setup_file_name, q_time_file_name, rddata_file_name,1)
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
    Flow_time, machine_util, util, makespan, Tardiness_time, Lateness_time, T_max,q_time_true,q_time_false,q_job_t, q_job_f, q_over_time = env.performance_measure()
    fig,fig2,fig3,fig4,fig5,fig6,fig7,fig8=env.gannt_chart()
    print("FlowTime:" , Flow_time)
    print("machine_util:" , machine_util)
    print("util:" , util)
    print("makespan:" , makespan)
    print("Score" , score)    #683 sim7 fab
    return fig,fig2,fig3,fig4,fig5,fig6,fig7,fig8, Flow_time, machine_util, util, makespan, Tardiness_time, Lateness_time, T_max,q_time_true,q_time_false,q_job_t, q_job_f, q_over_time,score


# data_dict = {
#     "p_time_data" :  "C:/Users/user/main_pro/duedate_DQN/data/FJSP_Sim_10_zero.csv",
#     "setup_data" : 'C:/Users/user/main_pro/duedate_DQN/data/FJSP_Set_10.csv',
#     "Q_data" : "C:/Users/user/main_pro/duedate_DQN/data/FJSP_Q_time_10_0.4.csv",
#     "rd_data" : "C:/Users/user/main_pro/duedate_DQN/data/FJSP_rd_time_10_10,60.csv"
#     }
# if "__main__":
#     dqn_params("nomorspt.pt", data_dict)
    