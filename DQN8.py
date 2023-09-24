

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
import collections
import random
from simulator_DFJSP import *


params = {
    "p_data" : "/Users/shin/DFJSP-Qtime/Data/DFJSP_test.csv",
    "s_data" : "/Users/shin/DFJSP-Qtime/Data/DFJSP_setup_test.csv",
    "q_data" : "/Users/shin/DFJSP-Qtime/Data/DFJSP_Qdata_test.csv",
    "rd_data" : "/Users/shin/DFJSP-Qtime/Data/DFJSP_rdData_test2.csv"
}

r_param = {
    "learning_rate" : 0.0001,
    "gamma" : 0.99,
    "buffer_limit" : 50000,
    "batch_size" : 32
}




for i in range(1):
    Flow_time, machine_util, util, makespan, score =main(params, r_param)
    print("FlowTime:" , Flow_time)
    print("machine_util:" , machine_util)
    print("util:" , util)
    print("makespan:" , makespan)
    print("Score" , score)



#형찬 데이터 파일 경로
"""
    "p_data" : "/Users/shin/DFJSP-Qtime/DFJSP_test.csv",
    "s_data" : "/Users/shin/DFJSP-Qtime/DFJSP_setup_test.csv",
    "q_data" : "/Users/shin/DFJSP-Qtime/DFJSP_Qdata_test.csv",
    "rd_data" : "/Users/shin/DFJSP-Qtime/DFJSP_rdData_test2.csv"

#기본

    "p_data" : "프로세스 타임 데이터 입력",
    "s_data" : "셋업 데이터 입력 칸",
    "q_data" : "q_data",
    "rd_data" : "rd_data"

"""

"""    
params = torch.load("nomorspt.pt")
q = Qnet()
q.load_state_dict(params)
q.eval()
env = FJSP_simulator('C:/Users/parkh/git_tlsgudcks/simulator/data/FJSP_SIM7_all.csv','C:/Users/parkh/FJSP_SETUP_SIM.csv',"C:/Users/parkh/git_tlsgudcks/simulator/data/FJSP_Fab.csv",1) 
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
Flow_time, machine_util, util, makespan = env.performance_measure()
print("FlowTime:" , Flow_time)
print("machine_util:" , machine_util)
print("util:" , util)
print("makespan:" , makespan)
print("Score" , score)
""" 