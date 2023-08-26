

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
import collections
import random
from simulator_DFJSP import *

learning_rate = 0.0001  
gamma = 0.99
buffer_limit = 50000
batch_size = 32

class ReplayBuffer():        #buffer class
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit);
    def put(self, transition):
        self.buffer.append(transition)
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [],[],[],[],[]
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
            
        return torch.tensor(s_lst, dtype=torch. float),torch.tensor(a_lst), torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch. float), torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):        #Qnet
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(12,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 9)
        else:
            return out.argmax().item()
        
    def select_action(self, obs, epsilon):
        out = self.forward(obs)
        return out.argmax().item(),out
        
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)
        #q.number_of_time_list[a] += 1    
        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max (1)[0].unsqueeze(1)
        #print(max_q_prime.shape)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
def main():
    env = FJSP_simulator('C:/Users/parkh/git_tlsgudcks/simulator/data/DFJSP_test.csv','C:/Users/parkh/git_tlsgudcks/simulator/data/DFJSP_setup_test.csv',
                          "C:/Users/parkh/git_tlsgudcks/simulator/data/DFJSP_Qdata_test.csv","C:/Users/parkh/git_tlsgudcks/simulator/data/DFJSP_rdData_test2.csv",i)
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    print_interval = 1
    q_load = 10
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    
    for n_epi in range(1000):
        #여기는 sample_action 구간
        epsilon = max(0.01 , 0.08 - 0.02*(n_epi/200))
        s = env.reset()
        done = False
        score = 0.0
        while not done:
            a = q.sample_action(torch.from_numpy(s). float(), epsilon)
            s_prime, r, done = env.step(a)
            done_mask =0.0 if done else 1.0
            if done == False:
                memory.put((s,a,r,s_prime,done_mask))
                s = s_prime
                score += r
            if done:
                break
            
        #학습구간    
        if memory.size() > 1000:
            train(q, q_target, memory, optimizer)
        
        #결과 및 파라미터 저장    
        if n_epi % print_interval==0 and n_epi!=0:
            #q_target.load_state_dict(q.state_dict())
            params = q.state_dict()
            param_name = str(n_epi)+"nomorspt.pt"
            #print(param_name)
            torch.save(params, param_name)
            Flow_time, machine_util, util, makespan, Tardiness_time, Lateness_time, T_max,q_time_true,q_time_false,q_job_t, q_job_f, q_over_time = env.performance_measure()
            print("--------------------------------------------------")
            print("flow time: {}, util : {:.3f}, makespan : {}".format(Flow_time, util, makespan))
            print("Tardiness: {}, Lateness : {}, T_max : {}".format(Tardiness_time, Lateness_time, T_max))
            print("q_true_op: {}, q_false_op : {}, q_true_job : {}, , q_false_job : {} , q_over_time : {}".format(q_time_true, q_time_false, q_job_t, q_job_f, q_over_time))
            print("n_episode: {}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(n_epi, score/print_interval,memory.size(),epsilon*100))
            #score=0.0
        
        #여기는 select_action 구간
        s = env.reset()
        done = False
        score = 0.0
        params = q.state_dict()
        torch.save(params, "nomorspt.pt" )
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
        print("--------------------------------------------------")
        print("flow time: {}, util : {:.3f}, makespan : {}".format(Flow_time, util, makespan))
        print("Tardiness: {}, Lateness : {}, T_max : {}".format(Tardiness_time, Lateness_time, T_max))
        print("q_true_op: {}, q_false_op : {}, q_true_job : {}, , q_false_job : {} , q_over_time : {}".format(q_time_true, q_time_false, q_job_t, q_job_f, q_over_time))
        print("n_episode: {}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(n_epi, score/print_interval,memory.size(),epsilon*100))
        
        
        
        if n_epi % q_load ==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
    
    
    s = env.reset()
    done = False
    score = 0.0
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
    Flow_time, machine_util, util, makespan, Tardiness_time, Lateness_time, T_max,q_time_true,q_time_false,q_job_t, q_job_f = env.performance_measure()
    env.gannt_chart()
    return Flow_time, machine_util, util, makespan, score
for i in range(1):
    Flow_time, machine_util, util, makespan, score =main()
    print("FlowTime:" , Flow_time)
    print("machine_util:" , machine_util)
    print("util:" , util)
    print("makespan:" , makespan)
    print("Score" , score)
    
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