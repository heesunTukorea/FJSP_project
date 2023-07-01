import streamlit as st
'''
import pandas as pd
import numpy as np
from fjsp_Q import *
import os
from DQN8 import *
from FAB2 import *
from Resource3 import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
import collections
import random'''


col1, col2 = st.columns([7,1])

with col2:
    st.image("tuk_img.png")

st.title("시뮬레이터")
'''
learning_rate = 0.0005  
gamma = 0.99
buffer_limit = 100000
batch_size = 32


ReplayBuffer()
Qnet(nn.Module)
train(q, q_target, memory, optimizer)
main(
    FJSP_simulator('FJSP_Sim.csv','FJSP_Set.csv',"FJSP_Job.csv",1),
    Qnet(),
    Qnet(),
    q_target.load_state_dict(q.state_dict()),
    ReplayBuffer(),
    1,
    10,
    0.0,
    optim.Adam(q.parameters(), lr=learning_rate))
    

for i in range(1):
    Flow_time, machine_util, util, makespan, score =main()
    st.write("FlowTime:" , Flow_time)
    st.write("machine_util:" , machine_util)
    st.write("util:" , util)
    st.write("makespan:" , makespan)
    st.write("Score" , score)
    '''