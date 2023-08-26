import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
import collections
import random
from FAB2 import *
from DQN8 import *


col1, col2 = st.columns([7,1])

with col2:
    st.image("tuk_img.png")
st.title("시뮬레이터")

rule_select_list=[]    
col5,col6,col7,col8,col9 = st.columns([1,1,1,1,1])
with col5:
    spt = st.checkbox('SPT')
    if spt:
        rule_select_list.append(1)
    lpt = st.checkbox('LPT')
    if lpt:
        rule_select_list.append(6)
with col6:
    sptssu = st.checkbox('SPTSSU')
    if sptssu:
        rule_select_list.append(2)
    ljrlor= st.checkbox('LJRLOR')
    if ljrlor:
        rule_select_list.append(7)
with col7:
    mor = st.checkbox('MOR')
    if mor:
        rule_select_list.append(3)
    mjrmor = st.checkbox('MJRMOR')
    if mjrmor:
        rule_select_list.append(8)
with col8:
    morspt = st.checkbox('MORSPT')
    if morspt:
        rule_select_list.append(4)
    mwrmor = st.checkbox('MWRMOR')
    if mwrmor:
        rule_select_list.append(9)
with col9:
    lor = st.checkbox('LOR')
    if lor:
        rule_select_list.append(5)

   



if st.button('클릭'):
    for i in rule_select_list:
        Flow_time, machine_util, util, makespan, score =main()
        st.write("FlowTime:" , Flow_time)
        st.write("machine_util:" , machine_util)
        st.write("util:" , util)
        st.write("makespan:" , makespan)
        st.write("Score" , score)
# '''
# learning_rate = 0.0005  
# gamma = 0.99
# buffer_limit = 100000
# batch_size = 32


# ReplayBuffer()
# Qnet(nn.Module)
# train(q, q_target, memory, optimizer)
# main(
#     FJSP_simulator('FJSP_Sim.csv','FJSP_Set.csv',"FJSP_Job.csv",1),
#     Qnet(),
#     Qnet(),
#     q_target.load_state_dict(q.state_dict()),
#     ReplayBuffer(),
#     1,
#     10,
#     0.0,
#     optim.Adam(q.parameters(), lr=learning_rate))
    

# for i in range(1):
#     Flow_time, machine_util, util, makespan, score =main()
#     st.write("FlowTime:" , Flow_time)
#     st.write("machine_util:" , machine_util)
#     st.write("util:" , util)
#     st.write("makespan:" , makespan)
#     st.write("Score" , score)
#     '''