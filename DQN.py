from Qnet import *
from ReplayBuffer import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
import collections
import random
from simulator_DFJSP import *
import streamlit as st
class DQN:
    def __init__(self,params, r_param, param):
        print("DQN on")
        self.params = params
        self.r_param =r_param
        self.param = param
    def train(self, q, q_target, memory, optimizer):
        for i in range(10):
            s, a, r, s_prime, done_mask = memory.sample(self.r_param["batch_size"])
            # q.number_of_time_list[a] += 1
            q_out = q(s)
            q_a = q_out.gather(1, a)
            max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
            # print(max_q_prime.shape)
            target = r + self.r_param["gamma"] * max_q_prime * done_mask
            loss = F.smooth_l1_loss(q_a, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def main_d(self):
        env = FJSP_simulator(self.params["p_data"], self.params["s_data"], self.params["q_data"], self.params["rd_data"],self.param)
        q = Qnet(self.r_param["input_layer"], self.r_param["output_layer"])
        q_target = Qnet(self.r_param["input_layer"], self.r_param["output_layer"])
        q_target.load_state_dict(q.state_dict())
        memory = ReplayBuffer(self.r_param["buffer_limit"])
        print_interval = 1
        q_load = 10
        score = 0.0
        optimizer = optim.Adam(q.parameters(), lr=self.r_param["learning_rate"])
        
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)

        total_iterations = self.r_param['episode']
        with st.expander("train"):
            for n_epi in range(total_iterations):
                # 여기는 sample_action 구간
                current_progress = int((n_epi + 1) / total_iterations * 100)
                my_bar.progress(current_progress, text=progress_text)
                epsilon = max(0.01, 0.08 - 0.02 * (n_epi / 200))
                s = env.reset()
                done = False
                score = 0.0
                while not done:
                    a = q.sample_action(torch.from_numpy(s).float(), epsilon)

                    s_prime, r, done = env.step(a)
                    done_mask = 0.0 if done else 1.0
                    if done == False:
                        memory.put((s, a, r, s_prime, done_mask))
                        s = s_prime
                        score += r
                    if done:
                        break

                # 학습구간
                if memory.size() > total_iterations:
                    self.train(q, q_target, memory, optimizer)
                self.script_performance(env,n_epi,epsilon,memory, score)
                # 결과 및 파라미터 저장
                if n_epi % print_interval == 0 and n_epi != 0:
                    params = q.state_dict()
                    param_name = str(n_epi) + "nomorspt.pt"
                    #torch.save(params, param_name)


                # 여기는 select_action 구간
                s = env.reset()
                done = False
                score = 0.0
                while not done:
                    a, a_list = q.select_action(torch.from_numpy(s).float(), epsilon)
                    s_prime, r, done = env.step(a)
                    s = s_prime
                    score += r
                    if done:
                        break
                self.script_performance(env,n_epi,epsilon,memory, score)

                if n_epi % q_load == 0 and n_epi != 0:
                    q_target.load_state_dict(q.state_dict())

        s = env.reset()
        done = False
        score = 0.0
        while not done:
            a, a_list = q.select_action(torch.from_numpy(s).float(), epsilon)
            # print(a_list)
            # print(a)
            s_prime, r, done = env.step(a)
            # print(r)
            s = s_prime
            score += r
            if done:
                break
        Flow_time, machine_util, util, makespan, Tardiness_time, Lateness_time, T_max, q_time_true, q_time_false, q_job_t, q_job_f, q_time = env.performance_measure()
        #fig,fig2,fig3,fig4,fig5,fig6,fig8 = env.gannt_chart()
        gantt = GanttChart(env.plotlydf, env.plotlydf_arrival_and_due,env.params)
        fig,fig2,fig3,fig4,fig5,fig6,fig8=gantt.play_gantt()
        return fig,fig2,fig3,fig4,fig5,fig6,fig8,Flow_time, machine_util, util, makespan, Tardiness_time, Lateness_time, T_max,q_time_true,q_time_false,q_job_t, q_job_f, q_time



    def script_performance(self, env, n_epi, epsilon,memory, score):
        Flow_time, machine_util, util, makespan, Tardiness_time, Lateness_time, T_max, q_time_true, q_time_false, q_job_t, q_job_f, q_over_time = env.performance_measure()
        #with st.expander("train"):
        
        container_style = """
            height: 500px;  # 원하는 높이로 조정
            overflow-y: auto;  # 수직 스크롤을 활성화
        """
        # 스크롤 가능한 컨테이너를 만듭니다.
        with st.container():
            st.write("--------------------------------------------------")
            st.write("flow time: {}, util : {:.3f}, makespan : {}".format(Flow_time, util, makespan))
            st.write("Tardiness: {}, Lateness : {}, T_max : {}".format(Tardiness_time, Lateness_time, T_max))
            st.write("q_true_op: {}, q_false_op : {}, q_true_job : {}, , q_false_job : {} , q_over_time : {}".format(
                q_time_true, q_time_false, q_job_t, q_job_f, q_over_time))
            st.write(
                "n_episode: {}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(n_epi, score,
                                                                                    memory.size(), epsilon * 100))
        st.markdown(f'<style>{container_style}</style>', unsafe_allow_html=True)