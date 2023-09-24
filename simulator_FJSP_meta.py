# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 13:33:05 2022

@author: parkh
"""
    
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import copy
import random
from matplotlib import pyplot as plt

import dispatcher
from Resource import *
from Job import *
from Event import *
from dispatcher import * 
from Parameter import *
from collections import defaultdict
from StateManager import *
from plotly.offline import plot
from GanttChart import *

class FJSP_simulator(object):
        
    #processing time
    #setup time
    #queue time
    #realase time
    #duedate time(realase time에 포함)
    # 5가지 데이터를 모두 불러옴    
    def __init__(self, p_time_data, s_time_data):
        # 고정 부분
        self.process_time_table = pd.read_csv(p_time_data,index_col=(0))
        self.setup_time_table = pd.read_csv(s_time_data, index_col=(0))
        self.machine_number = len(self.process_time_table.columns) #machine 개수
        """총 job 개수"""
        operation = self.process_time_table.index
        op_table=[]
        for i in range(len(operation)):
            op_table.append(operation[i][:3])
        self.job_number = set(op_table) #총 Job_type 개수
        """각 job별로 총 operation개수"""
        self.max_operation = [0 for x in range(self.job_number)]
        for i in range(1, self.job_number+1):
            for j in op_table:
                if i == int(j):
                    self.max_operation[i-1] +=1
        
        # 리셋 부분
        self.done = False #종료조건 판단
        self.time = 0 #시간
        self.plotlydf = pd.DataFrame([],columns=['Type','JOB_ID','Task','Start','Finish','Resource','Rule','Step','Q_diff','Q_check'])
        self.plotlydf_arrival_and_due = pd.DataFrame([],columns=['Type','JOB_ID','Task','Start','Finish','Resource','Rule','Step','Q_diff','Q_check'])
        self.step_number = 0
        self.j = 0
        self.j2 = 0
        #Job 객체 , jop, machine객체, 현재시간, 끝나는 시간, 이벤트이름"track_in_finish", rule_name,step_num, setup_time
        
        """job 인스턴스 생성"""
        self.j_list = defaultdict(Job)
        self.event_list = []
        for i in range(len(job_number):
            due_date = 0
            realase_date = 0
            if realase_date == 0:
                status = "WAIT"
            else:
                status = "NOTYET"
            
            job_type = "j" + str(i+1)
            
            job_type_int = int(job_type[1:])
            job_id = job_type + "-" + str(realase_date) + "-" + str(i)    
            j = Job(job_id, job_type_int ,self.max_operation[job_type_int-1], self.setup_time_table[job_type]
                    ,due_date,self.queue_time_table.loc[job_type].tolist(), realase_date, "NOTYET") 
            self.j_list[j.id] = j
            
            e = Event(j,"job_arrival" , "NONE", self.time, realase_date,"job_arrival","NONE","NONE","NONE",0)
            start = datetime.fromtimestamp((realase_date-1)*3600)
            realase = datetime.fromtimestamp(realase_date*3600)
            due = datetime.fromtimestamp(due_date*3600)
            due_end = datetime.fromtimestamp((due_date+1)*3600)
            self.event_list.append(e)
            self.plotlydf_arrival_and_due.loc[self.j2] = dict(Type = "job_arrival", JOB_ID = j.id  ,Task="job_arrival", Start=start, Finish=realase, Resource="NONE", Rule = "NONE", 
                                             Step = "NONE", Q_diff = "job_arrival", Q_check = "job_arrival")
            self.j2 += 1
            self.plotlydf_arrival_and_due.loc[self.j2] = dict(Type = "due_date", JOB_ID = j.id  ,Task="due_date", Start=due, Finish=due_end, Resource="NONE", Rule = "NONE", 
                                             Step = "NONE", Q_diff = "due", Q_check = "due")
            self.j2 += 1
        """machine 인스턴스 생성"""
        self.r_list = defaultdict(Resource)
        for i in range(self.machine_number):
            r = Resource("M"+str(i+1))
            self.r_list[r.id] = r
            
        
        
        


    def reset(self):
        # 리셋 부분
        self.state_manager = StateManager()
        self.done = False #종료조건 판단
        self.remain_job = copy.deepcopy(self.total_job)
        #self.num_of_op = sum(self.total_operation)
        self.time = 0 #시간
        self.plotlydf = pd.DataFrame([],columns=['Type','JOB_ID','Task','Start','Finish','Resource','Rule','Step','Q_diff','Q_check'])
        self.plotlydf_arrival_and_due = pd.DataFrame([],columns=['Type','JOB_ID','Task','Start','Finish','Resource','Rule','Step','Q_diff','Q_check'])
        self.step_number = 0
        self.j = 0
        self.j2 = 0
        #Job 객체 , jop, machine객체, 현재시간, 끝나는 시간, 이벤트이름"track_in_finish", rule_name,step_num, setup_time
        
        """job 인스턴스 생성"""
        self.j_list = defaultdict(Job)
        self.event_list = []
        for i in range(len(self.rtime_and_dtime_table)):
            due_date = self.rtime_and_dtime_table.iloc[i]["d_time"]
            realase_date = self.rtime_and_dtime_table.iloc[i]["r_time"]
            if realase_date == 0:
                status = "WAIT"
            else:
                status = "NOTYET"
            
            job_type = self.rtime_and_dtime_table.iloc[i].name
            
            job_type_int = int(job_type[1:])
            job_id = job_type + "-" + str(realase_date) + "-" + str(i)    
            j = Job(job_id, job_type_int ,self.max_operation[job_type_int-1], self.setup_time_table[job_type]
                    ,due_date,self.queue_time_table.loc[job_type].tolist(), realase_date, "NOTYET") 
            self.j_list[j.id] = j
            
            e = Event(j,"job_arrival" , "NONE", self.time, realase_date,"job_arrival","NONE","NONE","NONE",0)
            start = datetime.fromtimestamp((realase_date-1)*3600) #realase_date를 간트에 표현할 때 한칸으로 하려고
            realase = datetime.fromtimestamp(realase_date*3600)
            due = datetime.fromtimestamp(due_date*3600)
            due_end = datetime.fromtimestamp((due_date+1)*3600)
            self.event_list.append(e)
            self.plotlydf_arrival_and_due.loc[self.j2] = dict(Type = "job_arrival", JOB_ID = j.id  ,Task="job_arrival", Start=start, Finish=realase, Resource="NONE", Rule = "NONE", 
                                             Step = "NONE", Q_diff = "job_arrival", Q_check = "job_arrival")
            self.j2 += 1
            self.plotlydf_arrival_and_due.loc[self.j2] = dict(Type = "due_date", JOB_ID = j.id  ,Task="due_date", Start=due, Finish=due_end, Resource="NONE", Rule = "NONE", 
                                             Step = "NONE", Q_diff = "due", Q_check = "due")
            self.j2 += 1
        """machine 인스턴스 생성"""
        self.r_list = defaultdict(Resource)
        for i in range(self.machine_number):
            r = Resource("M"+str(i+1))
            self.r_list[r.id] = r
        
        s = [0] * 12
        
        df = pd.Series(s)
        s = df.to_numpy()
        
        return s
        
            
        return s
    
    def performance_measure(self):
        q_time_true = 0
        q_time_false = 0
        makespan = self.time
                    
        Flow_time = 0
        Tardiness_time = 0 #new
        Lateness_time = 0 #new
        T_max = 0 #new
        L_max = 0 #new
        value_time_table = []
        full_time_table = []
        machine_util = 0
        util = 0
        q_job_f = 0
        q_job_t = 0
        z = []
        total_q_time_over = 0
        makespan = self.time
        for machine in self.r_list:
            value_added_time, full_time = self.r_list[machine].util()
            value_time_table.append(value_added_time)
            full_time_table.append(full_time)
        util = sum(value_time_table)/sum(full_time_table)
        for job in self.j_list:
            Flow_time += self.j_list[job].job_flowtime
            if self.j_list[job].tardiness_time > T_max:
                T_max = self.j_list[job].tardiness_time
            Tardiness_time += self.j_list[job].tardiness_time
            Lateness_time += self.j_list[job].lateness_time
            k = []
            for q in self.j_list[job].q_time_check_list:
                k.append(q)
                if q > 0:
                    q_time_false += 1
                else:
                    q_time_true += 1
            z.append(k)
            if self.j_list[job].condition == True:
                q_job_t += 1
            else:
                q_job_f += 1
            total_q_time_over += self.j_list[job].cal_q_time_total()
        #fig = px.timeline(self.plotlydf, x_start="Start", x_end="Finish", y="Resource", color="Task", width=1000, height=400)
        #fig.show()
        return Flow_time, machine_util, util, makespan, Tardiness_time, Lateness_time, T_max,q_time_true,q_time_false,q_job_t, q_job_f, total_q_time_over
    
    def step(self, action):
       # print(self.num_of_op)
        #print(self.total_job)
        ##print(self.total_operation)
        r = 0
        #print(action)
        done = False
        while True:
            machine = self.check_availability()
            if machine == "NONE":
                self.process_event()
                #이벤트도 비워져 있고, #job들도 다 done이면 종료
                if len(self.event_list) == 0 and all(self.j_list[job].status == "DONE" for job in self.j_list): 
                    done = True
                    s_prime = [1]*12
                    df = pd.Series(s_prime)
                    s_prime = df.to_numpy()
                    r =  0
                    break
            else:
                candidate_list = self.get_candidate(machine)
                candidate_list, rule_name = self.dispatcher.dispatching_rule_decision(candidate_list ,action,self.time)
                q_time = self.get_event(candidate_list[0], machine, rule_name)

                s_prime = self.state_manager.set_state(self.j_list, self.r_list, self.time)
                reservation_time = self.r_list[machine].reservation_time
                last_work_finish_time = self.r_list[machine].last_work_finish_time
                max_reservation = 0
                min_reservation = 100000000
                total_idle = 0

                for machine in self.r_list:
                    if self.r_list[machine].reservation_time > max_reservation:
                        max_reservation = self.r_list[machine].reservation_time
                    if self.r_list[machine].reservation_time < min_reservation:
                        min_reservation = self.r_list[machine].reservation_time
                    if self.r_list[machine].reservation_time < last_work_finish_time:
                        total_idle += (last_work_finish_time - self.r_list[machine].reservation_time)
                        self.r_list[machine].reservation_time = last_work_finish_time

                if q_time == "None" :
                    r += 0
                else:
                    r+= q_time
                r -= (reservation_time-last_work_finish_time + total_idle)
                break
        return s_prime, r , done
    def run(self, rule):
        while True:
            machine = self.check_availability()
            if machine != "NONE":
                candidate_list = self.get_candidate(machine)
                candidate_list, rule_name = self.dispatcher.dispatching_rule_decision(candidate_list, rule, self.time)
                q_time = self.get_event(candidate_list[0], machine, rule_name)
            else:
                if len(self.event_list) == 0:
                    break
                self.process_event()
                
        
        Flow_time, machine_util, util, makespan, tardiness, lateness, t_max,q_time_true,q_time_false,q_job_t, q_job_f, q_time = self.performance_measure()
        gantt = GanttChart(self.plotlydf, self.plotlydf_arrival_and_due)
        gantt.play_gantt()


        print("FlowTime:" , Flow_time)
        print("machine_util:" , machine_util)
        print("util:" , util)
        print("makespan:" , makespan)
        print("Tardiness:" , tardiness)
        print("Lateness:" , lateness)
        print("T_max:" , t_max)
        print("Q time True", q_time_true)
        print("Q time False", q_time_false)
        print("Q job True", q_job_t)
        print("Q job False", q_job_f)
        print("Q total over time", q_time)
        return FFlow_time, machine_util, util, makespan, tardiness, lateness, t_max,q_time_true,q_time_false,q_job_t, q_job_f, q_time 
    #event = (job_type, operation, machine_type, start_time, end_time, event_type)
    def process_event(self):
        #print(self.event_list)
        self.event_list.sort(key = lambda x:x.end_time, reverse = False)
        event = self.event_list.pop(0)
        self.time = event.end_time
        if event.event_type == "job_arrival":
            event.job.arrival()
        else:
            if event.event_type != "track_in_finish":
                if event.event_type == "setup_change":
                    event_type = "setup"
                elif event.event_type == "NOTHING":
                    event_type = "NOTHING"
            else:
                #print(event.job)
                event_type = "j"+str(event.job.job_type)
                last = event.job.complete_setting(event.start_time, event.end_time ,event.event_type) # 작업이 대기로 변함, 시작시간, 종료시간, event_type
                event.machine.complete_setting(event.start_time, event.end_time ,event.event_type) # 기계도 사용가능하도록 변함
            rule = event.rule_name
            step = event.step_num
            start = datetime.fromtimestamp(event.start_time*3600)
            end = datetime.fromtimestamp(event.end_time*3600)
            q_time_diff = event.q_time_diff
            q_time_check = event.q_time_check
            #print(self.step_number) Q_Check , Q_time_over
            self.plotlydf.loc[self.j] = dict(Type = event_type, JOB_ID = event.job.id  ,Task=event.jop, Start=start, Finish=end, Resource=event.machine.id, Rule = rule, 
                                             Step = step, Q_diff = q_time_diff, Q_check = q_time_check) #간트차트를 위한 딕셔너리 생성, 데이터프레임에 집어넣음
            self.j+=1
    
    def assign_setting(self, job, machine,reservation_time): #job = 1 machine = 1
        q_time_diff = job.assign_setting(machine, self.time)
        if job.remain_operation == 0:
            self.total_job[job.job_type-1] -= 1
        #self.total_operation[job.job_type-1] -=1
        machine.assign_setting(job, reservation_time)
        return q_time_diff
        
    def check_availability(self):
        index_k = 0
        select_machine = "NONE"
        for machine in self.r_list:
            index_k += 1
            if self.r_list[machine].status == 0:
                machine = self.r_list[machine].id #machine 이름
                p_table=[]
                for job in self.j_list: #job 이름과 operation이름 찾기
                    jop = self.j_list[job].jop()
                    if jop not in self.process_time_table.index: #해당 jop가 없는 경우  
                        pass
                    elif self.process_time_table[machine].loc[jop] == 0 : #해당 jop가 작업이 불가능할 경우
                        pass
                    elif self.j_list[job].status != "WAIT": #해당 jop가 작업중일 경우
                        pass
                    else:
                        p_table.append([self.j_list[job], self.process_time_table[machine].loc[jop]])
                if len(p_table) == 0:#현재 이벤트를 발생시킬 수 없음
                    pass
                else:
                    select_machine = machine
                    break
        return select_machine

    def get_candidate(self, machine):
        machine_id = self.r_list[machine].id
        candidate_list = []
        for job in self.j_list:
            if self.j_list[job].status == "WAIT":
                jop = self.j_list[job].jop()
                setup_time = self.j_list[job].setup_table['j'+str(self.r_list[machine].setup_status)]
                if self.process_time_table[machine_id].loc[jop] != 0:
                    candidate_list.append([self.j_list[job], self.process_time_table[machine].loc[jop],setup_time,jop])

        return candidate_list

    def get_event(self, candidate, machine, rule_name):
        step_num = self.step_number
        job, process_time, setup_time, jop = candidate
        if setup_time != 0:
            e = Event(job, "setup", self.r_list[machine], self.time, self.time + setup_time,
                      "setup_change",
                      "NONE", step_num, setup_time, 0)
            self.event_list.append(e)
        q_time_diff = self.assign_setting(job, self.r_list[machine],
                                          self.time + setup_time + process_time)
        e = Event(job, jop, self.r_list[machine], self.time, self.time + setup_time + process_time,
                  "track_in_finish", rule_name, step_num, setup_time, q_time_diff)
        self.event_list.append(e)
        self.step_number +=1
        return q_time_diff
"""
for i in range(2,3):
    main = FJSP_simulator('C:/Users/parkh/git_tlsgudcks/simulator/data/DFJSP_test.csv','C:/Users/parkh/git_tlsgudcks/simulator/data/DFJSP_setup_test.csv',
                          "C:/Users/parkh/git_tlsgudcks/simulator/data/DFJSP_Qdata_test.csv","C:/Users/parkh/git_tlsgudcks/simulator/data/DFJSP_rdData_test2.csv",i)
    FT, util2, ms = main.run()
    makespan_table.append(ms)
    util.append(util2)
    ft_table.append(FT)
print(makespan_table)
print(ft_table)
print(util)
print("zz")
"""