# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 16:19:17 2023

@author: parkh
"""
import pandas as pd
class Job(object):

    # Default Constructor
    def __init__(self, job_id, job_type ,max_operation, setup_table, duedate, q_table, arrival_time, status):
        # 고정 정보
        self.id = job_id #job번호
        self.job_type = job_type #job type이 뭔지
        self.max_operation = max_operation # 이 job의 max operation이 언젠지
        self.duedate = duedate # 이 job의 duedate가 언제인지
        
        """
        공정 수 만큼 존재함
        [10, 14, 25, 34] J11 -> J12를 작업할 때 까지 10시간 안에 작업해야함
        """
        self.q_time_table = q_table 
        
        self.setup_table = setup_table #이 job의 setup 테이블
        self.job_arrival_time = arrival_time
        
        #변화하는 데이터
        self.current_operation_id = 1 #현재 공정이 어디인지
        #status 종류 -> "WAIT", "NOTYET", "DONE", "PROCESSING"
        self.status = status # 현재 job의 상태가 무엇인지
        self.remain_operation = self.max_operation #이 job의 남은 operation이 몇 개인지
        self.start_time = 0
        self.condition = True
        # For History and KPI
        
        self.history_list = []
        self.job_flowtime = 0 # job의 flow time
        self.tardiness_time = 0 #job의 tardiness time
        self.lateness_time = 0 # job의 lateness time
        self.operation_in_machine = [0 for x in range(max_operation)] #각각의 operation이 어떤 machine에 있었는지
        self.q_time_check_list = [ 0 for x in range(self.max_operation-1)]
    def jop(self):
        jop = ''
        if self.job_type < 10:
            jop = "j0"+str(self.job_type)
        else:
            jop = "j"+str(self.job_type)
        jop = jop+"0"+str(self.current_operation_id)
        return jop
    
    def assign_setting(self, machine, assign_time):
        machine_id = machine.id
        machine_number = int(machine_id[1:])
        self.operation_in_machine[self.current_operation_id - 1] = machine_number
        self.status = "PROCESSING"
        self.remain_operation -= 1
        q_time_diff = -1
        if self.current_operation_id != 1:
            q_time_diff =  max(0, (assign_time - self.start_time) - self.q_time_table[self.current_operation_id - 2])
            self.q_time_check_list[self.current_operation_id - 2] = q_time_diff
            if q_time_diff > 0:
                self.condition = False
        self.current_operation_id +=1
        return q_time_diff
    
    def complete_setting(self,start_time, end_time,event_type):
        self.status = "WAIT"
        last = False
        self.start_time = end_time
        if event_type == "track_in_finish" and self.remain_operation == 0:
            self.job_flowtime += end_time - self.job_arrival_time
            self.tardiness_time = max(0 , end_time-self.duedate)
            self.lateness_time = end_time-self.duedate
            last = True
        if last == True:
            self.status = "DONE"
        return last

    def arrival(self):
        self.status = "WAIT"
        
    def cal_flowtime(self, c_time):
        flow = c_time - self.job_arrival_time
        return flow
    
    def cal_tardiness(self, c_time):
        tardiness = max(0, c_time - self.duedate)
        return tardiness
    
    def cal_q_time(self, c_time):
        if self.start_time == 0:
            return 0
        else:
            q_time_diff = max(0, (c_time - self.start_time) - self.q_time_table[self.current_operation_id - 2])
            return q_time_diff
        
    def cal_q_time_total(self):
        total_q = sum(self.q_time_check_list)
        return total_q













