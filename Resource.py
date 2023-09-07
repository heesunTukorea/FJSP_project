# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 16:15:22 2023

@author: parkh
"""

class Resource(object):
    # Default Constructor
    def __init__(self, resource_id):
        self.id = resource_id #기계 아이디
        self.status = 0 #기계 작동유무 , 0은 쉬는 중, 1은 작동중
        self.setup_status = 0 #기계 셋업 상태
        self.last_work_finish_time = 0 #최근에 끝난 작업시간
        self.job_id_processed = 0 #작업되고 있는 job ID
        self.utilization = 0
        self.idle_time = 0
        self.value_added_time = 0
        self.reservation_time = 0
        # For History
        self.history_list = []

    def assign_setting(self, job, reservation_time):
        self.status = 1
        if job.job_type != "j0":            
            self.setup_status = job.job_type
        self.job_id_processed = job.job_type
        self.reservation_time = reservation_time
        
    def complete_setting(self,start_time, end_time, event_type):
        self.status = 0
        self.job_id_processed = 0
        if self.last_work_finish_time != start_time:
            self.idle_time += start_time - self.last_work_finish_time #setup이거나 idel이거나
        if event_type == "track_in_finish":
            self.value_added_time += end_time-start_time
        elif event_type == "setup_change":
            self.idle_time += end_time-start_time
        elif event_type == "NOTHING":
            self.idle_time += end_time-start_time
        self.last_work_finish_time = end_time
            
    def util(self):
        return self.value_added_time, self.idle_time + self.value_added_time
    # Clone constructor 대신 이 함수를 사용

