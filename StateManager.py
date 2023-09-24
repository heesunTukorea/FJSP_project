
from Job import *
from Resource import *
class StateManager:
    def __init__(self):
        #print("stateManager")
        self.time = 0
    def set_state(self, j_list, r_list, cuur_time):
        self.time = cuur_time
        """
                재공 정보 :
                    대기 중인 job들의 개수
                    작업 중인 job들의 개수
                    대기 중인 job들의 남은 operation 개수 평균
                    대기 중인 job들의 tardiness 평균
                    대기 중인 job들의 q-time 초과 평균
                    대기 중인 job들의 flow time 평균

                기계 정보 :
                    기계의 현재 시간
                    현재 시간 / 다른 기계의 최대 시간
                    다른 기계들과 차이의 평균

                누적 정보 :
                    현재까지 total tardiness
                    현재까지 total q over time
                    현재까지 처리한 job 개수
                """
        s = []
        number_of_jobs_wait = 0  # clear
        number_of_jobs_load = 0  # clear
        total_remain_operation = 0
        total_tardiness = 0
        total_q_time_over = 0
        total_flow_time = 0
        number_of_job_done = 0  # clear

        total_job_tardiness_done = 0  # clear
        total_job_q_time_over_done = 0  # clear
        for job in j_list:  # job 이름과 operation이름 찾기
            if j_list[job].status == "WAIT":
                number_of_jobs_wait += 1
                total_remain_operation += j_list[job].remain_operation
                total_tardiness += j_list[job].cal_tardiness(self.time)
                total_q_time_over += j_list[job].cal_q_time(self.time)
                total_flow_time += j_list[job].cal_flowtime(self.time)
            elif j_list[job].status == "PROCESSING":
                number_of_jobs_load += 1
            elif j_list[job].status == "DONE":
                number_of_job_done += 1
                total_job_tardiness_done += j_list[job].tardiness_time
                q_total = j_list[job].cal_q_time_total()
                total_job_q_time_over_done += q_total

        current_time = self.time
        total_reservation_time_diff = 0
        max_reservation_time = 0
        for machine in r_list:
            total_reservation_time_diff += r_list[machine].reservation_time - current_time
            if max_reservation_time > r_list[machine].reservation_time:
                max_reservation_time = r_list[machine].reservation_time

        s.append(number_of_jobs_wait)
        s.append(number_of_jobs_load)
        if number_of_jobs_wait == 0:
            for _ in range(4):
                s.append(0)
        else:
            s.append(total_remain_operation / number_of_jobs_wait)
            s.append(total_tardiness / number_of_jobs_wait)
            s.append(total_q_time_over / number_of_jobs_wait)
            s.append(total_flow_time / number_of_jobs_wait)

        s.append(current_time)
        if max_reservation_time == 0:
            s.append(0)
        else:
            s.append(current_time / max_reservation_time)
        s.append(total_reservation_time_diff / len(r_list))

        s.append(number_of_job_done)
        if number_of_job_done == 0:
            s.append(0)
            s.append(0)
        else:
            s.append(total_job_tardiness_done / number_of_job_done)
            s.append(total_job_q_time_over_done / number_of_job_done)

        df = pd.Series(s)
        s = df.to_numpy()

        return s
