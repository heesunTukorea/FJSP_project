import pandas as pd
import random as rd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import os
from collections import Counter




save_folder = 'fjsp_csv_folder'



    #머신과 job,operation의 process시간
#랜덤범위5~25


def sim(sim_csv_name,count,machine_count,pmin,pmax,opmin,opmax):
#sim_csv_name
#     count = 총job의 갯수    
#     machine_count = 기계의 갯수
#     pmin = processing time의 최솟값
#     pmax = processing time의 최댓값
#     opmin = operation난수 범위의 최솟갑
#     opmax = operation난수 범위의 최댓값
    #기계의 갯수 설정
    machine = np.array(range(1, machine_count+1))
    
    #job의 number와 type 생성의 데이터 프레임
    F_job=[]
    for i in range(1,count+1):
        job_number = i
        F_job.append([job_number])
    columns = ['job_num']
    pd_F_job = pd.DataFrame(F_job, columns = columns)
    pd_F_job.index = np.arange(1, len(pd_F_job) + 1)
    job_df = pd_F_job
    
    job_num = job_df['job_num'].nunique()
    
    #오퍼레이션 랜덤 생성
    job_df['job_operation'] = np.random.randint(opmin, opmax+1, size=(job_num, 1))#난수범위
    job_df_op = job_df['job_operation']
    
    
    # 작업 번호와 작업의 공정 수를 이용하여 작업 이름을 생성
    jo = []
    for index, row in job_df.iterrows():
        job_num = row['job_num']
        job_operation = row['job_operation']
        for j in range(1, job_operation + 1):
            b = "j" + str(job_num).zfill(2) + str(j).zfill(2)
            jo.append(b)
            
    # 작업 이름과 기계 번호를 이용하여 시뮬레이션 데이터프레임 생성
    sim_df = pd.DataFrame(np.random.randint(pmin, pmax+1, size=(len(jo), machine_count)), index=jo, columns=machine)
    
    # 컬럼 이름에 'M' 접두사 추가
    pd_sim = sim_df.add_prefix('M')
    pd_sim.to_csv('FJSP_Sim.csv', index=True, header=True)


#job들의 setup타임 생성
#랜덤범위 1~10

def setup(set_csv_name,smin,smax):
    job_pro = pd.read_csv('FJSP_Sim.csv', index_col=0)
    #processing time에서 데이터 추출 하는 코드
    job_pro_index = job_pro.index
    job_list=[]
    for i in range(1, len(job_pro_index)+1):
        job_list.append((job_pro_index[i-1][1:3]))
    max_job_list = int(max(job_list))   
    job_num = np.arange(1, max_job_list + 1)
    
    set_df = pd.DataFrame(np.random.randint(smin, smax+1, size=(max_job_list+1, max_job_list+1)))#난수범위
    
    #컬럼과 컬럼 이름 변경
    columns = job_num
    pd_F_set = pd.DataFrame(set_df, index=columns, columns=columns)
    pd_F_set = pd_F_set.add_prefix('j')
    pd_F_set.index = pd_F_set.index.astype(str)
    pd_F_set.index = 'j'+ pd_F_set.index 
    
    #인덱스와 컬럼이 같을 때 값을 0으로 변경
    np.fill_diagonal(pd_F_set.values, 0)
    pd_F_set
    
    #0인 행의 리스트 생성
    new_row = np.zeros((1, len(pd_F_set.columns)))
    index = ['j0']
    # NumPy 배열을 데이터프레임으로 변환하여 마지막 행에 추가
    pd_F_set = pd_F_set.append(pd.DataFrame(new_row, columns=pd_F_set.columns,index=index))
    pd_F_set = pd_F_set.astype(int)
    pd_F_set.to_csv(f'{set_csv_name}.csv')


#Q_time 을 생성하는 코드


def Q_time(q_csv_name,qmin,qmax):
    #기존 csv파일을 불러와 데이터를 사용
    job_pro = pd.read_csv('FJSP_Sim.csv', index_col=0)
    job_pro_index = job_pro.index
    counts = []
    current_count = 1

    #프로세싱 타이임의 인덱스를 추출해서 작업과 공저의 갯수 추출
    for i in range(1, len(job_pro_index)):
        if job_pro_index[i][:3] == job_pro_index[i-1][:3]:
            current_count += 1
        else:
            counts.append(current_count)
            current_count = 1

    counts.append(current_count)
    job_df_op = counts
    
    #
    job_df_op_max = max(job_df_op)
    job_df_op_max_1 = list(range(1,job_df_op_max+1))
    job_df_op_values = job_df_op

    j_op_num = len(job_df_op)

    #Q_time을 기계들의 processing time의 최댓값에서 범위를 지정해서 곱함
    #공정이 없는것은 0으로 처리
    columns = job_df_op_max_1

    q_time = pd.DataFrame(index=range(j_op_num), columns=job_df_op_max_1)  # 빈 데이터프레임 생성

    for i, val in enumerate(job_df_op_values):
        for j in range(job_df_op_max):
            if j < val:
                q_time.iloc[i, j] = int(np.random.uniform(qmin,qmax) * job_pro.iloc[i].max())

            else:
                q_time.iloc[i, j] = 0

    q_time.index = np.arange(1, len(q_time) + 1)
    q_time.index = q_time.index.astype(str)
    q_time.index = 'j'+ q_time.index
    q_time.to_csv(f'{q_csv_name}.csv', index=True, header=True)




#기계와 공정의 오류를 생성하는 코드

def add_unavailable_machines_to_sim(error_csv_name, unavailable_machine_options=None):
    sim_df = pd.read_csv('FJSP_Sim.csv', index_col=0)

    job_pro = sim_df
    job_pro_index = job_pro.index
    counts = []
    current_count = 1

    #프로세싱 타임에서 작업과 공정의 수를 추출
    for i in range(1, len(job_pro_index)):
        if job_pro_index[i][:3] == job_pro_index[i-1][:3]:
            current_count += 1
        else:
            counts.append(current_count)
            current_count = 1

    counts.append(current_count)
    job_df_op = counts
    
    #기계와 공정을 지정해서 오류를 가정해 0을 입력
    if unavailable_machine_options:
        valid_options = []
        for machine, job, operation in unavailable_machine_options:
            if job in range(1, len(job_df_op) + 1):  # 유효한 작업 번호만 추가
                valid_options.append([machine, job, operation])

        for machine, job, operation in valid_options:
            if operation is None:
                for operation1 in range(1, job_df_op[job - 1] + 1):
                    job_key = 'j' + str(job).zfill(2) + str(operation1).zfill(2)
                    machine_name = 'M' + str(machine)
                    sim_df.loc[job_key, machine_name] = 0
            else:
                job_key = 'j' + str(job).zfill(2) + str(operation).zfill(2)
                machine_name = 'M' + str(machine)
                sim_df.loc[job_key, machine_name] = 0  # 해당 작업의 해당 공정에 해당하는 기계 값을 0으로 변경

    sim_df.to_csv(f'{error_csv_name}.csv', index=True, header=True)

# 데이터프레임을 출력하여 0인 값에 색을 입힙니다.

def highlight_zero(val):
    return 'background-color: yellow' if val == 0 else ''

def highlight_max(s):
    if s.name in ['makespan', 'Flow_time','tardiness','lateness','t_max','q_time_false','q_job_false','q_total_over_time']:
        is_max = s == s.min()  # 행에서 최솟값과 일치하는지 여부
    elif s.name in ['util','q_time_true','q_job_true']:
        is_max = s == s.max()  # 행에서 최댓값과 일치하는지 여부
    else:
        is_max = pd.Series(False, index=s.index)  # 다른 열은 스타일을 적용하지 않음
    return ['background-color: yellow' if v else '' for v in is_max]



def get_csv_file_list(save_folder):
    current_directory = os.getcwd()
    files = os.listdir(save_folder)
    csv_files = [f for f in files if f.endswith('.csv')]
    return csv_files


def get_csv_files_with_string(save_folder,target_string):
    files = []
    for file in os.listdir(save_folder):
        if file.endswith(".csv") and target_string in file:
            files.append(file)
    return files

#error값 범위 지정을 위한 코드

def sim_list_remind():
    sim_df = pd.read_csv('FJSP_Sim.csv', index_col=0)

    job_pro = sim_df
    job_pro_index = job_pro.index
    counts = []
    current_count = 1

    for r in range(1, len(job_pro_index)):
        if job_pro_index[r][:3] == job_pro_index[r-1][:3]:
            current_count += 1
        else:
            counts.append(current_count)
            current_count = 1

    counts.append(current_count)
    job_df_op = counts
    job_df_op_count = int(len(job_df_op))

    unavailable_machine_options = []
    machine_list = job_pro.columns
    machine_num_list = machine_list.str.replace("M","")
    machine_max = int(max(machine_num_list))
    return job_df_op,machine_max,job_df_op_count


# @st.cache_data
# def job(selected_sim_csv4,count,jmin,jmax):

#     job_pro = pd.read_csv(f'{save_folder}\{selected_sim_csv4}', index_col=0)
#     #processing time에서 데이터 추출 하는 
#     job_pro_index = job_pro.index
#     counts = []
#     current_count = 1
#     for i in range(1, len(job_pro_index)):
#         if job_pro_index[i][:3] == job_pro_index[i-1][:3]:
#             current_count += 1
#         else:
#             counts.append(current_count)
#             current_count = 1

#     counts.append(current_count)
#     job_df_op = counts

#     F_job=[]
#     for i in range(1,count+1):
#         job_number = i
#         job_type = rd.randrange(jmin,jmax+1)#난수 범위
#         F_job.append([job_number,job_type])
#     columns = ['','number']
#     pd_F_job = pd.DataFrame(F_job, columns = columns)
#     pd_F_job.index = np.arange(1, len(pd_F_job) + 1)



def filtered_result_create(job_product_list):
    
    random_array_df = pd.DataFrame({'op_count': job_product_list})

    filtered_result = []
    for num, op_count in enumerate(job_product_list, start=1):
        for _ in range(op_count):
            filtered_result.append(num)

    rd.shuffle(filtered_result)  # 리스트 내의 숫자를 섞음
    sorted_counter = dict(Counter(filtered_result))
    sorted_counter= dict(sorted(sorted_counter.items()))
    return filtered_result,sorted_counter

    

def release_due_data(rd_csv_name,filtered_result,first_release_supply,arrival_time_list,r_min,r_max):
    pd_sim = pd.read_csv('FJSP_Sim.csv', index_col=0)
    filtered_result=filtered_result
    #r_time만드는 코드
    # 각 작업 번호별 초기 작업 가능한 물량을 저장한 딕셔너리 
    supply_dict = {}
    for idx, supply in enumerate(first_release_supply):
        supply_dict[idx + 1] = supply
        
    #r_time만드는 코드
    # 각 작업 번호별 초기 작업 가능한 물량을 저장한 딕셔너리 
    r_time_dict = []
    same_job_dict = {}
    previous_r_time = None

    # 작업 목록(filtered_result)을 반복하며 작업별 'r_time' 값을 계산
    for idx in filtered_result:
        # 현재 작업 번호에 해당하는 초기 작업 가능한 물량을 확인
        if supply_dict[idx] > 0:
            r_time_dict.append(0)  # 초기 물량이 있는 경우 'r_time'을 0으로 설정
            supply_dict[idx] -= 1  # 사용한 초기 물량 차감
            previous_r_time = 0  # 이전 'r_time'을 0으로 설정
            same_job_dict[idx] = 0  # 초기값을 0으로 설정
        else:
            # 같은 작업 번호의 이전 'r_time' 값을 이미 계산한 경우
            if idx in same_job_dict:
                # 이전에 계산한 작업 번호와 관련된 정보를 가져옴
    #             same_job = same_job_dict[idx]
                same_job_r_time = same_job_dict[idx]

                # 새로운 'r_time'을 계산하되, 이전 'r_time'에 arrival_time_list[idx - 1]을 더한 값에
                # 0.8 ~ 1.2 사이의 난수를 곱해서 구함
                new_r_time = round(same_job_r_time + (arrival_time_list[idx - 1] * rd.uniform(r_min, r_max)))

                r_time_dict.append(new_r_time)  # 계산된 'r_time'을 리스트에 추가
                previous_r_time = new_r_time  # 이전 'r_time'을 새로 계산한 값으로 업데이트
                same_job_dict[idx] = previous_r_time
            else:
                # 이전에 계산한 작업 번호와 관련된 정보가 없는 경우
                #if previous_r_time is None:
                previous_r_time = 0  # 첫 작업이라면 이전 'r_time'을 해당 작업의 도착 시간으로 설정

                # 새로운 'r_time'을 계산하되, 이전 'r_time'에 arrival_time_list[idx - 1]을 더한 값에
                # 0.8 ~ 1.2 사이의 난수를 곱해서 구함
                new_r_time = round(previous_r_time + (arrival_time_list[idx - 1] * rd.uniform(r_min, r_max)))

                r_time_dict.append(new_r_time)  # 계산된 'r_time'을 리스트에 추가
                previous_r_time = new_r_time  # 이전 'r_time'을 새로 계산한 값으로 업데이트
                same_job_dict[idx] = previous_r_time  # 같은 작업 번호의 인덱스 정보 저장


    # 'r-time' 딕셔너리와 filtered_result를 인덱스로 하는 DataFrame 생성
    df = pd.DataFrame({'r_time': r_time_dict}, index=filtered_result)
    # 'r-time' 기준으로 정렬후 index로 재정렬
    df_sorted = df[df['r_time'] == 0].sort_index().append(df[df['r_time'] != 0].sort_values(by='r_time'), ignore_index=False)


    #d_time 만드는 코드
    # 빈 리스트 생성: 작업별 d_time 값을 저장할 리스트
    d_time_list = []

    # 각 작업별로 d_time 계산하고 리스트에 추가
    for index, row in df_sorted.iterrows():
        job_prefix = 'j'+str(index).zfill(2)  # 작업 번호의 접두어를 추출 (예: 2 -> j02)

        # 작업 번호의 접두어로 시작하는 행들 선택
        related_rows = pd_sim[pd_sim.index.str.startswith(job_prefix)]

        # 관련 작업의 processing_time 값 평균 계산
        avg_processing_time = related_rows.mean(axis=1).sum()

        # 랜덤한 스케일링을 위한 난수 생성 (1.5에서 1.7 사이의 난수)
        random_multiplier = rd.uniform(1.5, 1.7)

        # 평균 processing_time에 스케일링 적용한 후 반올림하여 정수로 변환
        scaled_avg_processing_time = round(avg_processing_time * random_multiplier)

        d_time_list.append(scaled_avg_processing_time)  # 작업별 d_time 리스트에 추가

    # 작업별로 계산된 d_time을 df_sorted 데이터프레임에 추가
    df_sorted['d_time'] = d_time_list

    # d_time을 r-time에 더해줌으로써 새로운 r-time 계산

    df_sorted['d_time'] = df_sorted['r_time'] + d_time_list
    df_sorted.index = df_sorted.index.astype(str)
    df_sorted.index = 'j'+ df_sorted.index 
    
    df_sorted.to_csv(f'{rd_csv_name}.csv', index=True, header=True)