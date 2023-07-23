import streamlit as st



col1, col2 = st.columns([7,1])
st.title("테이터셋 코드")
with col2:
    st.image("tuk_img.png")
code = '''
import pandas as pd
import random as rd
import numpy as np
import streamlit as st

#job의 이름과 operation생성 파일
#count에 job의 갯수 할당
#랜덤범위 1~7
@st.cache_data
def job(count,jmin,jmax):
    F_job=[]
    for i in range(1,count+1):
        job_number = i
        job_type = rd.randrange(jmin,jmax+1)#난수 범위
        F_job.append([job_number,job_type])
    columns = ['','job_type']
    pd_F_job = pd.DataFrame(F_job, columns = columns)
    pd_F_job.index = np.arange(1, len(pd_F_job) + 1)
    pd_F_job.to_csv('FJSP_Job.csv', index = False)

#job들의 setup타임 생성
#랜덤범위 1~10
@st.cache_data
def setup(smin,smax):
    job_df = pd.read_csv('FJSP_Job.csv', index_col=False)
    job_df.columns = ['job_num','job_type']
    #고유값들의 수를 뽑아서 시뮬레이터 데이터 생성
    job_num = job_df['job_num'].nunique()+1
    set_df = pd.DataFrame(np.random.randint(smin, smax+1, size=(job_num, job_num)))#난수범위
    
    #컬럼과 컬럼 이름 변경
    columns = job_df['job_num'].unique()
    pd_F_set = pd.DataFrame(set_df, index=columns, columns=columns)
    pd_F_set = pd_F_set.add_prefix('j')
    pd_F_set.index = pd_F_set.index.astype(str)
    pd_F_set.index = 'j'+ pd_F_set.index 
    
    #인덱스와 컬럼이 같을 때 값을 0으로 변경
    np.fill_diagonal(pd_F_set.values, 0)
    pd_F_set.to_csv('FJSP_Set.csv')

    #머신과 job,operation의 process시간
#랜덤범위5~25
@st.cache_data
def sim(machine_count,pmin,pmax,opmin,opmax):
    
    machine = np.array(range(1, machine_count+1))
    job_df = pd.read_csv('FJSP_Job.csv', index_col=False)
    job_df.columns = ['job_num','job_type']
    job_num = job_df['job_num'].nunique()
    jo = []
    
    #오퍼레이션 랜덤 생성
    job_df['job_operation'] = np.random.randint(opmin, opmax+1, size=(job_num, 1))#난수범위
    job_df_op = job_df['job_operation']
    
    # 작업 번호와 작업의 공정 수를 이용하여 작업 이름을 생성
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

#Q_time 생성 코드
@st.cache_data
def Q_time(qmin,qmax):
    job_df = pd.read_csv('FJSP_Job.csv', index_col=0)
    job_pro = pd.read_csv('FJSP_Sim.csv', index_col=0)
    
    job_pro_index = job_pro.index
    counts = []
    current_count = 1
    # processing 데이터에서 각 작업의 공정 수 리스트를 추출
    for i in range(1, len(job_pro_index)):
        if job_pro_index[i][:3] == job_pro_index[i-1][:3]:
            current_count += 1
        else:
            counts.append(current_count)
            current_count = 1

    counts.append(current_count)
    job_df_op = counts
    
    #공정수 리스트의 최댓값과 리스트 생성
    job_df_op_max = max(job_df_op)
    job_df_op_max_1 = list(range(1,job_df_op_max+1))
    job_df_op_values = job_df_op

    j_op = job_df.index
    j_op = j_op.astype(str)
    j_op1 = 'j'+ j_op 
    j_op_num = j_op1.nunique()


    columns = job_df_op_max_1
    index = j_op1
    #빈 데이터 프레임 생성
    q_time = pd.DataFrame(index=range(j_op_num), columns=job_df_op_max_1)  # 빈 데이터프레임 생성
    
    #공정수 리스트를 려서 내용은 공정의 최댓값에서 범위를 지정해 랜덤의 값으로 입력 
    for i, val in enumerate(job_df_op_values):
        for j in range(job_df_op_max):
            if j < val:
                q_time.iloc[i, j] = int(np.random.uniform(qmin,qmax) * job_pro.iloc[i].max())

            else:
                q_time.iloc[i, j] = 0

    q_time.index = np.arange(1, len(q_time) + 1)
    q_time.index = q_time.index.astype(str)
    q_time.index = 'j'+ q_time.index

    q_time.to_csv('FJSP_Q_time.csv', index=True, header=True)'''
st.code(code, language='python')