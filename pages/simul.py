import streamlit as st
from simulator_DFJSP import *
from fjsp_Q import get_csv_file_list,get_csv_files_with_string,highlight_max
import os
import pandas as pd
from io import BytesIO
import time
# import seaborn as sns
# import matplotlib.pyplot as plt

col1, col2 = st.columns([7,1])

with col2:
    st.image("tuk_img.png")
st.title("시뮬레이터")


x1 = st.expander('사용법')
x1.write('''
- 디스패칭룰을 선택해 시뮬레이터를 돌려보는 페이지
- processing_time,setup_time,Q_time,rd_time파일 업로드 필요
                    
1. sim,setup_time,Q_time,rd_time 순서에 맞게 파일을 업로드해야함 클릭또는 드래그
2. 초록색표시로 확인메시지 확인후 데이터프레임 확인
3. 적용할 디스패칭룰 선택후 버튼을 클릭
4. 디스패칭룰 여러개 선택시 비교가능한 데이터프레임 생성
5. color에서 좋은값은 색칠 된 것을 확인 가능  

''')
st.markdown("---")



# 이전 업로드 파일 이름을 저장하기 위한 변수
prev_uploaded_filename = None

#csv_list = get_csv_file_list(save_folder)
#upload_csv = st.selectbox("CSV 파일 선택", csv_list)

# 여러 파일 업로더
uploaded_files = st.file_uploader("CSV 파일을 선택하세요.", accept_multiple_files=True)

# 업로드한 파일들을 저장할 딕셔너리
uploaded_files_dict = {}

# 업로드한 파일을 딕셔너리에 추가하고 저장

for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    
    try:
        # 파일 읽기 전에 데이터를 변수에 저장
        data = bytes_data
        uploaded_file_df = pd.read_csv(BytesIO(data))  # 저장한 데이터를 이용해 데이터프레임 생성
        # 빈 파일이 아닌 경우에만 딕셔너리와 파일 저장
        if not uploaded_file_df.empty:
            uploaded_files_dict[uploaded_file.name] = uploaded_file_df
            # 파일 저장
            save_folder = 'fjsp_stream'  # 저장할 폴더명
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            file_path = os.path.join(save_folder, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(data)  # 저장할 데이터 사용
            st.success(f"{uploaded_file.name} 파일이 업로드되었고 저장되었습니다.")
        else:
            st.warning(f"{uploaded_file.name} 파일은 비어있어 무시됩니다.")
    except pd.errors.EmptyDataError:
        st.warning(f"{uploaded_file.name} 파일은 비어있어 무시됩니다.")
    
uploaded_file_names=list(uploaded_files_dict.keys())
tabs = st.selectbox("데이터프레임 선택", list(uploaded_files_dict.keys()))

if tabs:
    with st.expander(f"데이터프레임: {tabs}"):
        selected_df = uploaded_files_dict[tabs]
        st.write(selected_df)


rule_select_list=[]    
col5,col6,col7,col8,col9 = st.columns([1,1,1,1,1])
with col5:
    spt = st.checkbox('SPT')
    if spt:
        rule_select_list.append(0)
    edd = st.checkbox('EDD')
    if edd:
        rule_select_list.append(5)
with col6:
    ssu = st.checkbox('SSU')
    if ssu:
        rule_select_list.append(1)
    mst= st.checkbox('MST')
    if mst:
        rule_select_list.append(6)
with col7:
    sptssu = st.checkbox('SPTSSU')
    if sptssu:
        rule_select_list.append(2)
    fifo = st.checkbox('FIFO')
    if fifo:
        rule_select_list.append(7)
with col8:
    mor = st.checkbox('MOR')
    if mor:
        rule_select_list.append(3)
    lifo = st.checkbox('LIFO')
    if lifo:
        rule_select_list.append(8)
with col9:
    lor = st.checkbox('LOR')
    if lor:
        rule_select_list.append(4)
    cr = st.checkbox('CR')
    if cr:
        rule_select_list.append(9)
#st.write(uploaded_file_names[0])


if st.button('클릭'):
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    total_iterations = len(rule_select_list)
    
    sim_file_name = f"{save_folder}/{uploaded_file_names[0]}"
    setup_file_name = f"{save_folder}/{uploaded_file_names[1]}"
    q_time_file_name = f"{save_folder}/{uploaded_file_names[2]}"
    rddata_file_name = f"{save_folder}/{uploaded_file_names[3]}"
    

    makespan_table = []
    machine_util_list=[]
    tardiness_list=[]
    lateness_list=[]
    util = []
    t_max_list=[]
    q_time_true_list=[]
    q_time_false_list=[]
    q_job_t_list=[]
    q_job_f_list=[]
    q_time_list=[]
    ft_table = []
    columns_name =[]

    for index, i in enumerate(rule_select_list):
        main = FJSP_simulator(sim_file_name, setup_file_name, q_time_file_name, rddata_file_name, i)
        FT,machine_util, util2, ms, tardiness, lateness, t_max,q_time_true,q_time_false,q_job_t, q_job_f, q_time = main.run()
        makespan_table.append(ms)
        util.append(util2)
        ft_table.append(FT)
        machine_util_list.append(machine_util)
        tardiness_list.append(tardiness)
        lateness_list.append(lateness)
        t_max_list.append(t_max)
        q_time_true_list.append(q_time_true)
        q_time_false_list.append(q_time_false)
        q_job_t_list.append(q_job_t)
        q_job_f_list.append(q_job_f)
        q_time_list.append(q_time)

        
        # Update progress bar
        current_progress = int((index + 1) / total_iterations * 100)
        my_bar.progress(current_progress, text=progress_text)
        if i == 0:
            s_rule_name = 'SPT'
        if i == 1:
            s_rule_name = 'SSU'
        if i == 2:
            s_rule_name = 'SPTSSU'
        if i == 3:
            s_rule_name = 'MOR'
        if i == 4:
            s_rule_name = 'LOR'
        if i == 5:
            s_rule_name = 'EDD'
        if i == 6:
            s_rule_name = 'MST'
        if i == 7:
            s_rule_name = 'FIFO'
        if i == 8:
            s_rule_name = 'LIFO'
        if i == 9:
            s_rule_name = 'CR'
        columns_name.append(s_rule_name)
    st.write(makespan_table)
    st.write(util)
    st.write(ft_table)
    re_index = ['makespan','util','machine_util','Flow_time','tardiness', 'lateness', 't_max','q_time_true','q_time_false','q_job_true', 'q_job_false', 'q_total_over_time']
    re_data = [makespan_table ,util,machine_util_list,ft_table,tardiness_list, lateness_list, t_max_list,q_time_true_list,q_time_false_list,q_job_t_list, q_job_f_list, q_time_list]
    rule_result_df = pd.DataFrame(data = re_data, columns =columns_name, index = re_index)
    st.write(rule_result_df)
    styled_df = rule_result_df.style.apply(highlight_max, axis=1)
    with st.expander("color"):
        st.write(styled_df)

    # for col_name in rule_result_df.index:
    #     plt.figure(figsize=(8, 6))
    #     sns.histplot(rule_result_df.loc[col_name], bins=10, kde=True,orient='vertical')
    #     plt.title(f'Histogram of {col_name}')
    #     plt.xlabel('Value')
    #     plt.ylabel('Frequency')
    #     st.pyplot(plt)
 
    
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