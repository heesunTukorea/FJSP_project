import streamlit as st
from simulator_DFJSP import *
from fjsp_Q import get_csv_file_list,get_csv_files_with_string
import os
import pandas as pd

col1, col2 = st.columns([7,1])

with col2:
    st.image("tuk_img.png")
st.title("시뮬레이터")



# 이전 업로드 파일 이름을 저장하기 위한 변수
prev_uploaded_filename = None

#csv_list = get_csv_file_list(save_folder)
#upload_csv = st.selectbox("CSV 파일 선택", csv_list)

# 여러 파일 업로더
uploaded_files = st.file_uploader("CSV 파일을 선택하세요.", accept_multiple_files=True)

# 업로드한 파일들을 저장할 딕셔너리
uploaded_files_dict = {}

# 업로드한 파일을 딕셔너리에 추가
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    try:
        uploaded_file_df = uploaded_file.read()
        # 빈 파일인 경우 데이터프레임이 생성되지 않습니다.
        if not uploaded_file_df.empty:
            uploaded_files_dict[uploaded_file.name] = uploaded_file_df
    except pd.errors.EmptyDataError:
        # 빈 파일인 경우 오류가 발생하므로, 오류를 처리합니다.
        st.warning(f"{uploaded_file.name} 파일은 비어있어 데이터프레임을 생성할 수 없습니다.")

# 선택된 탭 이름
tabs = st.selectbox("데이터프레임 선택", list(uploaded_files_dict.keys()))

if tabs:
    with st.beta_expander(f"데이터프레임: {tabs}"):
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

if st.button('클릭'):
    makespan_table = []
    util = []
    ft_table = []

    for i in rule_select_list:
        main = FJSP_simulator('C:/Users/parkh/git_tlsgudcks/simulator/data/DFJSP_test.csv','C:/Users/parkh/git_tlsgudcks/simulator/data/DFJSP_setup_test.csv',
                            "C:/Users/parkh/git_tlsgudcks/simulator/data/DFJSP_Qdata_test.csv","C:/Users/parkh/git_tlsgudcks/simulator/data/DFJSP_rdData_test2.csv",i)
        FT, util2, ms = main.run()
        makespan_table.append(ms)
        util.append(util2)
        ft_table.append(FT)
    print(makespan_table)
    print(ft_table)
    print(util)
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