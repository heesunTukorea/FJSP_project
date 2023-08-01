import streamlit as st
import pandas as pd
import numpy as np
from fjsp_Q import *
import os
import matplotlib.pyplot as plt


# Display editor's content as you type
st.set_page_config(page_title="data_set")

col1, col2 = st.columns([8, 2])

with col2:
    st.image("tuk_img.png")

st.title('데이터셋')

with st.sidebar:
    selecop = st.selectbox('설정 메뉴', ('job.csv', 'setup.csv', 'sim.csv', 'Q-time.csv','error_create.csv'))


if selecop == 'job.csv':
    st.subheader('job.csv파일 생성 설정')
    csv_files = get_csv_file_list()
    # selected_csv = st.selectbox("CSV 파일 선택", csv_files)
    # st.write(f"선택된 CSV 파일: {selected_csv}")
    number = st.number_input('job의 갯수',step=1)
    number = int(number)
    value1, value2 = st.slider('job타입의 난수범위', 0, 100, (1, 10))
    st.write('선택범위', value1, value2)

     # 파일 다운로드 이후에는 파일 이름 입력을 비활성화
    job_time_file_name = st.text_input("파일 이름을 입력하세요 (확장자 없이):", "FJSP_Job", key="job_time_file_name")

    if st.button('job.csv생성'):
        job(number, value1, value2)
        st.write('생성완료')

        # 생성한 job.csv 파일을 읽어와서 출력
        job_df = pd.read_csv('FJSP_Job.csv', index_col=False)
        st.subheader(job_time_file_name + '.csv')
        st.dataframe(job_df)
        with open('FJSP_Job.csv') as f:
            st.download_button(f"Download {job_time_file_name}.csv", f, file_name=f"{job_time_file_name}.csv", mime='text/csv')

       

if selecop == 'setup.csv':
    st.subheader('set.csv파일 생성 설정')

    value3, value4 = st.slider('setup시간 난수의 범위', 0, 100, (1, 10))
    st.write('선택범위', value3, value4)

    set_time_file_name = st.text_input("파일 이름을 입력하세요 (확장자 없이):", "FJSP_Set")
    if st.button('setup.csv생성'):
        setup(value3, value4)
        setup_df = pd.read_csv('FJSP_Set.csv', index_col=False)
        st.header(set_time_file_name + ".csv")
        st.write(setup_df)
        
        with open('FJSP_Set.csv') as f:
            st.download_button(f"Download {set_time_file_name}.csv", f, file_name=f"{set_time_file_name}.csv", mime='text/csv')

number2=0 
value5=0 
value6=0
value7=0
value8=0
if selecop == 'sim.csv':
    st.subheader('sim.csv파일 생성 설정')

    number2 = st.number_input('기계의 갯수')
    number2 = int(number2)
    value5, value6 = st.slider('processtime의 난수범위', 0, 100, (1, 10))
    st.write('선택범위', value5, value6)

    value7, value8 = st.slider('job_operation의 난수범위', 0, 100, (1, 10))
    st.write('선택범위', value7, value8)
    processing_time_file_name = st.text_input("파일 이름을 입력하세요 (확장자 없이):", "FJSP_Sim")
    if st.button('sim.csv생성'):
        sim(number2, value5, value6, value7, value8)
        sim_df = pd.read_csv('FJSP_Sim.csv', index_col=False)
        st.header(processing_time_file_name + ".csv")
        st.write(sim_df)
        
        with open('FJSP_Sim.csv') as f:
            st.download_button(f"Download {processing_time_file_name}.csv", f, file_name=f"{processing_time_file_name}.csv", mime='text/csv')
#job_df_op = sim(number2, value5, value6, value7, value8)


if selecop == 'Q-time.csv':
    st.subheader('Q-time.csv 생성 설정')

    # Q_time 생성 입력값 
    q_range_min, q_range_max = st.slider('Q-time에 적용될 배수 범위', 0.0, 5.0, (1.5, 2.0))
    st.write('선택범위', q_range_min, q_range_max)
    Q_time_file_name = st.text_input("파일 이름을 입력하세요 (확장자 없이):", "FJSP_Q_time")
    if st.button('Q-time 생성'):

        Q_time(q_range_min, q_range_max)
        q_time_df = pd.read_csv('FJSP_Q_time.csv', index_col=False)
        st.header(Q_time_file_name+".csv")
        st.write(q_time_df)

        
        with open('FJSP_Q_time.csv') as f:
            st.download_button(f"Download {Q_time_file_name}.csv", f, file_name=f"{Q_time_file_name}.csv", mime='text/csv')

    # sim.csv 생성 이후에 Q_time.csv 생성

#error값 범위 지정을 위한 코드
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




if selecop == 'error_create.csv':
    st.subheader('error설정')   
    num_inputs = st.number_input('기계와 작업을 입력할 갯수', min_value=1, max_value=100, value=1, step=1)
    unavailable_machine_options = []
    op_number = None 
    for idx in range(num_inputs):
        o_count=0
        col_m, col_j, col_o= st.columns([3, 3, 3])
        with col_m:
            machine_number = st.number_input('기계 지정', key=f'machine_{idx}', min_value=1,max_value= machine_max, step=1)
            machine_number = int(machine_number)
            st.write("지정된 기계: "+str(machine_number))
        with col_j:
            job_number = st.number_input('작업 지정', key=f'job_{idx}', min_value=1,max_value= job_df_op_count, step=1)
            job_number = int(job_number)
            st.write("지정된 작업: "+str(job_number))
            if job_number is not None:
                job_df_op_select = job_df_op[job_number-1]
                with col_o:
                    use_op_checkbox = st.checkbox('공정 지정', key=f'op_checkbox_{idx}')
         
                    if use_op_checkbox:
                        op_number = st.number_input('공정 지정',key=f'op_{idx}', min_value=1,max_value =job_df_op_select, step=1)
                        op_number = int(op_number)
                        st.write("지정된 공정: "+str(op_number))
                    else:
                        op_number = None
            else:
                st.write("없어용")
        
        unavailable_machine_options.append([machine_number, job_number, op_number])
    
 
    unavailable_machine_options_pd = pd.DataFrame(unavailable_machine_options, columns=['기계', '작업', '공정'])

    error_file_name = st.text_input("파일 이름을 입력하세요 (확장자 없이):", "FJSP_error_processing")
    
    selecop_er = st.selectbox('', ('', '문제 리스트', 'error_processing_time.csv'))

    

    if selecop_er == '문제 리스트':
        st.write(unavailable_machine_options_pd)
    if selecop_er == 'error_processing_time.csv':
        add_unavailable_machines_to_sim("FJSP_Sim.csv", unavailable_machine_options)
        error_processing_df = pd.read_csv('error_processing.csv', index_col=False)
        
        st.header(error_file_name+".csv")
        error_processing_styled_df = error_processing_df.style.applymap(highlight_zero)
        st.write(error_processing_styled_df)
        st.write(error_processing_df.shape)

        
        with open('error_processing.csv') as f:
            st.download_button(f"Download {error_file_name}.csv", f, file_name=f"{error_file_name}.csv", mime='text/csv')
        # with open('error_processing.csv') as f:
        #     st.download_button('Download CSV', f, file_name='error_prcessing.csv', mime='text/csv')


        
    

# job_df = pd.read_csv('FJSP_Job.csv', index_col=False)
# setup_df = pd.read_csv('FJSP_Set.csv', index_col=False)
# sim_df = pd.read_csv('FJSP_Sim.csv', index_col=False)
# q_time_df = pd.read_csv('FJSP_Q_time.csv', index_col=False)


# j_count=0
# s_count=0
# p_count=0
# q_count=0

# if os.path.exists("FJSP_Job.csv"):
#     j_count +=1
# if os.path.exists("FJSP_Set.csv"):
#     s_count +=1
# if os.path.exists("FJSP_Sim.csv"):
#     p_count +=1 

# c_sum=j_count+s_count+p_count

# if c_sum==3:
#     tab1, tab2, tab3, tab4 = st.tabs(["job.csv", "setup.csv", "process_time.csv", "q_time.csv"])
#     with tab1:
#         st.header("job.csv")
#         st.write(job_df)
#     with tab2:
#         st.header("setup.csv")
#         st.write(setup_df)
#     with tab3:
#         st.header("process_time.csv")
#         st.write(sim_df)
#         if st.button('Q_time.csv생성'):
#             q_range_min,q_range_max = st.slider(
#                     'Q-time에 적용될 배수 범위',0.0,5.0,(1.5,2.0))
#             st.write('선택범위', q_range_min,q_range_max)

#             if st.button('확인'):
#                 Q_time(q_range_min,q_range_max,job_df_op)
#                 st.write('생성완료')
#                 st.header("Q_time.csv")
#                 st.write(q_time_df)
       
# else:
#     st.write("데이터 정보를 입력해 주세요")
