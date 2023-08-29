import streamlit as st
import pandas as pd
import numpy as np
from fjsp_Q import *
import os
import matplotlib.pyplot as plt
import time

# Display editor's content as you type
st.set_page_config(page_title="data_set")

col1, col2 = st.columns([8, 2])

with col2:
    st.image("tuk_img.png")

st.title('데이터셋')

with st.sidebar:
    selecop = st.selectbox('설정 메뉴', ('sim.csv', 'setup.csv', 'Q-time.csv','error_create.csv','rd_time.csv'))
    
# if selecop == '파일 업로드':
#     uploaded_file = st.file_uploader("CSV 파일을 업로드하세요.", type=["csv"])
    
#     # 이전 업로드 파일 이름을 저장하기 위한 변수
#     prev_uploaded_filename = None
    
#     csv_list = get_csv_file_list(save_folder)
#     upload_csv = st.selectbox("CSV 파일 선택", csv_list)

#     if uploaded_file is not None:
#         uploaded_file_df = pd.read_csv(uploaded_file)
#         st.write("업로드된 파일 정보:")
#         st.write(uploaded_file.name)

        
#         save_folder = 'fjsp_csv_folder'
#          # 업로드된 파일을 저장

        
            
#         st.write(uploaded_file_df)
#         st.write(uploaded_file_df.shape)
        
#         if st.button("저장하기"):
#             if not os.path.exists(save_folder):
#                 os.makedirs(save_folder)

#             st.write("파일이 저장되었습니다.")
#             file_path = os.path.join(save_folder,uploaded_file.name)
#             with open(file_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())
#             # 파일이 변경되었는지 확인하고 Streamlit 애플리케이션 재실행
#             if prev_uploaded_filename != uploaded_file.name:
#                 prev_uploaded_filename = uploaded_file.name
#                 st.experimental_rerun()

#                 time.sleep(5)  # 일시적으로 쉬는 시간을 주어 업데이트 주기 조절 가능


#     else:
#         st.write('파일을 업로드 하세요')
#     # 계속해서 업데이트되는 내용 표시
    
save_folder = 'fjsp_csv_folder'
number2=0 
value5=0 
value6=0
value7=0
value8=0
if selecop == 'sim.csv':
    st.subheader('sim.csv파일 생성 설정')

    x1 = st.expander('사용법')
    x1.write('''
    - Processing_time 을 생성하는 페이지
    - 모든 작업은 이 작업을 수행 후 진행되어야 함              
    1. 작업의 갯수를 설정(job_type)
    2. 기계의 갯수를 설정
    3. 생성될 processing time의 난수 범위 설정
    4. 각 작업당 공정(job_operation)을 생성할 난수 범위 설정
    5. 원하는 csv이름을 지정후 생성 및 다운
    ''')
    st.markdown("---")

    

    number = st.number_input('job의 갯수',step=1)
    number = int(number)

    number2 = st.number_input('기계의 갯수')
    number2 = int(number2)

    value5, value6 = st.slider('processtime의 난수범위', 0, 100, (1, 10))
    st.write('선택범위', value5, value6)

    value7, value8 = st.slider('job_operation의 난수범위', 0, 100, (1, 10))
    st.write('선택범위', value7, value8)
    
    processing_time_file_name = st.text_input("파일 이름을 입력하세요 (확장자 없이):", "FJSP_Sim")
    if st.button('sim.csv생성'):
        sim(processing_time_file_name,number,number2, value5, value6, value7, value8)
        sim_df = pd.read_csv('FJSP_Sim.csv', index_col=False)
        st.header(processing_time_file_name + ".csv")
        st.write(sim_df)
        
        with open('FJSP_Sim.csv') as f:
            st.download_button(f"Download {processing_time_file_name}.csv", f, file_name=f"{processing_time_file_name}.csv", mime='text/csv')
            
#job_df_op = sim(number2, value5, value6, value7, value8)

if selecop == 'setup.csv':
    
    st.subheader('set.csv파일 생성 설정')
    
    x1 = st.expander('사용법')
    x1.write('''
    - Setup_time을 생성하는 페이지
    - 불러오는 파일에 맞게 작업이 설정됨
    - sim.csv생성이 선행 되어야 함
                       
    1. Setup_time의 난수 범위 설정 가능
    2. 원하는 csv이름을 지정후 생성 및 다운
    ''')
    st.markdown("---")

    job_target_string = 'FJSP_Si'
    
    # csv_files = get_csv_files_with_string(save_folder,job_target_string)
    # selected_sim_csv = st.selectbox("CSV 파일 선택", csv_files)
    # st.write(f"선택된 CSV 파일: {selected_sim_csv}")

    value3, value4 = st.slider('setup시간 난수의 범위', 0, 100, (1, 10))
    st.write('선택범위', value3, value4)

    set_time_file_name = st.text_input("파일 이름을 입력하세요 (확장자 없이):", "FJSP_Set")
    if st.button('setup.csv생성'):
        setup(set_time_file_name,value3, value4)
        setup_df = pd.read_csv(f'{set_time_file_name}.csv', index_col=False)
        st.header(set_time_file_name + ".csv")
        st.write(setup_df)
        
        with open(f'{set_time_file_name}.csv') as f:
            st.download_button(f"Download {set_time_file_name}.csv", f, file_name=f"{set_time_file_name}.csv", mime='text/csv')




if selecop == 'Q-time.csv':
    st.subheader('Q-time 생성 설정')
    x1 = st.expander('사용법')
    x1.write('''
    - Queue_tiem을 생성하는 페이지
    - 불러오는 파일에 맞게 기본값이 설정됨
    - sim.csv생성이 선행 되어야 함
                       
    1. 적용 배수 설정(processing_time의 최댓값 X 적용배수 )
    2. 원하는 csv이름을 지정후 생성 및 다운
    ''')
    st.markdown("---")
    #st.subheader('Q-time.csv 생성 설정')

    sim_target_string = "FJSP_Si"
    
    # csv_files1 = get_csv_files_with_string(save_folder,sim_target_string)
    # selected_sim_csv = st.selectbox("CSV 파일 선택", csv_files1)
    # st.write(f"선택된 CSV 파일: {selected_sim_csv}")

    # Q_time 생성 입력값 
    q_range_min, q_range_max = st.slider('Q-time에 적용될 배수 범위', 0.0, 5.0, (1.5, 2.0),step = 0.1)
    st.write('선택범위', q_range_min, q_range_max)
    Q_time_file_name = st.text_input("파일 이름을 입력하세요 (확장자 없이):", "FJSP_Q_time")
    if st.button('Q-time 생성'):

        Q_time(Q_time_file_name,q_range_min, q_range_max)
        q_time_df = pd.read_csv(f'{Q_time_file_name}.csv', index_col=False)
        st.header(Q_time_file_name+".csv")
        st.write(q_time_df)

        
        with open(f'{Q_time_file_name}.csv') as f:
            st.download_button(f"Download {Q_time_file_name}.csv", f, file_name=f"{Q_time_file_name}.csv", mime='text/csv')

    # sim.csv 생성 이후에 Q_time.csv 생성

if selecop == 'error_create.csv':
    st.subheader('error설정')
    x1 = st.expander('사용법')
    x1.write('''
    - processing_time데이터에서 Error(=0)을 생성하는 페이지
    - 불러오는 파일에 맞게 기본값이 설정됨
    - sim.csv생성이 선행 되어야 함
                      
    1. 생성할 에러의 갯수를 설정
    2. 기계와 작업을 지정시 해당 기계의 해당 작업이 값이 모두 0으로 변경됨
    3. '공정 지정'을 클릭 후 설정시 해당 기계와 작업의 지정 공정이 값이 0으로 변경됨
    4. 원하는 csv이름을 지정후 'csv선택'에서 Error테이블과 적용된 테이블 확인가능
    5. Error가 적용된 processing_time.csv파일 다운 가능                          
    ''')
    st.markdown("---")   

    # sim_target_string1 = "FJSP_Si"
    
    # csv_files2 = get_csv_files_with_string(save_folder,sim_target_string1)
    # selected_sim_csv2 = st.selectbox("CSV 파일 선택", csv_files2)
    # st.write(f"선택된 CSV 파일: {selected_sim_csv2}")
    

    job_df_op,machine_max,job_df_op_count = sim_list_remind()
    
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
    
    
    selecop_er = st.selectbox('csv 선택', ('', '문제 리스트', 'error_processing_time.csv'))

    
    
    if selecop_er == '문제 리스트':
        st.write(unavailable_machine_options_pd)
    if selecop_er == 'error_processing_time.csv':
        add_unavailable_machines_to_sim(error_file_name, unavailable_machine_options)
        error_processing_df = pd.read_csv(f'{error_file_name}.csv', index_col=False)
        
        st.header(error_file_name+".csv")
        error_processing_styled_df = error_processing_df.style.applymap(highlight_zero)
        st.write(error_processing_styled_df)
        st.write(error_processing_df.shape)

        
        with open(f'{error_file_name}.csv') as f:
            st.download_button(f"Download {error_file_name}.csv", f, file_name=f"{error_file_name}.csv", mime='text/csv')
        # with open('error_processing.csv') as f:
        #     st.download_button('Download CSV', f, file_name='error_prcessing.csv', mime='text/csv')

if selecop == 'rd_time.csv':

    x = st.expander('사용법')
    x.write('''
    - release_time과 due_time을 생성하는 페이지
    - 설정은 불러오는 파일에 맞게 작업의 갯수가 설정됨
    - sim.csv생성이 선행 되어야 함
                       
    1. 각 작업당 총 생산량을 설정
    2. 각 작업의 바로 시작 가능한 물량을 설정
    3. 각 작업의 평균 작업시간 설정 
    4. 평균 작업시간의 난수 범위를 설정
    5. 원하는 csv파일 이름을 지정 후 다운
    ''')
    st.markdown("---")
    filtered_result=None
    st.subheader('Release_time,due_time 생성') 
      

    # sim_target_string1 = "FJSP_Si"
    
    # csv_files3 = get_csv_files_with_string(save_folder,sim_target_string1)
    # selected_sim_csv3 = st.selectbox("CSV 파일 선택", csv_files3)
    # st.write(f"선택된 CSV 파일: {selected_sim_csv3}")

    job_df_op,machine_max,job_df_op_count = sim_list_remind()
    job_product_list=[]

    num_inputs1 = list(range(1,len(job_df_op)+1))

    num_per_row = 5
    num_rows = (len(num_inputs1) + num_per_row - 1) // num_per_row
    #st.write(sorted_counter)
   
    st.markdown("---")
    st.write('총 생산량 설정')
    for row_idx in range(num_rows):
        row_start = row_idx * num_per_row
        row_end = min(row_start + num_per_row, len(num_inputs1))
        
        # st.columns()를 사용하여 한 행의 열 생성
        columns = st.columns(row_end - row_start)
        
        for idx, col in enumerate(columns):
            data_idx = row_start + idx + 1  # 인덱스 1부터 시작하도록 설정
            if data_idx <= len(num_inputs1):
                with col:
                    # 도착물량 입력 받기
                    job_product = st.number_input(f'{data_idx}작업 총 물량', key=f'job_product{data_idx}', min_value=1, max_value=1000, step=1)
                    job_product = int(job_product)
            job_product_list.append(job_product)

    filtered_result, sorted_counter = filtered_result_create(job_product_list)
    
    #st.write(filtered_result)
    #st.write(sorted_counter)


    first_release_supply=[]
    arrival_time_list = []
    # st.columns()를 사용하여 열 생성
    # 5개씩 열을 배치하여 데이터 표시

    num_inputs = [sorted_counter[idx] for idx in range(1, len(sorted_counter) + 1)]

    num_per_row1 = 5
    num_rows1 = (len(num_inputs) + num_per_row1 - 1) // num_per_row1

    st.markdown("---")
    st.write('초기 생산량 설정')
    for row_idx in range(num_rows1):
        row_start = row_idx * num_per_row1
        row_end = min(row_start + num_per_row1, len(num_inputs))
        
        # st.columns()를 사용하여 한 행의 열 생성
        columns = st.columns(row_end - row_start)
        
        for idx, col in enumerate(columns):
            data_idx = row_start + idx + 1  # 인덱스 1부터 시작하도록 설정
            if data_idx <= len(num_inputs):
                with col:
                    # 도착물량 입력 받기
                    arrival_number = st.number_input(f'{data_idx}작업 초기 물량', key=f'arrival_{data_idx}', min_value=0, max_value=sorted_counter[data_idx], step=1)
                    arrival_number = int(arrival_number)
            first_release_supply.append(arrival_number)

    st.markdown("---")
    st.write('작업당 평균 작업시간 설정')                
    for row_idx in range(num_rows1):
        row_start = row_idx * num_per_row1
        row_end = min(row_start + num_per_row1, len(num_inputs))
        
        columns = st.columns(row_end - row_start)
        
        for idx, col in enumerate(columns):
            data_idx = row_start + idx + 1
            if data_idx <= len(num_inputs):
                with col:
                    # 평균 작업 시간 입력 받기
                    arrival_time = st.number_input(f'{data_idx}평균 작업 시간', key=f'arrival_time_{data_idx}', min_value=0, max_value=100, step=1,value=50)
                    arrival_time = int(arrival_time)
            arrival_time_list.append(arrival_time)
    st.markdown("---")
    r_min, r_max = st.slider('도착시간 난수 범위', 0.0, 4.0, (0.8, 1.2),step = 0.1)
    st.write(first_release_supply)
    st.write(arrival_time_list)

    rd_csv_name = st.text_input("파일 이름을 입력하세요 (확장자 없이):", "FJSP_rd_time")
    if st.button('rd_time 생성'):

        rd_df = release_due_data(rd_csv_name,filtered_result,first_release_supply,arrival_time_list,r_min,r_max)
        rd_time_df = pd.read_csv(f'{rd_csv_name}.csv', index_col=0)
        st.header(rd_csv_name+".csv")
        st.write(rd_time_df)
        st.write(rd_time_df.shape)

        
        with open(f'{rd_csv_name}.csv') as f:
            st.download_button(f"Download {rd_csv_name}.csv", f, file_name=f"{rd_csv_name}.csv", mime='text/csv')
        
    

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
