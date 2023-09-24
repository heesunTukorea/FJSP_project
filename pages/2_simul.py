import streamlit as st
from simulator_DFJSP import *
from fjsp_Q import get_csv_file_list,get_csv_files_with_string,highlight_max,draw_histogram
import os
import pandas as pd
from io import BytesIO
import time
from DQN import *
from run_simulator import *
from Parameter import *
import plotly.graph_objs as go
import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.pyplot as plt

col1, col2 = st.columns([7,1])

with col2:
    st.image("tuk_img.png")
st.title("시뮬레이터")


x1 = st.expander('사용법')
x1.write('''
- 데이터를 활용해 각 디스패칭 룰을 적용하거나 강화학습을 이용한 시뮬레이터 결과를 볼 수 있음
- 성능지표와 간트차트를 이용한 스케줄링 결과 확인 가능
- 필요한 파일로는 Processing_time,Setup_time,Q-time,Rd_time이 필요 

1. 각각에 알맞은 데이터를 파일을 업로드해준다
2. 업로드한 데이터를 확인 가능
3. selectbox를 통해서 각 룰을 적용 또느 강화학습을 적용 할 것인지 선택 가능

- 디스패칭룰 선택         
1. 디스패칭룰을 사용한 방법 선택시 원하는 룰 선택후 클릭
2. 결과를 데이터프레임의 형태로 띄워주고 color를 클릭하면 가장 좋은 값 확인 가능
         
- 강화학습 선택         
1. 클릭을 누르게되면 학습하는 과정을 기다리면서 볼 수 있고 오래걸림
2. 결과를 데이터프레임의 형태로 띄워주고 nomorspt.pt 파일을 다운로드 가능

- 결과
1. 간단한 시각화와 가장 최근에 돌린 디스패칭룰 시뮬레이터와 비교 가능
2. 간트차트를 이용한 시각화 확인 가능                           
''')
st.markdown("---")

st.subheader("데이터 업로드")



# 이전 업로드 파일 이름을 저장하기 위한 변수
prev_uploaded_filename = None
save_folder = 'fjsp_stream' 
#csv_list = get_csv_file_list(save_folder)
#upload_csv = st.selectbox("CSV 파일 선택", csv_list)

# 여러 파일 업로더
# uploaded_files = st.file_uploader("CSV 파일을 선택하세요.", accept_multiple_files=True)

# 업로드한 파일들을 저장할 딕셔너리
uploaded_files_dict = {}

# 업로드한 파일을 딕셔너리에 추가하고 저장
    # 여러 파일 업로더
uploaded_files_p = st.file_uploader("Processing 파일을 선택하세요.")
uploaded_files_s = st.file_uploader("Set up 파일을 선택하세요.")
uploaded_files_q = st.file_uploader("Q-time 파일을 선택하세요.")
uploaded_files_r = st.file_uploader("Rd_time 파일을 선택하세요.")
uploaded_files = [uploaded_files_p,uploaded_files_s,uploaded_files_q,uploaded_files_r]
if uploaded_files_p and uploaded_files_s and uploaded_files_q and uploaded_files_r is not None:
    for uploaded_file in uploaded_files:
        try:
            bytes_data = uploaded_file.read()
        except:
            st.warning('파일을 넣어주세요')
        
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
else:
    st.warning('파일을 넣어주세요')

        
uploaded_file_names=list(uploaded_files_dict.keys())
#tabs = st.selectbox("데이터프레임 선택", list(uploaded_files_dict.keys()))

if len(uploaded_file_names) >= 4:
    with st.expander("데이터프레임"):
        tab_1,tab_2,tab_3,tab4 = st.tabs(uploaded_file_names)
        t_list = [tab_1,tab_2,tab_3,tab4]
        for i in range(len(t_list)):
            t_list[i-1].subheader(uploaded_file_names[i-1])
            selected_df = uploaded_files_dict[uploaded_file_names[i-1]]
            t_list[i-1].write(selected_df)
# if tabs:
#     with st.expander(f"데이터프레임: {tabs}"):
#         selected_df = uploaded_files_dict[tabs]
#         st.write(selected_df)
if len(uploaded_file_names) >= 4:
    simul_select = st.selectbox('테스트 방식',('dispatching_rule_select','DQN'))
    st.markdown("---")
    if simul_select =='dispatching_rule_select':
        r_param = {
            "gamma": 0,
            "learning_rate": 0,
            "batch_size": 0,
            "buffer_limit": 0,
            "input_layer" : 0,
            "output_layer" : 0,
            "episode" :0
        }
        st.write('디스패칭 룰을 선택해주세요')
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

        simul_file_name = st.text_input("파일 이름을 입력하세요 (확장자 없이):", "FJSP_simul")
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
            util_list = []
            t_max_list=[]
            q_time_true_list=[]
            q_time_false_list=[]
            q_job_t_list=[]
            q_job_f_list=[]
            q_time_list=[]
            ft_table = []
            columns_name =[]
            fig_list = []
            # fig2_dict = {}
            # fig3_dict = {}
            # fig4_dict = {}
            # fig5_dict = {}
            # fig6_dict = {}
            # fig7_dict = {}
            # fig8_dict = {}

            for index, i in enumerate(rule_select_list):
                param = Parameters(sim_file_name, setup_file_name, q_time_file_name, rddata_file_name,r_param)

                #main = FJSP_simulator(sim_file_name, setup_file_name, q_time_file_name, rddata_file_name, i)
                simulator_si = Run_Simulator(param)

                fig,fig2,fig3,fig4,fig5,fig6,fig8,Flow_time, machine_util, util, makespan, tardiness, lateness, t_max,q_time_true,q_time_false,q_job_t, q_job_f, q_time = simulator_si.main("DSP_run",i)
                # FT,machine_util, util2, ms, tardiness, lateness, t_max,q_time_true,q_time_false,q_job_t, q_job_f, q_time = main.run()
                #fig,fig2,fig3,fig4,fig5,fig6,fig7,fig8 = main.gannt_chart()
                makespan_table.append(makespan)
                util_list.append(util)
                ft_table.append(Flow_time)
                machine_util_list.append(machine_util)
                tardiness_list.append(tardiness)
                lateness_list.append(lateness)
                t_max_list.append(t_max)
                q_time_true_list.append(q_time_true)
                q_time_false_list.append(q_time_false)
                q_job_t_list.append(q_job_t)
                q_job_f_list.append(q_job_f)
                q_time_list.append(q_time)
                fig_list.append([fig,fig2,fig3,fig4,fig5,fig6,fig8]) 
                # fig2_dict[i] = fig2 
                # fig3_dict[i] = fig3 
                # fig4_dict[i] = fig4 
                # fig5_dict[i] = fig5 
                # fig6_dict[i] = fig6 
                # fig7_dict[i] = fig7 
                # fig8_dict[i] = fig8 
                
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
                # st.plotly_chart(fig)
                # st.plotly_chart(fig2)
                # st.plotly_chart(fig3)
                # st.plotly_chart(fig4)
                # st.plotly_chart(fig5)
                # st.plotly_chart(fig6)
                # st.plotly_chart(fig7)
                #st.plotly_chart(fig8)
                # fig8_png = st.plotly_chart(fig8)
                # fig8_png.write_image("fig8.png")
            # st.write(makespan_table)
            # st.write(util)
            # st.write(ft_table)
            re_index = ['makespan','util','machine_util','Flow_time','tardiness', 'lateness', 't_max','q_time_true','q_time_false','q_job_true', 'q_job_false', 'q_total_over_time']
            re_data = [makespan_table ,util_list,machine_util_list,ft_table,tardiness_list, lateness_list, t_max_list,q_time_true_list,q_time_false_list,q_job_t_list, q_job_f_list, q_time_list]
            rule_result_df = pd.DataFrame(data = re_data, columns =columns_name, index = re_index)
            st.write(rule_result_df)
            styled_df = rule_result_df.style.apply(highlight_max, axis=1)
            rule_result_df.to_csv('simul_result.csv', index=True, header=True)
            rule_result_df.to_csv(f'{simul_file_name}.csv', index=True, header=True)
            with open(f'{simul_file_name}.csv') as f:
                st.download_button(f"Download {simul_file_name}.csv", f, file_name=f"{simul_file_name}.csv", mime='text/csv')
            with st.expander("color"):
                st.write(styled_df)
            with st.expander("simul_result"):
                #fig  = plot_histograms(rule_result_df)
                # 컬럼 이름을 "Algorithm" 컬럼으로 설정
                #rule_result_df.set_index('Algorithm', inplace=True)
                
                rule_result_df1 = rule_result_df.T
                rule_result_df2 = rule_result_df1[['makespan','q_total_over_time']] 
                # 히 스토그램을 그릴 인덱스 선택 (makespan과 q_total_over_time)
                selected_indices = ['makespan', 'q_total_over_time']

            #    rule_result_df = rule_result_df.reset_index().melt(id_vars=['index'])
                # 데이터프레임을 Melt하여 필요한 형태로 변환
                rule_result_df2 = rule_result_df2.reset_index().melt(id_vars=['index'])
                # Plotly로 나란히 두 개의 막대 그래프 그리기
                fig = px.bar(rule_result_df2, x='index', y='value', color='variable', barmode='group',
                            labels={'index': 'Data', 'value': 'Value', 'variable': 'Category'})

                # 그래프에 레이아웃 설정
                fig.update_layout(
                    xaxis=dict(tickvals=list(range(len(rule_result_df2['index']))), ticktext=rule_result_df2['index']),
                    xaxis_title='Data',
                    yaxis_title='Value',
                    showlegend=True
                )

                # 스트림릿에 그래프 표시
                st.plotly_chart(fig)

            with st.expander("simul_extra_result"):
                    g_index = ['makespan','util','Flow_time','tardiness', 'lateness', 't_max','q_time_true','q_job_true','q_total_over_time']
                    tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9 =st.tabs(g_index)
                    tab_l = [tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9]
                    for i in range(len(tab_l)+1):
                        tab_l[i-1].subheader(g_index[i-1])
                        rule_extra_result = rule_result_df1[g_index[i-1]]
                        tab_l[i-1].bar_chart(rule_extra_result)

            tab_list =['m_on_job_number','machine_gantt','DSP_gantt','step_DSP_gantt','heatMap_gantt','main_gantt','job_gantt_for_Q_time']
            #aa = st.selectbox("rule_select",(columns_name))
            for index,i in enumerate(columns_name):
                with st.expander(f"{i}"):
                    tab1,tab2,tab3,tab4,tab5,tab6,tab8 =st.tabs(tab_list)
                    tab_l = [tab1,tab2,tab3,tab4,tab5,tab6,tab8]
                    for j in range(len(tab_l)):
                        # if j == 1:
                        #     tab1.subheader("machine_on_job_number")
                        #     fig_n = go.Figure(data=fig_list[index][j-1]) 
                        #     tab1.plotly_chart(fig_n, use_container_width=True)
                        # else:
                        tab_l[j-1].subheader(tab_list[j-1])
                        fig_n = go.Figure(data=fig_list[index][j-1]) 
                        tab_l[j-1].plotly_chart(fig_n, use_container_width=True)
            # fig_n = go.Figure(data=fig_list[0][2]) 
            # st.plotly_chart(fig_n, use_container_width=True) 
    if simul_select == 'DQN':
        st.write('인공신경망을 통한 시뮬레이션') 
        st.markdown("---")
        # 디렉토리를 생성하여 그래프 이미지를 저장합니다.
        # output_dir = 'graph_images'
        # os.makedirs(output_dir, exist_ok=True)
        st.write('강화학습 파라미터 조정')
        col1,col2,col3 = st.columns([1,1,1])
        
        with col1:
            gamma = st.number_input('gamma',value = 0.99 , min_value=0.00,max_value=100000.00, step=0.1)
            buffer_limit = st.number_input('buffer_limit',value = 50000 , min_value=0, max_value=100000, step=1)
            episode = st.number_input('episode',value = 1000 , min_value=0, max_value=100000, step=1)
        with col2:
            learning_rate = st.number_input('learning_rate', value=0.0003, min_value=0.0000, max_value=10.0000, step=0.0001, format="%.4f")
            input_layer = st.number_input('input_layer',value = 12 , min_value=0, max_value=100000, step=1)
        with col3:
            batch_size = st.number_input('batch_size',value = 32 , min_value=0, max_value=100000, step=1)
            output_layer = st.number_input('output_layer',value = 10 , min_value=0, max_value=100000, step=1)
       
            
            
        
        makespan_table_d=[]
        machine_util_list_d=[]
        tardiness_list_d=[]
        lateness_list_d=[]
        util_d = []
        t_max_list_d=[]
        q_time_true_list_d=[]
        q_time_false_list_d=[]
        q_job_t_list_d=[]
        q_job_f_list_d=[]
        q_time_list_d=[]
        ft_table_d = []
        columns_name_d =[]
        score_list_d=[]
        fig_list_d = []
        # simul_file_name_d = st.text_input("csv 파일 이름을 입력하세요 (확장자 없이):", "FJSP_simul")
        pt_name = st.text_input("pt 파일 이름을 입력하세요 (확장자 없이):", "nomorspt")
        if st.button('클릭'):
            sim_file_name = f"{save_folder}/{uploaded_file_names[0]}"
            setup_file_name = f"{save_folder}/{uploaded_file_names[1]}"
            q_time_file_name = f"{save_folder}/{uploaded_file_names[2]}"
            rddata_file_name = f"{save_folder}/{uploaded_file_names[3]}"
            params = {
                "p_data" : sim_file_name,
                "s_data" : setup_file_name,
                "q_data" : q_time_file_name,
                "rd_data" : rddata_file_name
            }

            r_param = {
                "gamma": gamma,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "buffer_limit": buffer_limit,
                "input_layer" : input_layer,
                "output_layer" : output_layer,
                "episode" : episode
            }

            for i in range(1):
                param_d = Parameters(sim_file_name, setup_file_name, q_time_file_name, rddata_file_name,r_param)
                #fig,fig2,fig3,fig4,fig5,fig6,fig7,fig8,Flow_time, machine_util, util, makespan, Tardiness_time, Lateness_time, T_max,q_time_true,q_time_false,q_job_t, q_job_f,q_over_time, score =dqn_r.main_d()
                #fig,fig2,fig3,fig4,fig5,fig6,fig7,fig8 = main_d.gannt_chart()
                simulator_d = Run_Simulator(param_d)

                fig,fig2,fig3,fig4,fig5,fig6,fig8,Flow_time, machine_util, util, makespan, tardiness, lateness, t_max,q_time_true,q_time_false,q_job_t, q_job_f, q_time = simulator_d.main("DQN",'SPT')

                makespan_table_d.append(makespan)
                util_d.append(util)
                ft_table_d.append(Flow_time)
                machine_util_list_d.append(machine_util)
                tardiness_list_d.append(tardiness)
                lateness_list_d.append(lateness)
                t_max_list_d.append(t_max)
                q_time_true_list_d.append(q_time_true)
                q_time_false_list_d.append(q_time_false)
                q_job_t_list_d.append(q_job_t)
                q_job_f_list_d.append(q_job_f)
                q_time_list_d.append(q_time)
                #score_list_d.append(score)
                fig_list_d.append([fig,fig2,fig3,fig4,fig5,fig6,fig8]) 
                
                re_index = ['makespan','util','machine_util','Flow_time','tardiness', 'lateness', 't_max','q_time_true','q_time_false','q_job_true', 'q_job_false', 'q_total_over_time']
                re_data = [makespan_table_d ,util_d,machine_util_list_d,ft_table_d,tardiness_list_d, lateness_list_d, t_max_list_d,q_time_true_list_d,q_time_false_list_d,q_job_t_list_d, q_job_f_list_d, q_time_list_d]
                rule_result_df = pd.DataFrame(data = re_data, columns = ['result'], index = re_index)
                pd_result = pd.read_csv('simul_result.csv', index_col=0)
                pd_result['DQN'] = rule_result_df['result']
                tab1,tab2 =st.tabs(['dqn_result','all_result'])
                tab1.subheader('dqn_result')
                tab1.write(rule_result_df)
                tab2.subheader('all_result')
                tab2.write(pd_result)
                
                styled_df = pd_result.style.apply(highlight_max, axis=1)
                rule_result_df.to_csv('simul_result_dqn.csv', index=True, header=True)


                with open('nomorspt.pt','rb') as f:
                    st.download_button(f"Download {pt_name}.pt", f, file_name=f"{pt_name}.pt", mime='application/octet-stream')
                # with open(f'{simul_file_name}.csv') as f:
                #     st.download_button(f"Download {simul_file_name}.csv", f, file_name=f"{simul_file_name}.csv", mime='text/csv')
                with st.expander("color"):
                    st.write(styled_df)
                
                # rule_result_df.to_csv(f'{simul_file_name_d}.csv', index=True, header=True)
                # with open(f'{simul_file_name_d}.csv') as f:
                #     st.download_button(f"Download {simul_file_name_d}.csv", f, file_name=f"{simul_file_name_d}.csv", mime='text/csv')
                
                
                tab_list =['m_on_job_number','machine_gantt','DSP_gantt','step_DSP_gantt','heatMap_gantt','main_gantt','job_gantt_for_Q_time']
                #aa = st.selectbox("rule_select",(columns_name))
                # for index,i in enumerate(columns_name):
                with st.expander("simul_result"):
                    #fig  = plot_histograms(rule_result_df)
                    # 컬럼 이름을 "Algorithm" 컬럼으로 설정
                    #rule_result_df.set_index('Algorithm', inplace=True)
                    
                    rule_result_df1 = pd_result.T
                    rule_result_df2 = rule_result_df1[['makespan','q_total_over_time']] 
                    # 히 스토그램을 그릴 인덱스 선택 (makespan과 q_total_over_time)
                    selected_indices = ['makespan', 'q_total_over_time']

                #    rule_result_df = rule_result_df.reset_index().melt(id_vars=['index'])
                    # 데이터프레임을 Melt하여 필요한 형태로 변환
                    rule_result_df2 = rule_result_df2.reset_index().melt(id_vars=['index'])
                    # Plotly로 나란히 두 개의 막대 그래프 그리기
                    fig = px.bar(rule_result_df2, x='index', y='value', color='variable', barmode='group',
                                labels={'index': 'Data', 'value': 'Value', 'variable': 'Category'})

                    # 그래프에 레이아웃 설정
                    fig.update_layout(
                        xaxis=dict(tickvals=list(range(len(rule_result_df2['index']))), ticktext=rule_result_df2['index']),
                        xaxis_title='Data',
                        yaxis_title='Value',
                        showlegend=True
                    )

                    # 스트림릿에 그래프 표시
                    st.plotly_chart(fig)
                with st.expander("simul_extra_result"):
                    g_index = ['makespan','util','Flow_time','tardiness', 'lateness', 't_max','q_time_true','q_job_true','q_total_over_time']
                    tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9 =st.tabs(g_index)
                    tab_l = [tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9]
                    for i in range(len(tab_l)+1):
                        tab_l[i-1].subheader(g_index[i-1])
                        rule_extra_result = rule_result_df1[g_index[i-1]]
                        tab_l[i-1].bar_chart(rule_extra_result)
                    

                with st.expander("Result"):
                    tab1,tab2,tab3,tab4,tab5,tab6,tab7 =st.tabs(tab_list)
                    tab_l = [tab1,tab2,tab3,tab4,tab5,tab6,tab7]
                    for j in range(len(tab_l)):
                        # if j == 1:
                        #     tab1.subheader("m_on_job_number")
                        #     fig_d = go.Figure(data=fig_list_d[0][j-1])
                        #     tab1.plotly_chart(fig_d,use_container_width=True)
                        # else:
                        tab_l[j-1].subheader(tab_list[j-1])
                        fig_d = go.Figure(data=fig_list_d[0][j-1])
                        tab_l[j-1].plotly_chart(fig_d,use_container_width=True)

                            
            #aa = st.selectbox("rule_select",(columns_name))
            
            # 모든 그래프 이미지를 다운로드할 수 있는 버튼 추가
            # if st.button("모든 그래프 다운로드"):
            #     for i, fig in enumerate(fig_list_d):
            #         # 그래프를 이미지 파일로 저장
            #         image_file_path = os.path.join(output_dir, f"{tab_list[i]}.png")
            #         fig.write_image(image_file_path)

            #         # 이미지 파일을 다운로드할 수 있는 버튼 추가
            #         st.download_button(
            #             f"{tab_list[i]}.png",  # 이미지 파일 이름
            #             image_file_path,  # 이미지 파일 경로
            #             label=f"다운로드 {tab_list[i]}.png",
            #             key=f"download_button_{i}",
            #             mime="image/png",
            #         )
        # print("FlowTime:" , Flow_time)
        # print("machine_util:" , machine_util)
        # print("util:" , util)
        # print("makespan:" , makespan)
        # print("Score" , score)         
        
            # with st.expander("fig"):
            #     st.plotly_chart(fig)

            # with st.expander("fig2"):
            #     st.plotly_chart(fig2)

            # with st.expander("fig3"):
            #     st.plotly_chart(fig3)

            # with st.expander("fig4"):
            #     st.plotly_chart(fig4)

            # with st.expander("fig5"):
            #     st.plotly_chart(fig5)

            # with st.expander("fig6"):
            #     st.plotly_chart(fig6)

            # with st.expander("fig7"):
            #     st.plotly_chart(fig7)

            # with st.expander("fig8"):
            #     st.plotly_chart(fig8)
    # for i in range(0,len(columns_name)):
    #     with st.expander(f"{columns_name[i]}_graph"):
    #         st.plotly_chart(fig_dict[i])
    
    
    
    
    
    # with st.expander("graph"):
    #     st.image("fig8_png.png")
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