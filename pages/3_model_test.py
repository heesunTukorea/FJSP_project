import streamlit as st
import os
import pandas as pd
from io import BytesIO
import time
from param_DQN import * 
import plotly.graph_objs as go
from fjsp_Q import get_csv_file_list,get_csv_files_with_string,highlight_max,draw_histogram
import os

# import seaborn as sns
# import matplotlib.pyplot as plt

col1, col2 = st.columns([7,1])

with col2:
    st.image("tuk_img.png")
st.title("모델 테스트")


x1 = st.expander('사용법')
x1.write('''
- 이전에 강화학습을 통해 생성돤 nomorspt.pt 파일을 이용하여 시뮬레이션 진행
- 성능지표와 간트차트를 이용한 스케줄링 결과 확인 가능
- 필요한 파일로는 Processing_time,Setup_time,Q-time,Rd_time,nomorspt.pt가 필요 

1. 각각에 알맞은 데이터를 파일을 업로드해준다
2. 업로드한 데이터를 확인 가능
3. selectbox를 통해서 각 룰을 적용 또느 강화학습을 적용 할 것인지 선택 가능
4. 결과를 데이터프레임의 형태로 띄워줌
5. 간단한 시각화와 가장 최근에 돌린 디스패칭룰 시뮬레이터와 비교 가능
6. 간트차트를 이용한 시각화 확인 가능 
''')
st.markdown("---")

st.subheader("데이터 업로드")
save_folder = 'fjsp_stream'


# 이전 업로드 파일 이름을 저장하기 위한 변수
prev_uploaded_filename = None

#csv_list = get_csv_file_list(save_folder)
#upload_csv = st.selectbox("CSV 파일 선택", csv_list)

# 여러 파일 업로더
# uploaded_files = st.file_uploader("CSV 파일을 선택하세요.", accept_multiple_files=True)

# 업로드한 파일들을 저장할 딕셔너리
uploaded_files_dict = {}
pt_dict ={}
# 업로드한 파일을 딕셔너리에 추가하고 저장
    # 여러 파일 업로더
uploaded_files_p = st.file_uploader("Processing 파일을 선택하세요.")
uploaded_files_s = st.file_uploader("Set up 파일을 선택하세요.")
uploaded_files_q = st.file_uploader("Q-time 파일을 선택하세요.")
uploaded_files_r = st.file_uploader("Rd_time 파일을 선택하세요.")
uploaded_files_pt = st.file_uploader("Pt 파일을 선택하세요.")

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

if uploaded_files_pt is not None:
    bytes_data_pt = uploaded_files_pt.read()
    pt_names = uploaded_files_pt.name
    pt_dict[pt_names] = bytes_data_pt
    save_folder = 'fjsp_stream'  # 저장할 폴더명
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    file_path = os.path.join(save_folder, pt_names)
    with open(file_path, "wb") as f:
        f.write(bytes_data_pt)  # 저장할 데이터 사용
    st.success(f"{pt_names} 파일이 업로드되었고 저장되었습니다.")
    
uploaded_file_names=list(uploaded_files_dict.keys())
#tabs = st.selectbox("데이터프레임 선택", list(uploaded_files_dict.keys()))
# data_dict = {
#     "p_time_data" : f'{save_folder}/{uploaded_file_names[0]}',
#     "setup_data" :f'{save_folder}/{uploaded_file_names[1]}',
#     "Q_data" : f'{save_folder}/{uploaded_file_names[2]}',
#     "rd_data" :f'{save_folder}/{uploaded_file_names[3]}'
#     }
if len(uploaded_file_names) >= 4:
    with st.expander("데이터프레임"):
        tab_1,tab_2,tab_3,tab4 = st.tabs(uploaded_file_names)
        t_list = [tab_1,tab_2,tab_3,tab4]
        for i in range(len(t_list)):
            t_list[i-1].subheader(uploaded_file_names[i-1])
            selected_df = uploaded_files_dict[uploaded_file_names[i-1]]
            t_list[i-1].write(selected_df)

if len(uploaded_file_names) >= 4:
    st.subheader('인공신경망 모델을 통한 시뮬레이션') 
        # 디렉토리를 생성하여 그래프 이미지를 저장합니다.
        # output_dir = 'graph_images'
        # os.makedirs(output_dir, exist_ok=True)

        

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
    sim_file_name = f"{save_folder}/{uploaded_file_names[0]}"
    setup_file_name = f"{save_folder}/{uploaded_file_names[1]}"
    q_time_file_name = f"{save_folder}/{uploaded_file_names[2]}"
    rddata_file_name = f"{save_folder}/{uploaded_file_names[3]}"

    st.write('레이어 조정')
    col1,col2 = st.columns([1,1])
    st.markdown("---")
    

    with col1:
        input_layer = st.number_input('input_layer',value = 36 , min_value=0, max_value=100000, step=1)
    with col2:
        output_layer = st.number_input('output_layer',value = 10 , min_value=0, max_value=100000, step=1)
    

    r_param = {
                "gamma": 0,
                "learning_rate": 0,
                "batch_size": 0,
                "buffer_limit": 0,
                "input_layer" : input_layer,
                "output_layer" : output_layer,
                "episode" : 0
            }
    param = {
            "p_data" : sim_file_name,
            "s_data" : setup_file_name,
            "q_data" : q_time_file_name,
            "rd_data" : rddata_file_name
        }
    simul_file_name_d = st.text_input("파일 이름을 입력하세요 (확장자 없이):", "FJSP_simul")
    if st.button('클릭'):
        sim_file_name = f"{save_folder}/{uploaded_file_names[0]}"
        setup_file_name = f"{save_folder}/{uploaded_file_names[1]}"
        q_time_file_name = f"{save_folder}/{uploaded_file_names[2]}"
        rddata_file_name = f"{save_folder}/{uploaded_file_names[3]}"
        pt_file_name= f"{save_folder}/{pt_names}"
        for i in range(1):
            param_d = Parameters(sim_file_name, setup_file_name, q_time_file_name, rddata_file_name,r_param)
            fig,fig2,fig3,fig4,fig5,fig6,fig8,Flow_time, machine_util, util, makespan, Tardiness_time, Lateness_time, T_max,q_time_true,q_time_false,q_job_t, q_job_f, q_time =dqn_params(pt_file_name,param_d,r_param)
            #fig,fig2,fig3,fig4,fig5,fig6,fig7,fig8 = main_d.gannt_chart()

            #리스트 저장
            makespan_table_d.append(makespan)
            util_d.append(util)
            ft_table_d.append(Flow_time)
            machine_util_list_d.append(machine_util)
            tardiness_list_d.append(Tardiness_time)
            lateness_list_d.append(Lateness_time)
            t_max_list_d.append(T_max)
            q_time_true_list_d.append(q_time_true)
            q_time_false_list_d.append(q_time_false)
            q_job_t_list_d.append(q_job_t)
            q_job_f_list_d.append(q_job_f)
            q_time_list_d.append(q_time)

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
            rule_result_df.to_csv(f'{simul_file_name_d}.csv', index=True, header=True)
            with open(f'{simul_file_name_d}.csv') as f:
                st.download_button(f"Download {simul_file_name_d}.csv", f, file_name=f"{simul_file_name_d}.csv", mime='text/csv')
            with st.expander("color"):
                    st.write(styled_df)
            
            
            
            
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
            
            #탭에 그래프들을 저장 출력
            tab_list =['m_on_job_number','machine_gantt','DSP_gantt','step_DSP_gantt','heatMap_gantt','main_gantt','job_gantt_for_Q_time']
            #aa = st.selectbox("rule_select",(columns_name))
            # for index,i in enumerate(columns_name):
            with st.expander("Result"):
                tab1,tab2,tab3,tab4,tab5,tab6,tab7 =st.tabs(tab_list)
                tab_l = [tab1,tab2,tab3,tab4,tab5,tab6,tab7]
                for j in range(len(tab_l)):
                    tab_l[j-1].subheader(tab_list[j-1])
                    fig_d = go.Figure(data=fig_list_d[0][j-1])
                    tab_l[j-1].plotly_chart(fig_d,use_container_width=True)