import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Home")


col1, col2 = st.columns([8,2])

with col1:
    st.title("Q-time 제약이 있는 스케줄링 문제 생성 및 시뮬레이터")

with col2:
    st.image("tuk_img.png")

x2 = st.expander('사용법')
x2.write('''
- Queue time(Q-time)은 작업이 어떠한 공정을 거친 후 다음 공정을 수행하기까지의 마감기한으로 반도체 wafer공정 및 OLED공정에 존재하는 제약 조건임
- 본 연구에서는 Q-time제약이 존재하는 스케줄링 문제를 해결하기 위한 심층강화학습 기반의 알고리즘을 개발하고자 함
- 각 디스패칭 룰을 적용하거나 DDQN 알고리즘을 이용하여 의사결정 시점에 적절한 작업을 투입해, Q-time제약으로 인한 손실을 줄이고자 함
- 이 웹은 Streamlit을 통해서 Queue time을 포함한 공정에 대한 문제생성기와 시뮬레이터를 포함
- 시뮬레이터를 통해 성능평가표와 간트차트를 확인 가능

''')
st.markdown("---")
# x1 = st.expander('진행상황')
# x1.write('''
# - data set에 들어가서 데이터 생성가능
# - 원래있던 파일 지정해서 적용하는건 이전것을 자꾸 가져와서 일단 뺐음
# - copy는 테스트용
# - simul 돌려서 데이터 프레임 출력까지 적용
# - rd는 인덱스에 j 않붙어있는거 수정
# - 세부 코드는 data_set_code에서 확인 가능
# - 기본적으로 데이터 파일 하나씩 만들어져 있어서 sim.csv생성 안하고 하면 이미 생성되어있는 것으로 적용
# ''')
# st.markdown("---")



with st.sidebar:
    st.write("목록")