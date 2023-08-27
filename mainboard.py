import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Home")


col1, col2 = st.columns([8,2])

with col1:
    st.title("FJSP_DEMO")

with col2:
    st.image("tuk_img.png")


x1 = st.expander('진행상황')
x1.write('''
- data set에 들어가서 데이터 생성가능
- 원래있던 파일 지정해서 적용하는건 이전것을 자꾸 가져와서 일단 뺐음
- copy는 테스트용
- simul 돌려서 데이터 프레임 출력까지 적용
- rd는 인덱스에 j 않붙어있는거 수정
- 세부 코드는 data_set_code에서 확인 가능
- 기본적으로 데이터 파일 하나씩 만들어져 있어서 sim.csv생성 안하고 하면 이미 생성되어있는 것으로 적용
''')
st.markdown("---")



with st.sidebar:
    st.write("목록")