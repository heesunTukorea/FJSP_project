import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Home")


col1, col2 = st.columns([8,2])

with col1:
    st.title("FJSP_DEMO")

with col2:
    st.image("tuk_img.png")






with st.sidebar:
    st.write("목록")