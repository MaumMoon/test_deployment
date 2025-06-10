import streamlit as st
import time
from agent import response_from_llm
from streamlit.runtime.scriptrunner import get_script_run_ctx

# model 모듈 사용하여 응답 받아오는 부분
# Streamlit 응답 타이핑 시각화
def response_generator(user_text):
    session_id = get_script_run_ctx().session_id
    response = response_from_llm(user_text, session_id)
    for line in response.split('\n'):
        for word in line.split():
            yield word + " "
            time.sleep(0.05)
        yield "  \n"

# Streamlit Page
st.set_page_config(
    page_title="마음AI 챗봇",
    page_icon="💕",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.image("./data/maum_ai_logo.png", width=100)
st.subheader("마음AI '상담 Agent'입니다😊💕")
st.caption("편한 마음으로 상품에 대해 문의해 주세요!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            with st.status("답변을 확인하세요.", expanded=False, state="complete"):
                st.write(message["content"])

if prompt := st.chat_input("어떤 것이 궁금하세요?"):
    st.session_state.messages.append({"role": "user", "avatar":"❓", "content": prompt})
    with st.chat_message(name="user", avatar="❓"):
        st.markdown(prompt)

    with st.chat_message(name="assistant", avatar="😊"):
        with st.status("응답 중입니다.", expanded=True) as status:
            response = st.write_stream(response_generator(prompt))
            status.update(
                label="답변을 확인하세요.", state="complete", expanded=True
            )   
        
    st.session_state.messages.append({"role": "assistant", "avatar":"😊", "content": response})