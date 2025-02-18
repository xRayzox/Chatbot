import streamlit as st
from transformers import pipeline

st.title("Hugging Face Chatbot")

# Initialize pipeline
@st.cache_resource
def load_pipeline():
    pipe = pipeline("text-generation", model="MaziyarPanahi/calme-3.2-instruct-78b")
    return pipe

pipe = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I am a chatbot. How can I help you?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        for output in pipe(messages, max_new_tokens=50, return_full_text=False):
            full_response += output[0]['generated_text']
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})