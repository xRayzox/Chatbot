import streamlit as st
from transformers import pipeline
import torch

st.title("Optimized Hugging Face Chatbot")

# Load model with caching and GPU acceleration
@st.cache_resource
def load_pipeline():
    return pipeline("text-generation", model="MaziyarPanahi/calme-3.2-instruct-78b", 
                    torch_dtype=torch.float16, device=0)

pipe = load_pipeline()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I am a chatbot. How can I help you?"}]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-5:]]  # Keep only the last 5 messages
        
        for output in pipe(messages, max_new_tokens=30, return_full_text=False, stream=True):
            full_response += output[0]['generated_text']
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
