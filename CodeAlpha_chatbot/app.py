import streamlit as st

from src.chatbot import get_response

st.set_page_config(
    page_title="FAQ Assistant",
    page_icon="🤖"
)

st.title("AI FAQ Assistant")

query = st.text_input(
    "Ask a question"
)

if query:

    result = get_response(query)

    st.success(
        result["answer"]
    )

    st.metric(
        "Confidence",
        result["confidence"]
    )

    st.subheader(
        "Similar Questions"
    )

    for s in result["suggestions"]:
        st.write("•", s)
