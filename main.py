"""Streamlit entrypoint for Packy travel packing helper."""

from modules.agent import PackyAgent
from modules.history import ChatHistoryManager

import streamlit as st


def init_state() -> None:
    """Initialize Streamlit session state for chat and agent instances."""
    if "history_manager" not in st.session_state:
        st.session_state.history_manager = ChatHistoryManager()
    if "agent" not in st.session_state:
        st.session_state.agent = PackyAgent()


def render_chat() -> None:
    """Render the chat interface and handle user interactions."""
    st.title("Packy: 여행 짐 싸기 도우미")
    for message in st.session_state.history_manager.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("어디로 여행 가시나요? 궁금한 점을 물어보세요."):
        st.session_state.history_manager.add_user_message(prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        response = st.session_state.agent.handle_input(
            prompt,
            history=st.session_state.history_manager.history,
            history_tuples=st.session_state.history_manager.as_tuples(),
        )
        st.session_state.history_manager.add_assistant_message(response)
        with st.chat_message("assistant"):
            st.markdown(response)


def main() -> None:
    """Main application runner for Streamlit."""
    init_state()
    render_chat()


if __name__ == "__main__":
    main()
