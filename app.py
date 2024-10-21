import streamlit as  st 
import os
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langchain.agents import initialize_agent,AgentType
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
load_dotenv()

# Arxiv and Wikipedia Tools 
arxiv_wrapper=ArxivAPIWrapper(top_k_result=1,doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper=WikipediaAPIWrapper(top_k_result=1,doc_content_chars_max=250)
wiki=WikipediaQueryRun(api_wrapper=wiki_wrapper)

search=DuckDuckGoSearchRun(name="Search")


## Title
st.title("LangChain-->Search Chat Engine")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
API_KEY:-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
"""


## sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter the Groq API KEY: ",type="password")


if "messages" not in st.session_state:
    st.session_state["messages"]=[{"role":"assistant","content":"Hey,I'm ChatBot who can search the web. How I can help You?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:=st.chat_input(placeholder=""):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True)
    tools=[search,arxiv,wiki]

    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_error=True)


    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)