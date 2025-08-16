import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper ## Wrapper on top of query
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun,DuckDuckGoSearchRun ## Run the query in document/wiki
## DuckDuckGoSearchRun -> DuckDuckGoSearchRun class/tool, which can then be used by a LangChain agent to get real-time search results from the internet.
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler ##  This is used to display real-time agent activity inside a Streamlit app — a popular Python library for building web apps
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
## Load Groq API Key
## Groq API key use to call open source llm model
import openai
groq_api_key=os.getenv("GROQ_API_KEY")
print(groq_api_key)

api_wrapper_wiki=WikipediaAPIWrapper(top_k_result=1, doc_content_chars_max=250) ## top_k_result=1-> return only the top 1 most relevant result
                                                                                ## doc_content_chars_max -> length of the returned content
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

## Used the in-build tool of Arxiv, create wrapper and then tool on top of wrapper
api_wrapper_arxiv=ArxivAPIWrapper(top_k_result=1, doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search_in_web=DuckDuckGoSearchRun(name="Search")

st.title("Langchain - Chat with search")
"""
In this example, we are using StreamlitCallbackHandler to display the throughts and actions of an agent in
an interactive Streamlit app. Try more LangChain Streamlit Agent example at https://github.com/langchain-ai/streamlit-agent 
"""

st.sidebar.title("Settings")
## Create a sidebar text field in streamlit gui interface, put Open API key
api_key=st.sidebar.text_input("Enter your Groq API key: ", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {
            "role":"assistant","content":"Hi, I am a chatbot, who can search in web. How can I help you?"
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user", "content":prompt}) ## Append the asked question into session state
    st.chat_message("user").write(prompt) ## Write asked question in gui

    llm=ChatGroq(groq_api_key=api_key,model="Llama3-8b-8192", streaming=True) ## Model Llama3-8b-8192, streaming=True -> allowing you to render responses in real time
    tools=[search_in_web,arxiv,wiki]

    ##Transform tools to agent
    search_agent=initialize_agent(tools,llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
    ## llm query in tools to answer the asked question
    ## AgentType.ZERO_SHOT_REACT_DESCRIPTION -> No chat history consider here.
    ## AgentType.CONVERSATIONAL_REACT_DESCRIPTION -> Consider chat history.
    ## handle_parsing_errors=True -> Error handling

    ## st.chat_message("assistant") -> syambol is chat conversation is different, like robot assistent
    ## st.chat_message("user")      -> syambol is chat conversation is different, like user 

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=True) ## continous prompt to different tools to get response
        response=search_agent.run(st.session_state.messages, callbacks=[st_cb]) ## use callback in agent to continous query to diffent tools
        st.session_state.messages.append({"role":"assistant","content":response}) ## store the response as role assistant
        st.write(response)




