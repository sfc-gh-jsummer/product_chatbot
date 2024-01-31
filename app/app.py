from __future__ import annotations

from typing import Any, Dict, List, Optional, cast
from uuid import uuid4
import os

from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.docstore.document import Document
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import load_prompt

import streamlit as st
from streamlit.web.server.websocket_headers import _get_websocket_headers
from snowflake.snowpark import Session
from snowflake.snowpark.functions import current_timestamp
import pandas as pd
import weaviate
import openai
from uuid import uuid1

def get_properties(client) -> list[str]:
    """Returns all properties of first weaviate class"""
    properties = client.schema.get()['classes'][0]['properties']
    names = [i.get('name') for i in properties]
    return names

def prep_dns(text):
    """Makes DNS/API URL all lowercase and replaces _ with -."""
    return text.lower().replace("_", "-")

class MyCallbackHandler(BaseCallbackHandler):
    
    def __init__(self):
        self.message = ''
    
    def on_llm_new_token(self, token, **kwargs) -> None:
        # print every token on a new line
        self.message += token.replace(' -','*')
        st.write(self.message)
        
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        self.message = ''
        
@st.cache_resource
def initiate_snowpark_conn():
    with open("/snowflake/session/token", "r") as f:
        token = f.read()

    connection_parameters = {
        'host': os.environ['SNOWFLAKE_HOST'],
        'port': os.environ['SNOWFLAKE_PORT'],
        'protocol': 'https',
        'account': os.environ['SNOWFLAKE_ACCOUNT'],
        'authenticator': 'oauth',
        'token': token,
        'role': os.environ['SNOW_ROLE'],
        'warehouse': os.environ['SNOW_WAREHOUSE'],
        'database': os.environ["SNOW_DATABASE"],
        'schema': os.environ["SNOW_SCHEMA"],
        'client_session_keep_alive': True
    }
    snowpark_session = Session.builder.configs(connection_parameters).create()
    return snowpark_session

session = initiate_snowpark_conn()

if "run_id" not in st.session_state:
    st.session_state.run_id = str(uuid1())
run_id = st.session_state.run_id
log_table = os.environ["CHAT_LOG_TABLE"]

client = weaviate.Client(
    url = os.environ['WEAVIATE_URL'],
)

props = get_properties(client)
index_name = os.environ['INDEX_NAME'] # Weaviate class name
text_key = os.environ['TEXT_KEY'] # Property in weaviate to treat as page content

props = get_properties(client)
myretriever = WeaviateHybridSearchRetriever(
    client = client,
    index_name = index_name,
    text_key = text_key, # page content
    attributes = props, # what to return
    properties = props, # limits the set of properties that will be searched by the BM25 component of the search
    alpha = .8, # How much to weigh keyword search (0) vs. vector search (1)
    k = 10 # Number of documents to return
)

if "chain" not in st.session_state:
    model = os.environ['MODEL_NAME']
    api_base = prep_dns(f'http://api.{os.environ["SNOW_SCHEMA"]}.{os.environ["SNOW_DATABASE"]}.snowflakecomputing.internal:8000/v1')

    llm = ChatOpenAI(
        model=model,
        openai_api_key="EMPTY",
        openai_api_base=api_base,
        max_tokens=600,
        temperature=0.5,
        callbacks=[MyCallbackHandler()],
        streaming = True
    )

    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, output_key='answer', k=5)

    # Create the chat prompt templates
    system_message_prompt = SystemMessagePromptTemplate(prompt=load_prompt('system_prompt.yaml'))
    messages = [
        system_message_prompt,
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    chain = ConversationalRetrievalChain.from_llm(llm, 
                                                  myretriever,
                                                  return_source_documents=True, 
                                                  memory = memory,
                                                  combine_docs_chain_kwargs={"prompt": qa_prompt},
                                                  get_chat_history=lambda h : h)

    st.session_state.chain = chain

chain = st.session_state.chain
user = _get_websocket_headers()["Sf-Context-Current-User"]
st.header('☃️ Product Catalog', divider='gray')
st.subheader(f'Welcome {user.lower()}')
    

def conversational_chat(query):
    container = st.empty()
    with container:
        result = chain({"question": query})
    st.session_state['results'].append(result)
    st.session_state['history'].append((query, result['answer']))
    st.session_state['past'].append(query)
    st.session_state['generated'].append(result['answer'])
    return result

    # Initialize chat history
if 'results' not in st.session_state:
    st.session_state['results'] = []

if 'history' not in st.session_state:
    st.session_state['history'] = []

# Initialize messages
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

# Create containers for chat history and user input
response_container = st.container()
container = st.container()

props = get_properties(client)

if len(st.session_state['generated']) > 0:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            ai,blank,user = st.columns((2,6,2))
            user_messages = st.chat_message("user")
            ai_messages = st.chat_message("ai")
            with user:
                user_messages.write(st.session_state["past"][i])
            with ai:
                ai_messages.write(st.session_state["generated"][i])
else:
    ai_messages = st.chat_message("ai")
    ai_messages.write("Hello! How can I help you?")
                
if user_input := st.chat_input("How can I help you?"):
    ai,blank,user = st.columns((2,6,2))
    user_messages = st.chat_message("user")
    ai_messages = st.chat_message("ai")
    with user_messages:
        st.write(user_input)
    with ai_messages:
        answer = conversational_chat(user_input)
        # Below block writes source_documents to dataframe in chat
        # products = []
        # for product in answer['source_documents']:
        #     d = {}
        #     for k, v in product.metadata.items():
        #         if k in props:
        #             d[k] = v
        #     products.append(pd.DataFrame([d]))
        # ai_messages.write(pd.concat(products))
        
    timestamp = session.create_dataframe([1]).select(current_timestamp()).collect()[0][0]
    log = pd.DataFrame([(run_id,timestamp,user_input,answer['answer'],[str(i) for i in answer['source_documents']])],
                                   columns = ["RUN_ID","TIMESTAMP","USER_PROMPT","ASSISTANT_RESPONSE","SOURCE_DOCUMENTS"])
    log_write = session.write_pandas(log,log_table)
