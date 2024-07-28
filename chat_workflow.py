import pprint

import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
# from langchain.chat_models import ChatOpenAI
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
import os
from custom_embeddings import CustomAPIEmbeddings


# @st.cache_resource
def chain_workflow(openai_api_key):
    
    #llm 
    # llm_name = "gpt-3.5-turbo"
    llm_name = "internlm/internlm2_5-7b-chat-gguf"

    # persist_directory
    persist_directory = 'vector_index/'	


    print("before embeddings definition ........ ")

    # # Load OpenAI embedding model
    # embeddings = OpenAIEmbeddings(
    #         model="Xenova/text-embedding-ada-002",
    #         api_key="lm-studio",
    #         base_url="http://192.168.0.108:1234/v1/",
    #         # openai_api_base="http://192.168.0.108:1234/v1/",
    #         # openai_api_version="v1",
    #         # openai_api_key=openai_api_key
    #     )

    embeddings = CustomAPIEmbeddings(
        model_name="Xenova/text-embedding-ada-002",
        api_url="http://192.168.0.108:1234/v1/embeddings",
        # api_key="sss"
    )

    print("before embeddings ........ ")

    text = "This is a test query."
    query_result = embeddings.embed_query(text)
    print("after embeddings ........ ")
    print(f'{query_result}')


    # Check if the file exists
    if not os.path.exists("vector_index/chroma.sqlite3"):
        print("Vectorstore creation beginning........")

        # If it doesn't exist, create it

        # load document
        file = "mydocument/animalsinresearch.pdf"
        loader = PyPDFLoader(file)
        documents = loader.load()

        print("Vectorstore creation beginning........ after load ......")

        # split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(documents)

        pprint.pprint(splits)

        print("Vectorstore creation beginning........ after split_documents ......")

        # persist_directory
        persist_directory = 'vector_index/'

        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory
        )

        print("Vectorstore creation beginning........ after vectordb ......")

        vectordb.persist()
        print("Vectorstore created and saved successfully, The 'chroma.sqlite3' file has been created.")
    else:
        # if vectorstore already exist, just call it
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    
    # Load OpenAI chat model
    llm = ChatOpenAI(temperature=0, 
                     openai_api_key=openai_api_key,
                     model="internlm/internlm2_5-7b-chat-gguf",
                     api_key="lm-studio",
                     base_url="http://192.168.0.108:1234/v1/",
                     )
    
    # specify a retrieval to retrieve relevant splits or documents
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,base_retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3}))

    
    # Create memory 'chat_history' 
    memory = ConversationBufferWindowMemory(k=3,memory_key="chat_history")
    
    # create a chatbot chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type="map_reduce", 
        retriever=compression_retriever, 
        memory=memory,
        get_chat_history=lambda h : h,
        verbose=True
    )
    
    
    return qa


if __name__ == '__main__':
    chain_workflow('lm-studio')

    '''
    curl http://192.168.0.108:1234/v1/embeddings \
      -H "Content-Type: application/json" \
      -d '{
        "input": "Your text string goes here",
        "model": "Xenova/text-embedding-ada-002"
      }'
    '''


