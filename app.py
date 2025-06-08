import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    split_docs = text_splitter.split_documents(docs)

    vector_store = QdrantVectorStore.from_documents(
        documents=split_docs,
        url="http://localhost:6333",
        collection_name="learning_vectors",
        embedding=embedding_model
    )
    print("Embedding done")
    return vector_store

INITIAL_SYSTEM_PROMPT = """
You are a helpful AI Assistant who answers user queries based on the available context
retrieved from a PDF file along with page_contents and page number.

You should only answer the user based on the following context and also provide source of the page to the user from where the user can go check it themselves.
Also if the users says they dont understand a certain topic from the web page, add your own answer explaining it in simple terms to the user.
"""

st.set_page_config(page_title="Chat with Website", page_icon="🔗")
st.title("QUICK DOCS")

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")
    st.info("Just enter the website url and ask the chatbot anything you didn't understand or want to know")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content=INITIAL_SYSTEM_PROMPT),
        AIMessage(content="Hello! I am an AI Website chat bot. How can I help you?")
    ]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "current_url" not in st.session_state:
    st.session_state.current_url = ""

def load_vector_store_if_needed(url):
    if url and url != st.session_state.current_url:
        st.session_state.vector_store = get_vectorstore_from_url(url)
        st.session_state.current_url = url
        st.session_state.chat_history = [
            SystemMessage(content=INITIAL_SYSTEM_PROMPT),
            AIMessage(content="Hello! I am an AI Website chat bot. How can I help you?")
        ]

if not website_url:
    st.info("Please enter a website URL first to chat")
else:
    with st.spinner("Loading and embedding website content... This might take a while on first load."):
        load_vector_store_if_needed(website_url)

    if st.session_state.vector_store is None:
        st.error("Failed to load vector store from URL.")
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        user_query = st.chat_input("Enter your text here")

        if user_query:
            search_results = st.session_state.vector_store.similarity_search(query=user_query)

            current_turn_context = "\n\n\n".join(
                [
                    f"Page Content: {result.page_content}\nSource Location: {result.metadata.get('source', 'N/A')}"
                    for result in search_results
                ]
            )

            messages_for_current_turn = [
                SystemMessage(content=INITIAL_SYSTEM_PROMPT + "\n\nContext:\n" + current_turn_context)
            ]
            messages_for_current_turn.extend(st.session_state.chat_history[1:])
            messages_for_current_turn.append(HumanMessage(content=user_query))

            chat_completion = llm.invoke(messages_for_current_turn)
            ai_response_content = chat_completion.content

            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=ai_response_content))

        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.write(message.content)
