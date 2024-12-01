import streamlit as st
import os
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
import jwt
import pickle
from langchain.schema import Document, SystemMessage, HumanMessage
import base64
import os
# Initialize OpenAI API key
os.environ["OPENAI_API_KEY"] = "enter your key"


# PDF file paths
adglobal360_pdf_paths = [
    r"C:\Users\Madhur_Gauri\Desktop\koala_two\AGL_HR _Policy.pdf",
    r"C:\Users\Madhur_Gauri\Desktop\koala_two\AGL Policyy.pdf",
    r"C:\Users\Madhur_Gauri\Desktop\koala_two\AGL_suggestiion_xl.pdf",
    r"C:\Users\Madhur_Gauri\Desktop\koala_two\Product Manual _ Enviro.H (Mobile).pdf",
    r"C:\Users\Madhur_Gauri\Desktop\koala_two\Employees_check.pdf"
]

hakuhodo_pdf_paths = [
    r"C:\Users\Madhur_Gauri\Desktop\koala_two\Hakuhodo HR and Administration Policy.pdf"
]


# Initialize embeddings
agl_embedding = OpenAIEmbeddings()
hakuhodo_embedding = OpenAIEmbeddings()

# Text splitter for document chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)


def load_documents_for_domain(company_id):
    data = []
    if company_id == 1:
        loaders = [PyMuPDFLoader(pdf_path) for pdf_path in adglobal360_pdf_paths]
    elif company_id == 2:
        loaders = [PyMuPDFLoader(pdf_path) for pdf_path in hakuhodo_pdf_paths]
    else:
        return []
    
    for loader in loaders:
        docs = loader.load()
        data.extend([Document(page_content=doc.page_content) for doc in docs])
    return data


def get_vector_db(company_id, docs, embedding, vector_cache_path):
    docs = text_splitter.split_documents(docs)
    if os.path.exists(vector_cache_path):
        with open(vector_cache_path, "rb") as f:
            vector_db = pickle.load(f)
    else:
        vector_db = FAISS.from_documents(docs, embedding=embedding)
        with open(vector_cache_path, "wb") as f:
            pickle.dump(vector_db, f)
    return vector_db


def authenticate_user(jwt_token):
    """Authenticate user based on JWT token."""
    try:
        decoded_jwt = jwt.decode(jwt_token, options={"verify_signature": False})
        company_id = decoded_jwt.get("company_id")
        name = decoded_jwt.get("name", "User")
        if company_id:
            return company_id, name
    except jwt.ExpiredSignatureError:
        return None, None
    except jwt.InvalidTokenError:
        return None, None
    return None, None


def get_chatbot_response(company_id, question):
    # Load documents
    loaded_data = load_documents_for_domain(company_id)
    if not loaded_data:
        return "No relevant documents found for the given company ID."

    docs = text_splitter.split_documents(loaded_data)

    vector_db = get_vector_db(
        company_id,
        docs,
        agl_embedding if company_id == 1 else hakuhodo_embedding,
        "agl_vector_db.pkl" if company_id == 1 else "hakuhodo_vector_db.pkl",
    )

    retriever = vector_db.as_retriever()
    retrieved_docs = retriever.get_relevant_documents(question)
    retrieved_content = "\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
    - Only use information from the following documents:
    {retrieved_content}

    New Context:
    - Only use information from {loaded_data}.
    - Follow these specific rules:
      1. If company ID is 1, provide answers from Adglobal360 documents.
      2. If company ID is 1, and asks about other organizations or Hakuhodo, respond with "Information can't be provided under data privacy violation."
      3. If company ID is 2, provide answers from Hakuhodo documents.
      4. If company ID is 2, and asks about other organizations or AGL or agl or Adglobal360, respond with "Information can't be provided under data privacy violation."
      5. If a query involves privacy-sensitive terms, respond with "Information can't be provided under data privacy violation."
      6. If the provided information is in long paragraph, provide it in points or in tabular format for better readability.
      7. If no relevant document is found, explain why.
      8. Don't mix response from internet.
      9. If a user ask what is your name to model ,respond with Hi ,I'm KOLA, your all-in-one workplace assistant. Let me know how I can help you today.
         Remember to tailor the response to the specific context and capabilities of your HRMS system.
      11. Always give email or numbers mentioned in the document.
      12.when someone says Hey or hey  or HEY or hi I am {name} , the respond should be either Hi {name}  nice to meet you! What can I do for you?.
      13.when someone says  hi or Hi or HI  i am {name} , the response should be Good day, {name}. How may I assist you?.
      14.when someone says  hello i am {name}, the response should be Greetings {name},Please let me know if you have any questions or requests.
      15.when someone says hey whatsup?, then response should be Yo {name}, how's it going? Let me know if you need anything.
      16. when some says hi , hello , hey , repond with Hi ,hey ,hello ,how are you today, dont mention user anywhare in sentence.
      17. when someone enters some gibberish respond with please enter your question again but someone misspels some word it should understand by itself.
      18. if soneone says hi koala,or anything except kola,KOLA,Kola then respond with my name is Kola.
      19. please save the context of conversation eg: if someone asks about anything and in the next sentence the user refers the same thing by it , them or something the response should be in
    - Always be polite and professional.


    Context Instructions:
    1. This conversation is continuous. Use the context from prior questions and answers for seamless responses.
    2. When the user references something indirectly (e.g., "it," "them," or similar pronouns), infer the meaning based on previous exchanges.
    3. If a user asks something new but refers to prior context (e.g., "Tell me more about that policy"), include the relevant prior information in the response.

    You are a conversational AI assistant. Your primary goal is to assist users by maintaining the context of the conversation. Save the details of previous questions and responses to ensure continuity and relevance in your answers. If the user refers to a previous topic using pronouns like 'it,' 'them,' or 'that,' interpret these references based on the context of prior interactions. Ensure that each response builds upon the user’s past queries for a seamless and coherent conversational experience.

    Question: {question}
    """
    
    system_message = SystemMessage(content=prompt)
    user_message = HumanMessage(content=question)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=200)
    response = llm([system_message, user_message])

    return response.content



LOGO_IMAGE = r"C:\Users\Madhur_Gauri\Desktop\koala_two\koala.jfif"

st.markdown(
    """
    <style>
    .container {
        display: flex;
    }
    .logo-text {
        font-weight:700 !important;
        font-size:50px !important;
        color: #f9a01b !
            </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="container">
        <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
        <p class="logo-text"> Hello, I am Kola !</p>
    </div>
    """,
    unsafe_allow_html=True
)



# Streamlit App
st.title("")
st.sidebar.header("Authentication")

# Input for JWT Token
jwt_token = st.sidebar.text_input("Enter your JWT Token")

if jwt_token:
    company_id, name = authenticate_user(jwt_token)

    if company_id:
        st.sidebar.success(f"Welcome, {name}!")
        st.write(" ")

        # Ask questions to the chatbot
        question = st.text_input("Ask a question:")
        if question:
            with st.spinner("Fetching response..."):
                response = get_chatbot_response(company_id, question)
            st.write(f"**Response:**\n{response}")
    else:
        st.sidebar.error("Invalid or expired token. Please try again.")
else:
    st.sidebar.info("Enter your JWT token to continue.")
