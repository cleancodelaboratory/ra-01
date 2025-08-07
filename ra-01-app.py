from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
import streamlit as st

def simple_rag(api_key:str, question: str) -> str:
    
    reader = PdfReader("diary.pdf")
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = FAISS.from_documents(documents, embeddings)
    llm = ChatOpenAI(model_name="gpt-4o", api_key=api_key)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    return qa.invoke({"query":question})["result"]

st.header("	📖 철수 일기 내용 물어보기")
api_key = st.text_input("🔑 OPENAI API KEY를 입력하세요.", type="password")
question = st.text_input("💬 질문을 입력하세요.")

if st.button("✅ 답변 확인"):

    if not api_key or not question:
        st.warning("API 키와 질문을 모두 입력해주세요.")
        
    else:
        with st.spinner("일기 내용을 검토해 답변을 생성 중입니다..."):
            answer = simple_rag(api_key=api_key, question=question)
            st.markdown(answer)