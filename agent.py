from dotenv import load_dotenv
import os
import pandas as pd
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain


load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    # max_output_tokens=200,
    google_api_key=GEMINI_API_KEY
)

df = pd.read_excel("./maum_ai_product_sample.xlsx")
loader = DataFrameLoader(df, page_content_column="설명")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07", google_api_key=GEMINI_API_KEY)
vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

contextualize_q_system_prompt = """
    Given a chat history and the latest user question which might reference context in the chat history,
    formulate a standalone question which can be understood without the chat history.
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

class ResponseGenerator():
    def __init__(self):
        self.template = ""
        self.ragchain = self.rag_chain_generator()
        self.store = {}

    def template_generator(self):
        template = """
                당신은 이 회사의 상품을 추천하는 임부를 부여받았습니다.
                고객들은 당신에게 AI가 필요한 특정 상황을 설명할 것입니다.
                당신은 고객 문의에 대해 단계별로 차근차근 친절하게 설명해 주세요.
                없는 정보는 답하면 안 됩니다.
                반드시 회사 상품 하나를 추천해야 합니다.
                더 궁금한 내용이 있으면 'https://maum.ai/' 사이트에 문의 메시지를 남겨달라는 멘트를 남겨 주세요.
                끝에는 언제나 '감사합니다!'라는 멘트를 붙여주세요.
                \n\n
                {context}
                질문: {input}
                You MUST answer in Korean:
                """
        return template
    
    def prompt_generator(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.template_generator()),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        return prompt
    
    def rag_chain_generator(self):
        question_answer_chain = create_stuff_documents_chain(
            llm, self.prompt_generator()
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        return RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
            )
    
    def get_session_history(self, session_ids):
        if session_ids not in self.store:  
            self.store[session_ids] = ChatMessageHistory()
        return self.store[session_ids]  
    

rg = ResponseGenerator()

def response_from_llm(text, session_id):
    rag_chain = rg.ragchain
    
    result = rag_chain.invoke(
        {"input": text}, 
            config = {"configurable": {"session_id": session_id}}
    )["answer"]

    return result