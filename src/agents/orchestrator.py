import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .sql_agent import SQLAgent
from .rag_agent import RAGAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class Orchestrator:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        # Kita gunakan gpt-4o-mini untuk klasifikasi cepat dan murah
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        
        self.sql_agent = SQLAgent()
        self.rag_agent = RAGAgent()
        
        # PROMPT REVISI: Menambahkan kategori 'GENERAL'
        self.routing_prompt = ChatPromptTemplate.from_template(
            """Analyze the user query and categorize it into one of these three:
            
            1. 'SQL': If the query asks for specific numbers, counts, or structured data from the job database (e.g., "Berapa banyak lowongan Python?").
            2. 'RAG': If the query is about career advice, job descriptions, or requirements (e.g., "Apa saja syarat jadi Data Scientist?").
            3. 'GENERAL': If the query is UNRELATED to career or jobs (e.g., "Apa itu fotosintesis?", "Siapa pemenang piala dunia?", "Halo apa kabar?").

            User Query: {query}
            
            Respond with exactly one word: 'SQL', 'RAG', or 'GENERAL'."""
        )

    def route_query(self, user_query: str) -> str:
        logger.info(f"Orchestrator checking query: {user_query}")
        
        try:
            # 1. Klasifikasi Intent
            routing_chain = self.routing_prompt | self.llm | StrOutputParser()
            intent = routing_chain.invoke({"query": user_query}).strip().upper()
            
            # 2. Logika Eksekusi
            if "SQL" in intent:
                return self.sql_agent.run(user_query)
            
            elif "RAG" in intent:
                return self.rag_agent.run(user_query)
            
            else:
                # JIKA GENERAL: Langsung dijawab oleh LLM di sini (Tanpa Agent)
                logger.info("Out of context query detected. Responding directly.")
                general_prompt = ChatPromptTemplate.from_template(
                    "You are a helpful AI assistant. Answer the user's question politely even though it's not about careers. Respond in the same language as the user: {query}"
                )
                general_chain = general_prompt | self.llm | StrOutputParser()
                return general_chain.invoke({"query": user_query})

        except Exception as e:
            logger.error(f"Orchestrator Error: {str(e)}")
            return "Maaf, sistem sedang mengalami kendala. Bisa ulangi pertanyaan Anda?"
