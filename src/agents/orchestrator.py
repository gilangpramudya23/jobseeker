import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .sql_agent import SQLAgent
from .rag_agent import RAGAgent

# Konfigurasi logging agar lebih informatif di Streamlit Cloud
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class Orchestrator:
    def __init__(self):
        """
        Initializes the Orchestrator Agent.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        # Suhu 0 agar pemilihan routing konsisten/tidak ngawur
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        
        # Inisialisasi sub-agents
        self.sql_agent = SQLAgent()
        self.rag_agent = RAGAgent()
        
        # PROMPT REVISI: Menambahkan instruksi bahasa dan deskripsi tugas yang lebih jelas
        self.prompt = ChatPromptTemplate.from_template(
            """Analyze the following user query. 
            Determine if the user wants:
            1. 'SQL': Structured data (counting, averaging, or specific records from the job database).
            2. 'RAG': Descriptive info, advice, or career questions (job descriptions, requirements, policy, or general career help).

            User Query: {query}
            
            Respond with exactly one word: 'SQL' or 'RAG'."""
        )

    def route_query(self, user_query: str) -> str:
        """
        Routes the user query to the appropriate agent with error handling.
        """
        logger.info(f"Orchestrator received query: {user_query}")
        
        try:
            # 1. Tentukan Intent (SQL atau RAG)
            chain = self.prompt | self.llm | StrOutputParser()
            intent = chain.invoke({"query": user_query}).strip().upper()
            
            logger.info(f"Orchestrator determined intent: {intent}")
            
            # 2. Routing Logic
            if "SQL" in intent:
                logger.info("Routing to SQL Agent...")
                return self.sql_agent.run(user_query)
            
            # Default ke RAG karena RAG sekarang punya "Fallback Knowledge" 
            # sehingga bisa menjawab pertanyaan umum apa pun.
            else:
                logger.info("Routing to RAG Agent (Fallback Handler)...")
                return self.rag_agent.run(user_query)

        except Exception as e:
            logger.error(f"Orchestrator Error: {str(e)}")
            # Berikan respon dalam bahasa yang ramah jika sistem error
            return "Maaf, saya sedang mengalami kendala teknis dalam memproses permintaan Anda. Silakan coba beberapa saat lagi."

if __name__ == "__main__":
    orchestrator = Orchestrator()
