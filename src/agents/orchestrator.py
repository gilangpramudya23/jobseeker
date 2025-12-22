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
        # Menggunakan GPT-4o-mini untuk kecerdasan maksimal dalam menentukan rute
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)
        
        # Inisialisasi sub-agents
        self.sql_agent = SQLAgent()
        self.rag_agent = RAGAgent()
        
        # Prompt untuk menentukan apakah butuh 'TOOLS' atau 'CHAT'
        self.router_prompt = ChatPromptTemplate.from_template(
            """You are a smart AI Career Router. Your job is to analyze the user input.
            
            USER INPUT: {query}

            CATEGORIES:
            - 'USE_SQL': If the user asks for numbers, statistics, or database records (e.g., "How many jobs?", "List Python jobs").
            - 'USE_RAG': If the user asks for specific career advice, job requirements, or company info found in documents.
            - 'CHAT': If the user is just greeting, saying thank you, or asking general/out-of-context questions (e.g., "Hi", "Who are you?", "Tell me a joke", "What is 1+1?").

            Respond with ONLY the category name."""
        )

    def route_request(self, user_query, history_text):
        full_prompt = f"""
        Berikut adalah riwayat percakapan sebelumnya:
        {history_text}

        Pertanyaan baru user: {user_query}
        """
        response = self.llm.invoke(full_prompt)
        return response.content

    def route_query(self, user_query: str) -> str:
        try:
            # 1. Tentukan rute
            router_chain = self.router_prompt | self.llm | StrOutputParser()
            decision = router_chain.invoke({"query": user_query}).strip()
            
            logger.info(f"Routing Decision: {decision}")

            # 2. Eksekusi berdasarkan rute
            if "USE_SQL" in decision:
                return self.sql_agent.run(user_query)
            
            elif "USE_RAG" in decision:
                return self.rag_agent.run(user_query)
            
            else:
                # JIKA CHAT/GENERAL: AI menjawab langsung dengan kepribadian yang ramah
                logger.info("Handling as General Chat")
                chat_prompt = ChatPromptTemplate.from_template(
                    """You are a helpful and friendly AI Career Assistant. 
                    Even if the user asks something unrelated to careers, respond politely and naturally. 
                    
                    USER INPUT: {query}
                    
                    INSTRUCTION:
                    - Respond in the SAME LANGUAGE as the user.
                    - Be professional but warm.
                    - Do not say 'I am only for jobs'. Just help the user.
                    """
                )
                chat_chain = chat_prompt | self.llm | StrOutputParser()
                return chat_chain.invoke({"query": user_query})

        except Exception as e:
            logger.error(f"Orchestrator Error: {str(e)}")
            return "Maaf, ada kendala teknis. Bisa ulangi pertanyaannya?"
            
