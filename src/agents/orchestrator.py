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


    def route_query(self, user_query: str, conversation_history: str = "") -> str:
        """
        Routes the user query to the appropriate agent.
        
        Args:
            user_query (str): The current user question
            conversation_history (str): Previous conversation context
            
        Returns:
            str: The agent's response
        """
        logger.info(f"Orchestrator received query: {user_query}")
        
        # Determine intent with history context
        chain = self.prompt | self.llm | StrOutputParser()
        intent = chain.invoke(
            {"query": user_query, "history": conversation_history},
            config={"callbacks": [self.langfuse_handler]}
        ).strip().upper()
        
        logger.info(f"Orchestrator determined intent: {intent}")
        
        if "SQL" in intent:
            logger.info("Routing to SQL Agent...")
            return self.sql_agent.run(user_query, conversation_history)
        elif "RAG" in intent:
            logger.info("Routing to RAG Agent...")
            return self.rag_agent.run(user_query, conversation_history)
        else:
            # Fallback to RAG if unsure
            logger.warning(f"Unclear intent '{intent}', defaulting to RAG Agent.")
            return self.rag_agent.run(user_query, conversation_history)

if __name__ == "__main__":
    orchestrator = Orchestrator()
            
