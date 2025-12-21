import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .sql_agent import SQLAgent
from .sql_agent import SQLAgent
from .rag_agent import RAGAgent
from langfuse.langchain import CallbackHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class Orchestrator:
    def __init__(self):
        """
        Initializes the Orchestrator Agent.
        This agent routes queries to either the SQL Agent or the RAG Agent.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        
        # Initialize sub-agents
        self.sql_agent = SQLAgent()
        self.rag_agent = RAGAgent()
        
        # Initialize Langfuse CallbackHandler
        self.langfuse_handler = CallbackHandler()
        
        self.prompt = ChatPromptTemplate.from_template(
            """Analyze the following user query and determine if it requires retrieving specific structured data (like counts, averages, specific records from a database) or if it requires finding descriptive information from documents (like job descriptions, requirements, policy documents).
            
            User Query: {query}
            
            Respond with exactly one word: 'SQL' for structured data/database queries, or 'RAG' for descriptive/document search queries."""
        )

    def route_query(self, user_query: str) -> str:
        """
        Routes the user query to the appropriate agent.
        """
        logger.info(f"Orchestrator received query: {user_query}")
        
        # Determine intent
        chain = self.prompt | self.llm | StrOutputParser()
        intent = chain.invoke({"query": user_query}, config={"callbacks": [self.langfuse_handler]}).strip().upper()
        
        logger.info(f"Orchestrator determined intent: {intent}")
        
        if "SQL" in intent:
            logger.info("Routing to SQL Agent...")
            return self.sql_agent.run(user_query)
        elif "RAG" in intent:
            logger.info("Routing to RAG Agent...")
            return self.rag_agent.run(user_query)
        else:
            # Fallback or default to RAG if unsure, or ask clarification
            logger.warning(f"Unclear intent '{intent}', defaulting to RAG Agent.")
            return self.rag_agent.run(user_query)

if __name__ == "__main__":
    orchestrator = Orchestrator()
    # print(orchestrator.route_query("How many jobs are there?"))
    # print(orchestrator.route_query("Find me a job description for a software engineer."))