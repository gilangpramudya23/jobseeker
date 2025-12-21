import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import logging
from typing import List, Optional
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.database.setup_qdrant import get_qdrant_client
from langfuse.langchain import CallbackHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class RAGAgent:
    def __init__(self, collection_name: str = "job_market"):
        """
        Initializes the RAG Agent with a Qdrant client, Embedding model, and LLM.
        """
        self.collection_name = collection_name
        self.client = get_qdrant_client()
        
        # Initialize Embeddings (must match the vector size in setup_qdrant, default 1536)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY is not set. RAG Agent may fail.")
            
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
        
        # Initialize LLM
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        
        # Prompt Template
        self.template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        
        If you cannot find the answer in the context, please say "I don't have enough information in my knowledge base to answer this."
        """
        self.prompt = ChatPromptTemplate.from_template(
            """You are a helpful Careeer Assistant.
            
            Context from our database:
            {context}
            
            User Question:
            {question}
            
            Instructions:
            1. If the context contains relevant information, answer the question clearly using that information.
            2. If the context is empty or doesn't have the answer, DO NOT say "I don't know" or "No data".
            3. Instead, give a general professional response based on your general knowledge as an AI, then suggest what the user can ask or do next (e.g., "Currently, I don't have specific job listings for that, but generally for this role you should prepare...")
            4. Keep the tone encouraging and professional.

            Your Response:
            """
        )
        
        # Initialize Langfuse CallbackHandler
        self.langfuse_handler = CallbackHandler()

    def retrieve_documents(self, query: str, limit: int = 3) -> List[Document]:
        """
        Embeds the query and searches the Qdrant collection.
        Returns a list of LangChain Documents.
        """
        try:
            query_vector = self.embeddings.embed_query(query)
            
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit
            ).points
            
            documents = []
            for hit in search_results:
                # Extract text content from payload
                # Adjust 'text' key based on how you ingested data
                page_content = hit.payload.get("text", hit.payload.get("content", str(hit.payload)))
                metadata = hit.payload
                documents.append(Document(page_content=page_content, metadata=metadata))
            
            return documents
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []

    def run(self, query: str) -> str:
        """
        End-to-end RAG run: Retrieve -> Generate.
        """
        logger.info(f"RAG Agent received query: {query}")
        
        # 1. Retrieve
        docs = self.retrieve_documents(query)
        
        if not docs:
            context_text = "No specific data found in the database. Please answer using your general knowledge."
        else:
            context_text = "\n\n".join([doc.page_content for doc in docs])
        
        flexible_prompt = ChatPromptTemplate.from_template(
            """You are a professional Career Assistant.
            CONTEXT FROM DATABASE:
            {context}
            USER QUESTION:
            {question}
            INSTRUCTIONS:
            1. Respond in the SAME LANGUAGE as the user's question.
            2. If CONTEXT is empty, use your internal professional knowledge to provide a helpful answer.
            3. Never say 'I don't know' or 'No data found'
            4. Be encouraging and provide actionable advice.
            """
        )
        
        # 3. Generate
        chain = flexible_prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({"context": context_text, "question": query})
        return response

if __name__ == "__main__":
    # Test run
    agent = RAGAgent()
    # print(agent.run("what is this data about?"))
