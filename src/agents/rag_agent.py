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
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)
        
        # Prompt Template
        self.template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        
        If you cannot find the answer in the context, please response anything that relevant according to the user's question
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

    def run(self, query: str, conversation_history: str = "") -> str:
        """
        End-to-end RAG run: Retrieve -> Generate.
        Enhanced to handle both specific database queries and general career questions.
        Now with conversational memory support.
        
        Args:
            query (str): Current user question
            conversation_history (str): Previous conversation context
            
        Returns:
            str: AI response
        """
        logger.info(f"RAG Agent received query: {query}")
        
        # 1. Retrieve relevant documents
        docs = self.retrieve_documents(query)
        
        # 2. Prepare context
        if docs:
            context_text = "\n\n".join([doc.page_content for doc in docs])
            logger.info(f"Found {len(docs)} relevant documents")
        else:
            context_text = ""
            logger.info("No specific documents found in database")
        
        # 3. Create smart prompt that handles both scenarios with history
        smart_prompt = ChatPromptTemplate.from_template(
            """You are a professional and helpful AI Career Assistant.

CONVERSATION HISTORY:
{history}

CONTEXT FROM DATABASE:
{context}

CURRENT USER QUESTION:
{question}

IMPORTANT INSTRUCTIONS:
1. ALWAYS respond in the SAME language as the user's question (Bahasa Indonesia or English)
2. Use CONVERSATION HISTORY to understand context and references (like "it", "that", "the one you mentioned")
3. If CONTEXT contains relevant information → use it to answer the question
4. If CONTEXT is empty or not relevant → still answer using your general knowledge as an AI Career Expert
5. NEVER say "I don't have enough information" or "No data available"
6. For general questions like greetings, introductions, or general career advice → respond naturally and helpfully
7. For specific questions about job listings → search the context first, if not found provide useful general advice
8. If user asks follow-up questions, reference previous answers from history
9. Tone: friendly, professional, and encouraging

YOUR RESPONSE:"""
        )
        
        # 4. Generate response
        chain = smart_prompt | self.llm | StrOutputParser()
        
        response = chain.invoke(
            {
                "context": context_text, 
                "question": query,
                "history": conversation_history if conversation_history else "No previous conversation."
            },
            config={"callbacks": [self.langfuse_handler]}
        )
        
        return response

if __name__ == "__main__":
    # Test run
    agent = RAGAgent()
    # print(agent.run("what is this data about?"))
