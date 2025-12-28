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

    def run(self, query: str) -> str:
        """
        End-to-end RAG run: Retrieve -> Generate.
        """
        logger.info(f"RAG Agent received query: {query}")
        
        # Cek jika query mengandung referensi ke percakapan sebelumnya
        if "[DAFTAR LOWONGAN DIBERIKAN SEBELUMNYA]" in query:
            # Ekstrak bagian konteks dari query
            parts = query.split("[DAFTAR LOWONGAN DIBERIKAN SEBELUMNYA]")
            if len(parts) > 1:
                context_part = parts[1].split("\n\nPertanyaan user:")[0]
                question_part = parts[-1]
                
                logger.info("Detected previous job list in context")
                
                # Gunakan konteks yang ada tanpa retrieval tambahan
                context_text = context_part
                final_query = question_part
            else:
                context_text = ""
                final_query = query
        else:
            # 1. Retrieve seperti biasa
            docs = self.retrieve_documents(query)
            
            if not docs:
                context_text = "No specific data found in the database."
            else:
                context_text = "\n\n".join([doc.page_content for doc in docs])
            
            final_query = query
        
        # Prompt yang lebih cerdas untuk menangani referensi
        smart_prompt = ChatPromptTemplate.from_template(
            """You are a professional Career Assistant.
    
            IMPORTANT CONTEXT FROM PREVIOUS CONVERSATION:
            {context}
    
            CURRENT USER QUESTION:
            {question}
    
            CRITICAL INSTRUCTIONS:
            1. CONTEXT AWARENESS: If the user refers to something specific (e.g., "job number 1", "the second point"), LOOK AT THE CONTEXT provided above.
            2. REFERENCE HANDLING: When user says "nomor 1" or "lowongan pertama", find that specific item in the context and explain ONLY that one.
            3. NO HALLUCINATION: If the context doesn't contain the specific item mentioned, DO NOT make up information. Say: "Based on our conversation, I don't see that specific item. Let me check..."
            4. LANGUAGE: Respond in the SAME LANGUAGE as the user's question.
            5. ACCURACY: Be precise and accurate. Cross-check numbers and details.
    
            YOUR RESPONSE:
            """
        )
        
        # 3. Generate
        chain = smart_prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({"context": context_text, "question": final_query})
        return response

if __name__ == "__main__":
    # Test run
    agent = RAGAgent()
    # print(agent.run("what is this data about?"))
