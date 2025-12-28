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

    def route_query(self, user_query: str, history: list = None) -> str:
        """
        Route user query dengan mempertimbangkan history percakapan.
        
        Args:
            user_query: Pertanyaan user saat ini
            history: List of dicts dengan format [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        
        Returns:
            Response dari agent yang sesuai
        """
        try:
            # 1. Tentukan rute (gunakan hanya query saat ini untuk routing decision)
            router_chain = self.router_prompt | self.llm | StrOutputParser()
            decision = router_chain.invoke({"query": user_query}).strip()
            
            logger.info(f"Routing Decision: {decision}")

            # 2. Eksekusi berdasarkan rute dengan mempertimbangkan history
            if "USE_SQL" in decision:
                # Untuk SQL, gabungkan konteks dari history dengan query
                if history and len(history) > 0:
                    # Ambil pesan terakhir dari assistant untuk konteks
                    context_history = self._format_history_for_context(history)
                    enriched_query = f"Context from previous conversation:\n{context_history}\n\nCurrent question: {user_query}"
                    return self.sql_agent.run(enriched_query)
                else:
                    return self.sql_agent.run(user_query)
            
            elif "USE_RAG" in decision:
                # Untuk RAG, gunakan history sebagai konteks tambahan
                if history and len(history) > 0:
                    # Format history untuk RAG context
                    context_history = self._format_history_for_context(history)
                    enriched_query = f"Previous conversation context:\n{context_history}\n\nCurrent question: {user_query}"
                    return self.rag_agent.run(enriched_query)
                else:
                    return self.rag_agent.run(user_query)
            
            else:
                # JIKA CHAT/GENERAL: Gunakan history lengkap untuk menjaga konteks percakapan
                logger.info("Handling as General Chat")
                return self._handle_general_chat(user_query, history)

        except Exception as e:
            logger.error(f"Orchestrator Error: {str(e)}")
            return "Maaf, ada kendala teknis. Bisa ulangi pertanyaannya?"

    def _handle_general_chat(self, user_query: str, history: list = None) -> str:
        """
        Handle general chat dengan mempertimbangkan history percakapan.
        Mengikuti pola dari contoh manager.
        """
        from langchain_core.messages import HumanMessage, AIMessage
        
        # Buat messages array berdasarkan contoh dari manager
        messages = []
        
        # Tambahkan system prompt
        messages.append({
            "role": "system", 
            "content": """You are a helpful and friendly AI Career Assistant. 
            Even if the user asks something unrelated to careers, respond politely and naturally. 
            Respond in the SAME LANGUAGE as the user.
            Be professional but warm.
            Do not say 'I am only for jobs'. Just help the user."""
        })
        
        # Add conversation history (jika ada)
        if history:
            for msg in history:
                if msg["role"] == "user":
                    messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    messages.append({"role": "assistant", "content": msg["content"]})
        
        # Tambahkan current question
        messages.append({"role": "user", "content": user_query})
        
        # Buat prompt dari messages
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", messages[0]["content"]),
            *[(msg["role"], msg["content"]) for msg in messages[1:]]
        ])
        
        chat_chain = chat_prompt | self.llm | StrOutputParser()
        return chat_chain.invoke({})

    def _format_history_for_context(self, history: list, max_messages: int = 3) -> str:
        """
        Format history menjadi string untuk konteks.
        
        Args:
            history: List of conversation history
            max_messages: Jumlah maksimum pesan terakhir yang digunakan
        
        Returns:
            Formatted history string
        """
        if not history:
            return ""
        
        # Ambil beberapa pesan terakhir
        recent_history = history[-max_messages:]
        
        formatted = []
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted)

    # Metode lama untuk kompatibilitas (bisa dihapus setelah update semua pemanggilan)
    def route_request(self, user_query, history_text):
        """
        Metode lama untuk kompatibilitas.
        Akan mengonversi history_text ke format list.
        """
        # Convert history_text to list format jika perlu
        history = []
        if history_text:
            # Simple parsing - bisa disesuaikan dengan format yang digunakan
            lines = history_text.strip().split('\n')
            for line in lines:
                if line.startswith('user:'):
                    history.append({"role": "user", "content": line[5:].strip()})
                elif line.startswith('assistant:'):
                    history.append({"role": "assistant", "content": line[10:].strip()})
        
        return self.route_query(user_query, history)
