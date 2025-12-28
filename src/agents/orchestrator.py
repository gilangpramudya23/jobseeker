import os
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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
        
        # Context Memory untuk menyimpan percakapan terbaru
        self.context_memory = {}
        
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

    def _extract_job_list_from_context(self, context: str) -> List[Dict]:
        """Ekstrak daftar lowongan dari konteks percakapan"""
        # Logika sederhana untuk menemukan pola daftar lowongan
        jobs = []
        lines = context.split('\n')
        
        current_job = None
        for line in lines:
            line = line.strip()
            if line.startswith('###') or line.startswith('**') or 'Lowongan' in line:
                if current_job:
                    jobs.append(current_job)
                current_job = {"title": line.replace('###', '').replace('**', '').strip()}
            elif '**Perusahaan:**' in line or 'Perusahaan:' in line:
                if current_job:
                    current_job["company"] = line.split(':')[-1].strip()
            elif '**Lokasi:**' in line or 'Lokasi:' in line:
                if current_job:
                    current_job["location"] = line.split(':')[-1].strip()
        
        if current_job and current_job not in jobs:
            jobs.append(current_job)
        
        return jobs

    def _create_context_summary(self, history: List[Dict]) -> str:
        """Buat ringkasan konteks dari percakapan"""
        if not history:
            return ""
        
        # Ambil 5 pesan terakhir (2-3 percakapan terakhir)
        recent_history = history[-5:] if len(history) > 5 else history
        
        summary_parts = []
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            
            # Jika assistant memberikan daftar lowongan, tambahkan penanda khusus
            if role == "Assistant" and any(keyword in content for keyword in ['lowongan', 'Lowongan', 'job', 'Job']):
                summary_parts.append(f"[DAFTAR LOWONGAN DIBERIKAN SEBELUMNYA]\n{content[:500]}...")
            else:
                summary_parts.append(f"{role}: {content[:200]}")
        
        return "\n".join(summary_parts)

    def route_query(self, user_query: str, history: List[Dict] = None, session_id: str = "default") -> str:
        """
        Route user query dengan mempertimbangkan history dan konteks.
        
        Args:
            user_query: Pertanyaan user saat ini
            history: List percakapan [{"role": "user/assistant", "content": "..."}]
            session_id: ID sesi untuk memory
        
        Returns:
            Response dari agent yang sesuai
        """
        try:
            # 1. Update context memory untuk sesi ini
            if session_id not in self.context_memory:
                self.context_memory[session_id] = []
            
            if history:
                # Simpan history terbaru ke memory
                self.context_memory[session_id] = history[-10:]  # Simpan 10 pesan terakhir
            
            # 2. Analisis apakah user merujuk ke sesuatu di percakapan sebelumnya
            context_ref = ""
            if history:
                context_ref = self._create_context_summary(history)
                
                # Cek jika user merujuk ke nomor tertentu (e.g., "lowongan nomor 1")
                if any(word in user_query.lower() for word in ['nomor', 'no.', 'lowongan', 'point', 'poin']):
                    # Cari daftar lowongan dalam history
                    for msg in reversed(history):
                        if msg["role"] == "assistant" and any(keyword in msg["content"] for keyword in ['lowongan', 'Lowongan']):
                            # Tambahkan konteks spesifik ke query
                            context_ref = f"Konteks spesifik (dari percakapan sebelumnya):\n{msg['content']}\n\n"
                            break
            
            # 3. Tentukan rute dengan konteks
            router_chain = self.router_prompt | self.llm | StrOutputParser()
            
            # Gabungkan konteks dengan query untuk routing decision
            query_with_context = f"Konteks percakapan sebelumnya: {context_ref[:500]}\n\nPertanyaan user: {user_query}"
            decision = router_chain.invoke({"query": query_with_context}).strip()
            
            logger.info(f"Routing Decision: {decision}")
            logger.info(f"Context length: {len(context_ref)} chars")

            # 4. Eksekusi berdasarkan rute dengan context-aware query
            if "USE_SQL" in decision:
                # Untuk SQL, tambahkan konteks jika relevan
                if context_ref and any(word in user_query.lower() for word in ['berapa', 'jumlah', 'statistik', 'data']):
                    enhanced_query = f"{context_ref}\n\n{user_query}"
                    return self.sql_agent.run(enhanced_query)
                return self.sql_agent.run(user_query)
            
            elif "USE_RAG" in decision:
                # Untuk RAG, SELALU sertakan konteks
                enhanced_query = f"{context_ref}\n\nPertanyaan user: {user_query}"
                logger.info(f"Sending to RAG with enhanced query")
                return self.rag_agent.run(enhanced_query)
            
            else:
                # Untuk CHAT GENERAL: Gunakan konteks lengkap
                logger.info("Handling as General Chat with context")
                return self._handle_chat_with_context(user_query, history)

        except Exception as e:
            logger.error(f"Orchestrator Error: {str(e)}")
            return "Maaf, ada kendala teknis. Bisa ulangi pertanyaannya?"

    def _handle_chat_with_context(self, user_query: str, history: List[Dict] = None) -> str:
        """Handle chat dengan mempertimbangkan konteks percakapan"""
        
        # Build messages dengan konteks
        messages = []
        
        # System prompt yang lebih cerdas
        system_prompt = """Anda adalah Asisten Karir AI yang membantu dan ramah.
        
        INSTRUKSI PENTING:
        1. RESET KONTEKS: Selalu perhatikan konteks percakapan sebelumnya.
        2. REFERENSI SPESIFIK: Jika user merujuk ke sesuatu yang spesifik (misal "lowongan nomor 1", "point kedua"), lihat di konteks percakapan sebelumnya.
        3. BAHASA: Tanggapi dalam BAHASA YANG SAMA dengan user.
        4. JANGAN BOHONG: Jika tidak tahu atau tidak yakin, katakan dengan jujur.
        5. BANTU TETAP: Tetap bantu user meski pertanyaan di luar topik karir.
        
        KONTEKS PERCAKAPAN SEBELUMNYA:
        """
        
        # Tambahkan konteks dari history
        if history:
            context_text = self._create_context_summary(history)
            system_prompt += f"\n{context_text}\n"
        else:
            system_prompt += "\n[Tidak ada percakapan sebelumnya]\n"
        
        messages.append(("system", system_prompt))
        
        # Tambahkan history sebagai messages
        if history:
            for msg in history[-6:]:  # Ambil 6 pesan terakhir
                role = "user" if msg["role"] == "user" else "assistant"
                messages.append((role, msg["content"]))
        
        # Tambahkan query terbaru
        messages.append(("user", user_query))
        
        # Buat prompt
        chat_prompt = ChatPromptTemplate.from_messages(messages)
        chat_chain = chat_prompt | self.llm | StrOutputParser()
        
        return chat_chain.invoke({})

    def clear_memory(self, session_id: str = "default"):
        """Bersihkan memory untuk sesi tertentu"""
        if session_id in self.context_memory:
            del self.context_memory[session_id]
            logger.info(f"Cleared memory for session: {session_id}")
