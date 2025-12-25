import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pypdf import PdfReader
from .rag_agent import RAGAgent
from langfuse.langchain import CallbackHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class AdvisorAgent:
    def __init__(self):
        """
        Initializes the Advisor Agent.
        This agent is responsible for providing high-level advice, 
        synthesizing information, or handling general queries.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Using a slightly higher temperature for more creative/advisory tone
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)
        
        # Initialize RAG Agent
        self.rag_agent = RAGAgent()
        
        # Initialize Langfuse CallbackHandler
        self.langfuse_handler = CallbackHandler()
        
        self.prompt = ChatPromptTemplate.from_template(
            """You are an expert AI Career Consultant. Your role is to provide detailed, helpful, and professional career advice.
            
            User Query: {input}
            
            Provide your advice:"""
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extracts text from a PDF file.
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""

    def analyze_and_recommend(self, pdf_path: str) -> str:
        """
        Orchestrates the career consultation process:
        1. Extract text from CV
        2. Analyze CV (User Profiling)
        3. Retrieve relevant jobs via RAG
        4. Generate final recommendation
        """
        # 1. Extract text from CV
        cv_text = self.extract_text_from_pdf(pdf_path)
        if not cv_text:
            return "Could not extract text from the provided PDF."

        # 2. User Profiling
        logger.info("Analyzing CV for user profiling...")
        profile_prompt = ChatPromptTemplate.from_template(
            """Analyze the following CV and extract a summary of the candidate's core skills, experience level, and preferred job roles.
            Output a concise search query string that can be used to find relevant job openings.
            
            CV Content:
            {cv_text}
            
            Search Query:"""
        )
        profile_chain = profile_prompt | self.llm | StrOutputParser()
        search_query = profile_chain.invoke({"cv_text": cv_text}, config={"callbacks": [self.langfuse_handler]})
        logger.info(f"Generated search query: {search_query}")

        # 3. Delegate to RAGAgent
        logger.info("Delegating to RAGAgent for job search...")
        # We use the retrieve_documents method directly to get the docs, 
        # so we can feed them into our final recommendation prompt.
        job_docs = self.rag_agent.retrieve_documents(search_query, limit=5)
        
        jobs_context = "\n\n".join([f"Job {i+1}:\n{doc.page_content}" for i, doc in enumerate(job_docs)])

        if not jobs_context:
            jobs_context = "No specific job matches found in the database."

        # 4. Give career recommendation
        logger.info("Generating career recommendation...")
        consultation_prompt = ChatPromptTemplate.from_template(
            """You are an expert Career Consultant. A candidate has provided their CV, and we have found some potential job matches from our database.
            
            Your task is to:
            1. Analyze how the candidate's profile matches the found jobs.
            2. Recommend which jobs they should apply for and why.
            3. Suggest any skills they might need to improve or highlight.
            4. Provide general career advice based on their profile.

            Candidate's CV Summary:
            {cv_text}

            Potential Job Matches from Database:
            {jobs_context}

            Consultation Report:"""
        )
        
        # We can pass the raw CV text or a summary. Passing raw text might be token-heavy but more accurate. 
        # Let's pass a truncated version if it's too long, or just the full text for now assuming it fits in context.
        consultation_chain = consultation_prompt | self.llm | StrOutputParser()
        
        recommendation = consultation_chain.invoke({
            "cv_text": cv_text[:5000], # Truncate to safety if extremely long
            "jobs_context": jobs_context
        }, config={"callbacks": [self.langfuse_handler]})
        
        return recommendation

    def run(self, query: str, context: str = None) -> str:
        """
        Generates advice based on the query. 
        Optionally takes 'context' if you want to feed it previous RAG/SQL results.
        """
        logger.info(f"Advisor Agent received query: {query}")
        
        # If context is provided, we might want to adjust the prompt dynamically or append it
        if context:
            input_text = f"Context:\n{context}\n\nUser Query: {query}"
        else:
            input_text = query

        chain = self.prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({"input": input_text}, config={"callbacks": [self.langfuse_handler]})
        return response

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os

class CareerAdvisor:
    def extract_text(self, file_path):
        ext = file_path.split('.')[-1].lower()
        extracted_text = ""

        if ext in ['jpg', 'jpeg', 'png']:
            # Pemrosesan Gambar Langsung
            extracted_text = pytesseract.image_to_string(Image.open(file_path))
        
        elif ext == 'pdf':
            # Coba baca sebagai PDF teks biasa dulu (Gunakan library PDF Anda, misal pdfplumber)
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text += text
            
            # Jika teks kosong, berarti ini PDF Scan/Gambar. Gunakan OCR.
            if not extracted_text.strip():
                images = convert_from_path(file_path)
                for img in images:
                    extracted_text += pytesseract.image_to_string(img)
        
        return extracted_text

    def analyze_and_recommend(self, file_path):
        # 1. Ekstrak teks (Teks atau Scan tetap terbaca)
        cv_text = self.extract_text(file_path)
        
        if not cv_text.strip():
            return "Maaf, saya tidak dapat membaca teks di dokumen tersebut. Pastikan gambar cukup jelas."

        # 2. Kirim cv_text ke LLM untuk analisis (seperti logika lama Anda)
        # response = self.chain.invoke({"cv_data": cv_text})
        return f"Hasil Analisis Berdasarkan CV Anda:\n\n{cv_text[:500]}..." # Contoh output

if __name__ == "__main__":
    # Ensure this script is run from the project root or src is in pythonpath
    # Example usage:
    # agent = AdvisorAgent()
    # print(agent.analyze_and_recommend("path/to/cv.pdf"))
    pass
