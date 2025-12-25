import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pypdf import PdfReader
from .rag_agent import RAGAgent
from langfuse.langchain import CallbackHandler

# OCR imports
try:
    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logging.warning("OCR libraries not available. Install with: pip install pdf2image pytesseract pillow")

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
        Supports both text-based PDFs and scanned/image PDFs (with OCR).
        """
        try:
            # First, try normal text extraction
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            # Check if we got meaningful text (more than just whitespace)
            if text.strip() and len(text.strip()) > 50:
                logger.info(f"Successfully extracted text using pypdf: {len(text)} characters")
                return text
            
            # If no text or very little text, try OCR
            logger.info("Text extraction yielded minimal results. Attempting OCR...")
            return self._extract_text_with_ocr(pdf_path)
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            # Try OCR as fallback
            logger.info("Attempting OCR as fallback...")
            return self._extract_text_with_ocr(pdf_path)

    def _extract_text_with_ocr(self, pdf_path: str) -> str:
        """
        Extracts text from PDF using OCR (for scanned documents or images).
        """
        if not OCR_AVAILABLE:
            logger.error("OCR libraries not installed. Cannot process scanned PDFs.")
            return "Error: This appears to be a scanned PDF, but OCR capabilities are not installed. Please install: pip install pdf2image pytesseract pillow"
        
        try:
            logger.info(f"Converting PDF to images for OCR: {pdf_path}")
            
            # Convert PDF to images
            # Note: This requires poppler-utils to be installed on the system
            images = convert_from_path(pdf_path, dpi=300)
            
            logger.info(f"Processing {len(images)} pages with OCR...")
            
            # Extract text from each image
            full_text = ""
            for i, image in enumerate(images):
                logger.info(f"OCR processing page {i+1}/{len(images)}...")
                
                # Perform OCR
                page_text = pytesseract.image_to_string(image, lang='eng')
                full_text += f"\n--- Page {i+1} ---\n{page_text}\n"
            
            if full_text.strip():
                logger.info(f"OCR successful: Extracted {len(full_text)} characters")
                return full_text
            else:
                logger.warning("OCR completed but no text was extracted")
                return "Error: Could not extract text from the PDF. The document might be empty or the image quality is too low."
                
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return f"Error during OCR processing: {str(e)}. Please ensure the PDF is readable and poppler-utils is installed."

    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extracts text from an image file (JPG, PNG, etc.) using OCR.
        """
        if not OCR_AVAILABLE:
            logger.error("OCR libraries not installed.")
            return "Error: OCR capabilities not available. Install: pip install pytesseract pillow"
        
        try:
            logger.info(f"Performing OCR on image: {image_path}")
            
            # Open image
            image = Image.open(image_path)
            
            # Perform OCR
            text = pytesseract.image_to_string(image, lang='eng')
            
            if text.strip():
                logger.info(f"OCR successful: Extracted {len(text)} characters from image")
                return text
            else:
                logger.warning("OCR completed but no text was extracted from image")
                return "Error: Could not extract text from the image. The image might be empty or quality is too low."
                
        except Exception as e:
            logger.error(f"Image OCR failed: {e}")
            return f"Error processing image: {str(e)}"

    def analyze_and_recommend(self, pdf_path: str) -> str:
        """
        Orchestrates the career consultation process:
        1. Extract text from CV (supports both text and scanned PDFs)
        2. Analyze CV (User Profiling)
        3. Retrieve relevant jobs via RAG
        4. Generate final recommendation
        """
        # 1. Extract text from CV (now with OCR support)
        cv_text = self.extract_text_from_pdf(pdf_path)
        
        # Check if extraction was successful
        if not cv_text or cv_text.startswith("Error:"):
            return cv_text if cv_text else "Could not extract text from the provided PDF."
        
        if len(cv_text.strip()) < 50:
            return "The extracted text is too short. Please ensure the CV is readable and contains sufficient information."

        # 2. User Profiling
        logger.info("Analyzing CV for user profiling...")
        profile_prompt = ChatPromptTemplate.from_template(
            """Analyze the following CV and extract a summary of the candidate's core skills, experience level, and preferred job roles.
            Output a concise search query string that can be used to find relevant job openings.
            
            Note: This text may have been extracted using OCR, so there might be minor formatting issues. Focus on extracting the key information.
            
            CV Content:
            {cv_text}
            
            Search Query:"""
        )
        profile_chain = profile_prompt | self.llm | StrOutputParser()
        search_query = profile_chain.invoke({"cv_text": cv_text}, config={"callbacks": [self.langfuse_handler]})
        logger.info(f"Generated search query: {search_query}")

        # 3. Delegate to RAGAgent
        logger.info("Delegating to RAGAgent for job search...")
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

            Note: The CV text may have been extracted using OCR, so focus on the content rather than formatting.

            Candidate's CV Summary:
            {cv_text}

            Potential Job Matches from Database:
            {jobs_context}

            Consultation Report:"""
        )
        
        consultation_chain = consultation_prompt | self.llm | StrOutputParser()
        
        recommendation = consultation_chain.invoke({
            "cv_text": cv_text[:5000],  # Truncate if too long
            "jobs_context": jobs_context
        }, config={"callbacks": [self.langfuse_handler]})
        
        return recommendation

    def run(self, query: str, context: str = None) -> str:
        """
        Generates advice based on the query. 
        Optionally takes 'context' if you want to feed it previous RAG/SQL results.
        """
        logger.info(f"Advisor Agent received query: {query}")
        
        # If context is provided, adjust the prompt
        if context:
            input_text = f"Context:\n{context}\n\nUser Query: {query}"
        else:
            input_text = query

        chain = self.prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({"input": input_text}, config={"callbacks": [self.langfuse_handler]})
        return response

if __name__ == "__main__":
    # Test OCR capability
    # agent = AdvisorAgent()
    # print(agent.analyze_and_recommend("path/to/scanned_cv.pdf"))
    pass
