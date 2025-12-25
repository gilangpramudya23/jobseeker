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
        Enhanced with better error handling and debugging.
        """
        logger.info(f"Starting text extraction from: {pdf_path}")
        
        try:
            # First, try normal text extraction
            logger.info("Attempting standard text extraction with pypdf...")
            reader = PdfReader(pdf_path)
            text = ""
            
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                logger.info(f"Page {i+1}: Extracted {len(page_text)} characters")
            
            # Check if we got meaningful text (more than just whitespace)
            cleaned_text = text.strip()
            logger.info(f"Total extracted text length: {len(cleaned_text)} characters")
            
            if cleaned_text and len(cleaned_text) > 100:  # Lowered threshold to 100
                logger.info("✅ Standard text extraction successful!")
                return text
            
            # If no text or very little text, try OCR
            logger.warning(f"Text extraction yielded only {len(cleaned_text)} characters. Attempting OCR...")
            return self._extract_text_with_ocr(pdf_path)
            
        except Exception as e:
            logger.error(f"Error in standard text extraction: {e}")
            # Try OCR as fallback
            logger.info("Attempting OCR as fallback...")
            return self._extract_text_with_ocr(pdf_path)

    def _extract_text_with_ocr(self, pdf_path: str) -> str:
        """
        Extracts text from PDF using OCR (for scanned documents or images).
        Enhanced with better configuration and error messages.
        """
        if not OCR_AVAILABLE:
            error_msg = """OCR libraries not installed. Cannot process scanned PDFs.

To install OCR support:
1. Install Python packages: pip install pdf2image pytesseract pillow
2. Install system dependencies:
   - Ubuntu/Debian: sudo apt-get install tesseract-ocr poppler-utils
   - macOS: brew install tesseract poppler
   - Windows: Install Tesseract and Poppler (see documentation)
"""
            logger.error(error_msg)
            return f"Error: {error_msg}"
        
        try:
            logger.info(f"🔍 Converting PDF to images for OCR: {pdf_path}")
            
            # Convert PDF to images with higher DPI for better quality
            try:
                images = convert_from_path(
                    pdf_path, 
                    dpi=300,  # High DPI for better OCR accuracy
                    fmt='jpeg'
                )
            except Exception as conv_error:
                logger.error(f"PDF to image conversion failed: {conv_error}")
                return f"Error: Could not convert PDF to images. Ensure poppler-utils is installed. Error: {conv_error}"
            
            logger.info(f"✅ Successfully converted PDF to {len(images)} image(s)")
            
            if not images:
                return "Error: PDF conversion resulted in 0 images. The PDF might be corrupted or empty."
            
            # Extract text from each image with enhanced OCR config
            full_text = ""
            
            # Enhanced OCR configuration for better accuracy
            custom_config = r'--oem 3 --psm 6'  # OEM 3 = Default, PSM 6 = Uniform block of text
            
            for i, image in enumerate(images):
                logger.info(f"🔄 OCR processing page {i+1}/{len(images)}...")
                
                try:
                    # Try multiple languages (English and Indonesian)
                    page_text = pytesseract.image_to_string(
                        image, 
                        lang='eng+ind',  # Support both English and Indonesian
                        config=custom_config
                    )
                    
                    if page_text.strip():
                        full_text += f"\n{'='*50}\n"
                        full_text += f"PAGE {i+1}\n"
                        full_text += f"{'='*50}\n"
                        full_text += page_text + "\n"
                        logger.info(f"   ✅ Page {i+1}: Extracted {len(page_text)} characters")
                    else:
                        logger.warning(f"   ⚠️ Page {i+1}: No text extracted")
                        # Try with different PSM mode as fallback
                        logger.info(f"   🔄 Retrying page {i+1} with alternative OCR mode...")
                        page_text = pytesseract.image_to_string(
                            image, 
                            lang='eng',
                            config=r'--oem 3 --psm 3'  # PSM 3 = Fully automatic page segmentation
                        )
                        if page_text.strip():
                            full_text += f"\n{'='*50}\n"
                            full_text += f"PAGE {i+1} (Alternative mode)\n"
                            full_text += f"{'='*50}\n"
                            full_text += page_text + "\n"
                            logger.info(f"   ✅ Page {i+1}: Extracted {len(page_text)} characters (alternative mode)")
                
                except Exception as ocr_error:
                    logger.error(f"   ❌ OCR failed for page {i+1}: {ocr_error}")
                    full_text += f"\n[Error processing page {i+1}: {ocr_error}]\n"
            
            # Final check
            cleaned_full_text = full_text.strip()
            logger.info(f"📊 OCR Summary: Total extracted {len(cleaned_full_text)} characters from {len(images)} pages")
            
            if cleaned_full_text and len(cleaned_full_text) > 50:
                logger.info("✅ OCR successful!")
                return full_text
            else:
                error_msg = f"""OCR completed but extracted very little text ({len(cleaned_full_text)} characters).

Possible reasons:
1. Image quality is too low - try rescanning at 300+ DPI
2. Text is too small or blurry
3. PDF contains only images without text
4. Language not supported (currently using: eng+ind)

Please try:
- Rescan the document at higher quality
- Ensure the PDF contains readable text
- Check if the file is corrupted
"""
                logger.error(error_msg)
                return f"Error: {error_msg}"
                
        except Exception as e:
            logger.error(f"❌ OCR extraction failed: {e}")
            import traceback
            detailed_error = traceback.format_exc()
            logger.error(f"Detailed error: {detailed_error}")
            
            return f"""Error during OCR processing: {str(e)}

Common solutions:
1. Ensure Tesseract is installed and in PATH
2. Ensure poppler-utils is installed
3. Check PDF file is not corrupted
4. Verify Python packages are installed: pip install pdf2image pytesseract pillow

Technical details:
{detailed_error}
"""

    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extracts text from an image file (JPG, PNG, etc.) using OCR.
        """
        if not OCR_AVAILABLE:
            return "Error: OCR capabilities not available. Install: pip install pytesseract pillow"
        
        try:
            logger.info(f"Performing OCR on image: {image_path}")
            
            # Open image
            image = Image.open(image_path)
            
            # Perform OCR with dual language support
            text = pytesseract.image_to_string(image, lang='eng+ind')
            
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
        # 1. Extract text from CV (now with enhanced OCR support)
        logger.info("="*60)
        logger.info("Starting CV Analysis Process")
        logger.info("="*60)
        
        cv_text = self.extract_text_from_pdf(pdf_path)
        
        # Check if extraction was successful
        if not cv_text or cv_text.startswith("Error:"):
            logger.error("❌ Text extraction failed")
            return cv_text if cv_text else "Could not extract text from the provided PDF."
        
        cleaned_cv_text = cv_text.strip()
        logger.info(f"📄 Extracted CV text length: {len(cleaned_cv_text)} characters")
        
        if len(cleaned_cv_text) < 50:
            return f"The extracted text is too short ({len(cleaned_cv_text)} characters). Please ensure the CV is readable and contains sufficient information."

        # Log first 500 characters for debugging (remove in production)
        logger.info(f"📝 CV Preview (first 500 chars):\n{cleaned_cv_text[:500]}...")

        # 2. User Profiling
        logger.info("🔍 Analyzing CV for user profiling...")
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
        logger.info(f"✅ Generated search query: {search_query}")

        # 3. Delegate to RAGAgent
        logger.info("🔍 Delegating to RAGAgent for job search...")
        job_docs = self.rag_agent.retrieve_documents(search_query, limit=5)
        
        jobs_context = "\n\n".join([f"Job {i+1}:\n{doc.page_content}" for i, doc in enumerate(job_docs)])

        if not jobs_context:
            jobs_context = "No specific job matches found in the database."
            logger.warning("⚠️ No job matches found")
        else:
            logger.info(f"✅ Found {len(job_docs)} job matches")

        # 4. Give career recommendation
        logger.info("📋 Generating career recommendation...")
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
        
        logger.info("✅ Career recommendation generated successfully")
        logger.info("="*60)
        
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
    # Test OCR capability with detailed logging
    # agent = AdvisorAgent()
    # result = agent.analyze_and_recommend("test_cv.pdf")
    # print(result)
    pass
