import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pypdf import PdfReader
from langfuse.langchain import CallbackHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class CoverLetterAgent:
    def __init__(self):
        """
        Initializes the Cover Letter Agent.
        This agent is responsible for generating tailored cover letters.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Using a professional but creative tone
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)
        
        # Initialize Langfuse CallbackHandler
        self.langfuse_handler = CallbackHandler()
        
        self.prompt_template = """You are an expert Career Coach and Professional Writer.
        
        Your task is to write a compelling, professional, and tailored cover letter for a candidate applying for a specific job.
        
        Candidate's Context (from CV):
        {cv_text}
        
        Job Description:
        {job_description}
        
        Instructions:
        1. Analyze the candidate's skills and experience from the CV.
        2. Analyze the requirements and responsibilities from the Job Description.
        3. Write a cover letter that highlights the candidate's most relevant qualifications for this specific role.
        4. The tone should be professional, enthusiastic, and confident.
        5. Keep it concise (approx. 300-400 words).
        6. Use standard business letter formatting (Subject line, Salutation, Body, Closing).
        7. If the CV text is missing or unclear, make reasonable assumptions based on standard industry practices but prioritize the provided info.
        
        Cover Letter:"""
        
        self.prompt = ChatPromptTemplate.from_template(self.prompt_template)

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

    def generate_cover_letter(self, cv_path: str, job_description: str) -> str:
        """
        Generates a cover letter based on the provided CV PDF and Job Description.
        """
        logger.info(f"Generating cover letter for CV: {cv_path}")
        
        # 1. Extract text from CV
        cv_text = self.extract_text_from_pdf(cv_path)
        if not cv_text:
            return "Could not extract text from the provided CV PDF."

        # 2. Generate Cover Letter
        chain = self.prompt | self.llm | StrOutputParser()
        
        # Truncate CV text if it's too long to avoid token limits, though gpt-4o-mini has good context window.
        # 10000 chars is usually safe for a CV.
        cover_letter = chain.invoke({
            "cv_text": cv_text[:10000], 
            "job_description": job_description
        }, config={"callbacks": [self.langfuse_handler]})
        
        return cover_letter

if __name__ == "__main__":
    # Example usage code (commented out)
    # agent = CoverLetterAgent()
    # cv_path = "path/to/cv.pdf"
    # job_desc = "Software Engineer at Google..."
    # print(agent.generate_cover_letter(cv_path, job_desc))
    pass
