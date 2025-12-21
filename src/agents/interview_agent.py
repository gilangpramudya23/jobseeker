import os
import speech_recognition as sr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class InterviewAgent:
    def __init__(self):
        """
        Initializes the Interview Agent with SpeechRecognition and LangChain.
        """
        self.recognizer = sr.Recognizer()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        
        # Define the interview prompt
        self.template = """
        You are an expert technical interviewer. 
        Your goal is to conduct a mock interview with the candidate.
        
        Current conversation history:
        {history}
        
        Candidate's answer: {answer}
        
        Task:
        1. Evaluate the candidate's answer (briefly).
        2. Ask the next relevant follow-up question.
        
        Keep your response conversational but professional.
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.history = ""

    def listen(self):
        """
        Listens to the microphone and converts speech to text using OpenAI Whisper (via SpeechRecognition if available or API).
        For this implementation, we'll use local Google Speech Recognition as a default fallback, 
        or specifically configured for Whisper API if the user has keys/setup.
        
        To keep it simple and robust:
        We will use `recognizer.recognize_google` for quick testing if Whisper API isn't explicitly requested 
        but the plan mentioned "Send audio to OpenAI Whisper". 
        
        To use OpenAI Whisper API via speech_recognition:
        recognizer.recognize_whisper_api(audio, api_key=OPENAI_API_KEY)
        """
        with sr.Microphone() as source:
            print("\nListening... (Speak now)")
            # Adjust for ambient noise
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=30)
                print("Processing audio...")
                
                # Using OpenAI Whisper API for better accuracy
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    text = self.recognizer.recognize_whisper_api(audio, api_key=api_key)
                else:
                    # Fallback to Google if no key (though this project seems to have one)
                    text = self.recognizer.recognize_google(audio)
                
                print(f"You said: {text}")
                return text
            except sr.WaitTimeoutError:
                print("No speech detected.")
                return None
            except sr.UnknownValueError:
                print("Could not understand audio.")
                return None
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                return None

    def run(self):
        """
        Starts the interview loop.
        """
        print("Starting Mock Interview Agent...")
        print("Say 'exit' or 'stop' to end the interview.")
        
        # Initial greeting
        initial_question = "Tell me about yourself and your background in software engineering."
        print(f"\nAgent: {initial_question}")
        self.history += f"Agent: {initial_question}\n"
        
        while True:
            # 1. Listen
            user_input = self.listen()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "stop", "quit", "bye"]:
                print("Ending interview. Good luck!")
                break
            
            # 2. Update History
            self.history += f"Candidate: {user_input}\n"
            
            # 3. Generate Response
            chain = self.prompt | self.llm | StrOutputParser()
            response = chain.invoke({"history": self.history, "answer": user_input})
            
            # 4. Output Response
            print(f"\nAgent: {response}")
            
            # 5. Update History with Agent response
            self.history += f"Agent: {response}\n"
            
    def get_response(self, history: str, candidate_answer: str) -> str:
        """
        Generate interviewer's response based on conversation history and candidate's answer.
        This method is designed for Streamlit integration.

        Args:
             history (str): The full conversation history so far
             candidate_answer (str): The candidate's latest answer

        Returns:
            str: The interviewer's response (feedback + next question)
        """
        try:
            chain = self.prompt | self.llm | StrOutputParser()
            response = chain.invoke(
                "history": history, "answer": candidate_answer}, config={"callbacks": [self.langfuse_handler]})
            return response
        except Exception as e:
            return f"I apologize, I encountered an error: {str(e)}. Let's continue - could you tell me more about your experience?"

if __name__ == "__main__":
    agent = InterviewAgent()
    agent.run()
