from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
import os
import logging
from dotenv import load_dotenv

# Import Langfuse secara aman (opsional)
try:
    from langfuse.langchain import CallbackHandler
    HAS_LANGFUSE = True
except ImportError:
    HAS_LANGFUSE = False

load_dotenv()
logger = logging.getLogger(__name__)

class SQLAgent:
    def __init__(self, db_path: str = None):
        """
        Initialize the SQL Agent dengan path yang kompatibel untuk Cloud & Local.
        """
        # 1. Tentukan Root Project secara dinamis
        current_dir = os.path.dirname(os.path.abspath(__file__)) # src/agents
        project_root = os.path.abspath(os.path.join(current_dir, '../../'))
        
        # 2. Setup Path Database
        if db_path is None:
            # Default ke data/processed/jobs.db dari root
            db_path = os.path.join(project_root, 'data', 'processed', 'jobs.db')
        
        # Pastikan path absolut
        if not os.path.isabs(db_path):
            db_path = os.path.join(project_root, db_path)

        # 3. Validasi Keberadaan File (Penting untuk Debugging di Cloud)
        if not os.path.exists(db_path):
            logger.error(f"DATABASE TIDAK DITEMUKAN: {db_path}")
            # Jika di Cloud, kita beri peringatan jelas
            raise FileNotFoundError(f"File database tidak ditemukan di: {db_path}. Pastikan folder 'data/processed' sudah di-upload ke GitHub.")

        # 4. Konstruksi SQLite URI yang aman
        # sqlite:////path/to/file (4 slash) untuk path absolut di Linux/Cloud
        # sqlite:///C:\path (3 slash) untuk Windows
        if os.name == 'nt': # Windows
            db_uri = f"sqlite:///{db_path}"
        else: # Linux/Streamlit Cloud
            db_uri = f"sqlite:////{db_path}"
        
        logger.info(f"Connecting to database at: {db_uri}")
        
        self.db = SQLDatabase.from_uri(db_uri)
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        
        self.agent_executor = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            verbose=True,
            handle_parsing_errors=True
        )
        
        # Initialize Langfuse CallbackHandler hanya jika terinstal
        self.langfuse_handler = CallbackHandler() if HAS_LANGFUSE else None

    def run(self, query: str) -> str:
        """
        Menjalankan query melalui SQL Agent.
        """
        try:
            # Siapkan config callbacks jika langfuse tersedia
            config = {"callbacks": [self.langfuse_handler]} if self.langfuse_handler else {}
            
            response = self.agent_executor.invoke({"input": query}, config=config)
            
            if isinstance(response, dict) and "output" in response:
                return response["output"]
            return str(response)
        except Exception as e:
            logger.error(f"Error executing SQL query: {str(e)}")
            return f"Maaf, terjadi kesalahan saat mengakses database: {str(e)}"

if __name__ == "__main__":
    # Test lokal
    try:
        agent = SQLAgent()
        print(agent.run("Berapa total lowongan kerja yang tersedia?"))
    except Exception as e:
        print(f"Gagal inisialisasi: {e}")
