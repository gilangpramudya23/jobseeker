from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
import os
import logging
from dotenv import load_dotenv

# Konfigurasi Logging agar kita bisa lihat error di Streamlit Cloud Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class SQLAgent:
    def __init__(self, db_path: str = None):
        # 1. Dapatkan Path Absolut dari Root Project
        # file ini ada di: src/agents/sql_agent.py
        current_file_path = os.path.abspath(__file__)
        agents_dir = os.path.dirname(current_file_path) # src/agents
        src_dir = os.path.dirname(agents_dir) # src
        project_root = os.path.dirname(src_dir) # root
        
        if db_path is None:
            # Mengarah ke: root/data/processed/jobs.db
            db_path = os.path.join(project_root, 'data', 'processed', 'jobs.db')
        
        # 2. Normalisasi path untuk Linux (Streamlit Cloud)
        db_path = os.path.abspath(db_path)
        
        # DEBUG: Muncul di log Streamlit Cloud untuk memastikan path benar
        logger.info(f"Mencoba mengakses database di: {db_path}")

        # 3. Cek apakah file benar-benar ada sebelum koneksi
        if not os.path.exists(db_path):
            logger.error("FILE DATABASE TIDAK DITEMUKAN!")
            # Tampilkan list file di folder tersebut untuk memudahkan debug
            parent_dir = os.path.dirname(db_path)
            if os.path.exists(parent_dir):
                logger.info(f"Isi folder {parent_dir}: {os.listdir(parent_dir)}")
            else:
                logger.error(f"Folder {parent_dir} juga tidak ditemukan!")
            
            raise FileNotFoundError(f"Database tidak ditemukan di {db_path}")

        # 4. Gunakan URI yang benar (4 slash untuk absolut di Linux)
        # sqlite:////mount/src/jobseeker/data/processed/jobs.db
        db_uri = f"sqlite:///{db_path}"
        
        self.db = SQLDatabase.from_uri(db_uri)
        self.llm = ChatOpenAI(
            temperature=0, 
            model="gpt-4o-mini", 
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        
        self.agent_executor = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            verbose=True,
            handle_parsing_errors=True
        )

    def run(self, query: str) -> str:
        try:
            # Hapus callback langfuse sementara untuk memastikan tidak ada error lain
            response = self.agent_executor.invoke({"input": query})
            if isinstance(response, dict) and "output" in response:
                return response["output"]
            return str(response)
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return f"Error database: {str(e)}"

if __name__ == "__main__":
    agent = SQLAgent()
    print(agent.run("Berapa jumlah data di database?"))
