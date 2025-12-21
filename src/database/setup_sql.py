import os
import logging
from sqlalchemy import create_engine, MetaData
from dotenv import load_dotenv

load_dotenv()

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_uri():
    """
    Constructs the database URI from environment variables or defaults to SQLite.
    """
    db_type = os.getenv("DB_TYPE", "sqlite")
    
    if db_type == "postgres":
        user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD", "postgres")
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "postgres")
        return f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
    
    # Default to SQLite
    # Use the processed jobs database by default
    default_db_path = os.path.join("data", "processed", "jobs.db")
    
    # Allow overriding via environment variable
    db_path = os.getenv("SQLITE_DB_PATH", default_db_path)
    
    # Ensure the directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        
    return f"sqlite:///{db_path}"

def verify_db_connection():
    """
    Verifies that the database connection works.
    """
    uri = get_db_uri()
    try:
        engine = create_engine(uri)
        with engine.connect() as connection:
            logger.info(f"Successfully connected to database at {uri}")
            
            # Optional: Print table names
            metadata = MetaData()
            metadata.reflect(bind=engine)
            logger.info(f"Existing tables: {list(metadata.tables.keys())}")
            
        return True
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return False

if __name__ == "__main__":
    verify_db_connection()
