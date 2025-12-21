from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler

load_dotenv()

class SQLAgent:
    def __init__(self, db_path: str = None):
        """
        Initialize the SQL Agent.
        
        Args:
            db_path (str, optional): The path to the SQLite database file.
                                   If None, defaults to 'data/processed/jobs.db' in the project root.
        """
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '../../'))
        
        if db_path is None:
            db_path = os.path.join(project_root, 'data', 'processed', 'jobs.db')
        
        # Ensure the path is absolute
        if not os.path.isabs(db_path):
            db_path = os.path.join(project_root, db_path)

        # Construct SQLite URI
        # on windows slashes might be an issue if not handled, but os.path.join usually handles it.
        # SQLAlchemy expects sqlite:///path/to/db
        db_uri = f"sqlite:///{db_path}"
        
        self.db = SQLDatabase.from_uri(db_uri)
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        
        self.agent_executor = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            verbose=True,
            handle_parsing_errors=True
        )
        
        # Initialize Langfuse CallbackHandler
        self.langfuse_handler = CallbackHandler()

    def run(self, query: str) -> str:
        """
        Run the agent with the given query.
        
        Args:
            query (str): The natural language query to ask the database.
            
        Returns:
            str: The agent's response.
        """
        try:
            response = self.agent_executor.invoke({"input": query}, config={"callbacks": [self.langfuse_handler]})
            # The response format might vary based on the agent version, 
            # usually it returns a dict with "output" key or just the string.
            if isinstance(response, dict) and "output" in response:
                return response["output"]
            return str(response)
        except Exception as e:
            return f"Error executing query: {str(e)}"

if __name__ == "__main__":
    # Test with a specific path if needed, or default
    # db_path = os.path.join("data", "processed", "jobs.db")
    agent = SQLAgent()
    # print(agent.run("How many rows are in the main table?"))