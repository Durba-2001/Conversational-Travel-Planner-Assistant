from pymongo import AsyncMongoClient   # Import AsyncMongoClient to connect to MongoDB asynchronously
from loguru import logger
from src.config import MongoDB_url
client=None
# Define an async function to get a MongoDB database
async def get_db(db_name="travel_planner_db"):
  global client

  try:
      if not client:
        logger.info("Connecting to MongoDB...")   # Log info before connecting
        client = AsyncMongoClient(MongoDB_url)    # Create an asynchronous MongoDB client using the connection URL
        logger.info(f"Connected to database: {db_name}")
      db = client[db_name]                      # Get the database with the specified name     
      return db                   # Return the database object for future used       
  except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")  # Log any errors
        return None
                    
# Function to close DB connection
def close_db():
    global client
    if client:
        client.close()
        logger.info("MongoDB connection closed")
        client = None   # Reset client so it can reconnect next time