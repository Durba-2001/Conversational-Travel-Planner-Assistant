import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
api_key = os.getenv("GOOGLE_API_KEY")
MongoDB_url=os.environ.get("MONGODB_URI")
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "secret")
ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", 60))