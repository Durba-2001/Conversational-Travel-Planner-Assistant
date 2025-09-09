# src/auth/router.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from datetime import datetime, timezone, timedelta
from jose import jwt, JWTError
from hashlib import sha256
from src.database.session import get_db
from src.auth.models import UserCreate, UserResponse, UserSchema
from src.config import SECRET_KEY,ALGORITHM,ACCESS_TOKEN_EXPIRE_MINUTES


router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# -------------------------
# Utility functions
# -------------------------
def hash_password(password: str) -> str:
    """Hash password using SHA256."""
    return sha256(password.encode()).hexdigest()

def create_access_token(username: str, expires_delta: timedelta = None):
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

async def authenticate_user(db, username: str, password: str):
    user = await db["users"].find_one({"username": username})
    if not user or user["hashed_password"] != hash_password(password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return UserSchema(**user)

async def get_current_user(token: str = Depends(oauth2_scheme), db=Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        user = await db["users"].find_one({"username": username})
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        return UserSchema(**user)
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

# -------------------------
# Routes
# -------------------------
@router.post("/register", response_model=UserResponse)
async def register(user: UserCreate, db=Depends(get_db)):
    existing = await db["users"].find_one({"username": user.username})
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User already exists")
    hashed_pw = hash_password(user.password)
    user_doc = UserSchema(username=user.username, hashed_password=hashed_pw, created_at=datetime.now(timezone.utc))
    await db["users"].insert_one(user_doc.model_dump())
    return UserResponse(username=user_doc.username, created_at=user_doc.created_at)

@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db=Depends(get_db)):
    user = await authenticate_user(db, form_data.username, form_data.password)
    token = create_access_token(user.username)
    return {"access_token": token, "token_type": "bearer"}
