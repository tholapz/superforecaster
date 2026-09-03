from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings

_bearer = HTTPBearer()


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(_bearer),
) -> str:
    if credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=401, detail="Invalid API key", headers={"WWW-Authenticate": "Bearer"}
        )
    return credentials.credentials
