from fastapi import APIRouter
from api.v1.endpoints import auth , users , chat 

api_router = APIRouter()

api_router.include_router(
    auth.router, 
    prefix="/auth", 
    tags=["Authentication"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not Found"},
        500: {"description": "Internal Server Error"}
    }
)

api_router.include_router(
    users.router,
    prefix="/users",
    tags=["Users"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not Found"},
        500: {"description": "Internal Server Error"}
    }
)


api_router.include_router(
    chat.router,
    prefix="/chats",
    tags=["Chat"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not Found"},
        500: {"description": "Internal Server Error"}
    }
)