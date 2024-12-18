from fastapi import APIRouter

router = APIRouter()

@router.get("/healthcheck", status_code=200)
def health_check():
    """
    Endpoint to check the API connection.
    """
    return {"status": "healthy"}