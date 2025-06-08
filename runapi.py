from api.app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("runapi:app", host="0.0.0.0", port=18200, reload=True) 
