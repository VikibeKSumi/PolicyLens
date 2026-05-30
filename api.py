import os
from pydantic import BaseModel

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader

from src.graph import AgenticRAG


load_dotenv()


class QueryRequest(BaseModel):
    query: str


app = FastAPI(title="Policy Analysis")
api_key_header = APIKeyHeader(
    name="X-API-Key"
)

engine = AgenticRAG()
 
@app.post('/ask')
def ask(query: QueryRequest, api_key: str = Security(api_key_header)):
    if api_key == os.environ["AUTH_KEY"]:
        response = engine.run_graph(query.query)
        return response
    else:
        raise HTTPException(401, detail="Invalid API Key")


