from fastapi import FastAPI
from pydantic import BaseModel
from detector import check


app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Prompt(BaseModel):
    text: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/check")
def check_prompt(p: Prompt):
    result = check(p.text)
    return {
        "blocked": result["action"] == "block",
        "layer": f"rule={result['rule']}",
        "score": result["prob"],
        "hits": result["hits"],
    }

