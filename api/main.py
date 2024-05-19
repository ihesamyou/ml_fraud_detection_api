import pickle
import pandas as pd
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(
    title="Transaction Fraud Classifier",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

model = pickle.load(
    open('transaction_fraud_pipeline.pkl', 'rb')
)


@app.get("/")
def read_root(text: str = ""):
    return f"For using this API try /predict route. It accepts json with post requests."


class Transaction(BaseModel):
    amount: float
    source_prefix: str
    source_postfix: int
    dest_prefix: str
    dest_postfix: int
    status: str
    agent: str


@app.post("/predict/")
def predict(transactions: List[Transaction]) -> List[str]:
    X = pd.DataFrame([dict(transaction) for transaction in transactions])
    y_pred = model.predict(X)
    return [str(pred) for pred in y_pred]
