
import json

from datasets import Dataset
from src.graph import AgenticRAG
from loguru import logger

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

import os

from dotenv import load_dotenv

load_dotenv()


def evaluation(n: int):
    
    logger.info(".......Evaluation is running.......")
    engine = AgenticRAG()
    api_key = os.environ['GROQ_API_KEY']
    llm = LangchainLLMWrapper(ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key
    ))
    embedding = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    ))

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    with open('data/golden/policy.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i == n:
                break
            data["question"].append(json.loads(line).get("question"))
            data["ground_truth"].append(json.loads(line).get("answer"))

  
    for i, q in enumerate(data.get('question')):
        if i == n:
            break
        response = engine.run_graph(query=q)
        data["answer"].append(response.get("answer"))
        data["contexts"].append(response.get("contexts"))
        
    data_set = Dataset.from_dict(data)
    
    scores = evaluate(
        dataset=data_set,
        metrics=[faithfulness,answer_relevancy],
        llm=llm,
        embeddings=embedding,
        run_config=RunConfig(max_workers=2, timeout=120)
    )
    return data, scores

if __name__ == "__main__":

    data, scores = evaluation(n=13)
    logger.info(f"generating eval response....")
    for q, a, c, gt in zip(data["question"], data["answer"], data["contexts"], data["ground_truth"]):
        print("QUESTION:", q)
        print("ANSWER:", a)
        print("CONTEXT LENGTH:", len(c))
        print("GROUND TRUTH:", gt)

    print(f"SCORES: {scores}")

   



