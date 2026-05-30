
from loguru import logger

from src.graph import AgenticRAG

if __name__ == "__main__":
    logger.info("..........starting runtime.........")

    graph_app = AgenticRAG()

    while True:
        try:
            q = input("Enter your query or 'exit' to quit: ")
            if q.lower() == 'exit':
                break
            response = graph_app.run_graph(query=q)
            print(f"{'='*50}")
            print(f"cache_hit: {response.get('cache_hit')}")
            print(f"answer: {response.get('answer')}")
        
        except Exception as e:
            logger.error(e)

        
    logger.info("..........clossing runtime.........")
    
    

