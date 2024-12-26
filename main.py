import os
from dotenv import load_dotenv
from llm import Llm


llm = Llm(
    data_path=os.getenv("DATA_PATH"),
    rag_prompt_id="daethyra/rag-prompt",
    evaluate_prompt_id="jisujiji/rag-prompt",
)

question = "比特幣為了解決什麼問題?"

generated_answer = llm.rag_chain(question)

evaluation_score = llm.evaluate(
    question=question,
    answer=generated_answer,
    topic="比特幣",
    criteria="相關性和準確性",
    examples="比特幣是一種去中心化的電子現金系統",
).content

print(evaluation_score)
