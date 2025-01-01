import os
from llm import Llm
from workflow import app


def main():
    llm = Llm(
        data_path=os.getenv("DATA_PATH"),
        rag_prompt_id="daethyra/rag-prompt",
        evaluate_prompt_id="jisujiji/rag-prompt",
        evaluate_docs_prompt_id="teddynote/retrieval-question-grader",
        rewrite_question_prompt_id="efriis/self-rag-question-rewriter",
    )

    question = "比特幣為了解決什麼問題?"
    # question = "比特幣2024重大事件"
    inputs = {
        "question": question,
        "documents": [],
        "generation": "",
        "web_search": "",
        "llm": llm,
    }

    last_state = None
    for state in app.stream(inputs):
        last_state = state

    if "generate" in last_state:
        generated_answer = last_state["generate"]["generation"].content
    elif "summarize_web_result" in last_state:
        generated_answer = last_state["summarize_web_result"]["generation"].content
    else:
        raise KeyError("No valid generation result found in the workflow state.")
    print("=== Generated Answer ===")
    print(generated_answer)

    evaluation_score = llm.evaluate_answer(
        question=question,
        answer=generated_answer,
        topic="比特幣",
        criteria="相關性和準確性",
        examples="比特幣是一種去中心化的電子現金系統",
    ).content
    print("\n=== Evaluation Result ===")
    print(evaluation_score)


if __name__ == "__main__":
    main()
