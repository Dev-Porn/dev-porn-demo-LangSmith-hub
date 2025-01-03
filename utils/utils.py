def format_documents(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_web_search(documents: list) -> str:
    """
    將 web_search 返回的文檔列表格式化為一個字串，上下文供模型使用。
    """
    formatted_docs = []
    for i, doc in enumerate(documents):
        formatted_docs.append(
            f"Document {i + 1}:\n" f"URL: {doc['url']}\n" f"Content: {doc['content']}\n"
        )
    return "\n\n".join(formatted_docs)
