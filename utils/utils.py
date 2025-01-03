import io
from PIL import Image


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


def save_grpah_as_image(graph, out_path):
    graph_png = graph.get_graph().draw_mermaid_png()
    image = Image.open(io.BytesIO(graph_png))
    image.save(out_path, format="PNG")
