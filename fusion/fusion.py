# fusion.py

def format_for_llm(retrieved_docs, query):
    """
    Advanced fusion: constructs a rich, well-structured prompt for multimodal RAG QA over charts/tables/doc context.
    """
    evidence_blocks = []
    for idx, doc in enumerate(retrieved_docs, 1):
        md = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}
        block = [f"[Evidence {idx}]"]
        if md.get("type"):
            block.append(f"Type: {md['type']}")
        if md.get("title"):
            block.append(f"Title: {md['title']}")
        if md.get("page"):
            block.append(f"Page: {md['page']}")
        if md.get("caption"):
            block.append(f"Caption: {md['caption']}")
        content = getattr(doc, "page_content", None) or doc.get("page_content", "")
        if content:
            block.append(f"Extracted Content:\n{content.strip()}")
        evidence_blocks.append("\n".join(block))

    evidence_str = "\n\n====\n\n".join(evidence_blocks)

    prompt = (
        "You are a highly accurate question-answering assistant for scientific, financial, and medical documents containing charts and tables.\n"
        "Your goal: Given the user question and retrieved evidence from multiple modalities (charts, tables, context), synthesize a clear, fact-based answer.\n"
        "Instructions:\n"
        " - Carefully analyze each evidence block: cross-reference data, chart titles, captions, and context.\n"
        " - Use numerical, tabular, and textual evidence as appropriate.\n"
        " - Clearly cite page numbers or chart/table names when explaining or justifying your answer.\n"
        " - If multiple pieces of evidence contribute, synthesize and reconcile their content.\n"
        " - Use step-by-step reasoning if required to arrive at the answer.\n"
        " - If the answer requires information outside the provided evidence, say so explicitly.\n\n"
        f"User Question:\n{query}\n\n"
        "Retrieved Evidence:\n"
        f"{evidence_str}\n\n"
        "Please provide your detailed answer below, referencing the most relevant evidence and clearly citing page, chart, or table where used."
    )
    return prompt
