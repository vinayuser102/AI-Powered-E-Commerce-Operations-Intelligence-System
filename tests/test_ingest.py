from rag_qa_engine.ingest import chunk_text


def test_chunk_text_keeps_all_content():
    text = "First policy paragraph.\n\nSecond policy paragraph.\n\nThird policy paragraph."
    chunks = chunk_text(text, chunk_size=35, overlap=5)
    assert len(chunks) >= 2
    assert "First policy paragraph." in chunks[0]
    assert "Third policy paragraph." in chunks[-1]
