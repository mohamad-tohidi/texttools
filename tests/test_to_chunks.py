from texttools.core.engine import to_chunks


def test_single_chunk():
    text = "Short text"
    chunks = to_chunks(text, size=100, overlap=0)
    assert len(chunks) == 1
    assert chunks[0] == "Short text"


def test_empty_text():
    chunks = to_chunks("", size=10, overlap=0)
    assert len(chunks) == 0
