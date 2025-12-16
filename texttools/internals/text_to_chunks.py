import re


def text_to_chunks(text: str, size: int, overlap: int) -> list[str]:
    separators = ["\n\n", "\n", " ", ""]
    is_separator_regex = False
    keep_separator = True  # Equivalent to 'start'
    length_function = len
    strip_whitespace = True
    chunk_size = size
    chunk_overlap = overlap

    def _split_text_with_regex(
        text: str, separator: str, keep_separator: bool
    ) -> list[str]:
        if not separator:
            return [text]
        if not keep_separator:
            return re.split(separator, text)
        _splits = re.split(f"({separator})", text)
        splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
        if len(_splits) % 2 == 0:
            splits += [_splits[-1]]
        return [_splits[0]] + splits if _splits[0] else splits

    def _join_docs(docs: list[str], separator: str) -> str | None:
        text = separator.join(docs)
        if strip_whitespace:
            text = text.strip()
        return text if text else None

    def _merge_splits(splits: list[str], separator: str) -> list[str]:
        separator_len = length_function(separator)
        docs = []
        current_doc = []
        total = 0
        for d in splits:
            len_ = length_function(d)
            if total + len_ + (separator_len if current_doc else 0) > chunk_size:
                if total > chunk_size:
                    pass  # Optionally add warning
                if current_doc:
                    doc = _join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    while total > chunk_overlap or (
                        total + len_ + (separator_len if current_doc else 0)
                        > chunk_size
                        and total > 0
                    ):
                        total -= length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += len_ + (separator_len if len(current_doc) > 1 else 0)
        doc = _join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    def _split_text(text: str, separators: list[str]) -> list[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            separator_ = _s if is_separator_regex else re.escape(_s)
            if not _s:
                separator = _s
                break
            if re.search(separator_, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break
        separator_ = separator if is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, separator_, keep_separator)
        _separator = "" if keep_separator else separator
        good_splits = []
        for s in splits:
            if length_function(s) < chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged_text = _merge_splits(good_splits, _separator)
                    final_chunks.extend(merged_text)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = _split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if good_splits:
            merged_text = _merge_splits(good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks

    return _split_text(text, separators)
