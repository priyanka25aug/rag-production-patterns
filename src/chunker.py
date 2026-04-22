"""
Hierarchical chunker: preserves document structure (sections, paragraphs)
instead of blindly splitting on token windows.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    text: str
    chunk_id: str
    level: str  # "section" | "paragraph" | "sentence"
    parent_id: str
    doc_id: str
    metadata: dict


class HierarchicalChunker:
    def __init__(self, max_section_tokens: int = 512, max_para_tokens: int = 128):
        self.max_section_tokens = max_section_tokens
        self.max_para_tokens = max_para_tokens

    def chunk(self, text: str, doc_id: str = "doc") -> List[Chunk]:
        chunks = []
        sections = self._split_sections(text)

        for s_idx, section in enumerate(sections):
            section_id = f"{doc_id}-s{s_idx}"
            section_tokens = len(section.split())

            if section_tokens <= self.max_section_tokens:
                chunks.append(Chunk(
                    text=section,
                    chunk_id=section_id,
                    level="section",
                    parent_id=doc_id,
                    doc_id=doc_id,
                    metadata={"token_count": section_tokens, "section_index": s_idx},
                ))
            else:
                paragraphs = self._split_paragraphs(section)
                for p_idx, para in enumerate(paragraphs):
                    para_id = f"{section_id}-p{p_idx}"
                    para_tokens = len(para.split())
                    chunks.append(Chunk(
                        text=para,
                        chunk_id=para_id,
                        level="paragraph",
                        parent_id=section_id,
                        doc_id=doc_id,
                        metadata={"token_count": para_tokens, "section_index": s_idx, "para_index": p_idx},
                    ))

        return chunks

    def _split_sections(self, text: str) -> List[str]:
        import re
        sections = re.split(r'\n#{1,3} |\n\n\n+', text)
        return [s.strip() for s in sections if s.strip()]

    def _split_paragraphs(self, text: str) -> List[str]:
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if p.strip()]
