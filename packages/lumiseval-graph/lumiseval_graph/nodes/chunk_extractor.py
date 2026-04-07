# Run smoke test:
#   python -m lumiseval_graph.nodes.chunking
"""
Chunk Extractor Node — splits a generation string into semantically coherent
chunks using semchunk.

Mirrors the structure of ClaimExtractorNode so the chunking step is a
first-class pipeline node rather than a bare function call in graph.py.

No LLM calls — cost is always $0.
"""

import hashlib

import semchunk

from lumiseval_graph.nodes.base import BaseNode
from lumiseval_core.constants import CHUNK_MIN_TOKENS_FOR_SPLIT, GENERATION_CHUNK_SIZE_TOKENS
from lumiseval_core.types import Chunk, ChunkEstimate, Item
from lumiseval_core.utils import _count_tokens

from lumiseval_graph.log import get_node_logger

log = get_node_logger("chunk")

_MIN_TOKENS_FOR_SPLIT = CHUNK_MIN_TOKENS_FOR_SPLIT


class ChunkExtractorNode(BaseNode):
    """Splits generation text into semantically coherent chunks.

    Uses semchunk with a configurable token target.  Each chunk is SHA-256
    hashed so downstream dedup can skip identical chunks across repeated runs
    on the same text.

    No LLM calls are made — this node is purely deterministic.
    """

    node_name = "chunk"

    def __init__(self, chunk_size: int = GENERATION_CHUNK_SIZE_TOKENS) -> None:
        self.chunk_size = chunk_size

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} chunk_size={self.chunk_size}>"

    def run(self, item: Item) -> ChunkPayload:  # type: ignore[override]
        """Split ``text`` into semantic chunks.

        If the text is under ``_MIN_TOKENS_FOR_SPLIT`` tokens it is returned
        as a single chunk without splitting.

        Args:
            text: The generation string to split.

        Returns:
            Ordered list of Chunk objects with char offsets and SHA-256 hashes.
        """
        text = item.text
        if item.tokens == 0.0:
            num_tokens = _count_tokens(text)
        else:
            num_tokens = item.tokens

        if num_tokens < _MIN_TOKENS_FOR_SPLIT:
            log.info("text below split threshold — returning as single chunk")
            return ChunkPayload(
                chunks=[
                    Chunk(
                        index=0,
                        item=Item(
                            text=text,
                            tokens=num_tokens
                        )
                        char_start=0,
                        char_end=len(text),
                        sha256=hashlib.sha256(text.encode()).hexdigest(),
                    )
                ],
                tokens=num_tokens,
                num_chunks=1,
                cost_meta=self.estimate(input_tokens=num_tokens, output_tokens=num_tokens)
            )

        chunker = semchunk.chunkerify(_count_tokens, self.chunk_size)
        raw_chunks: list[str] = list(chunker(text))  # type: ignore[arg-type]

        chunks: list[Chunk] = []
        cursor = 0
        for i, chunk_str in enumerate(raw_chunks):
            start = text.find(chunk_str, cursor)
            if start == -1:
                start = cursor
            end = start + len(chunk_str)
            chunks.append(
                Chunk(
                    index=i,
                    item=Item(
                        text=chunk_str,
                        tokens=_count_tokens(chunk_str)
                    ),
                    char_start=start,
                    char_end=end,
                    sha256=hashlib.sha256(chunk_str.encode()).hexdigest(),
                )
            )
            cursor = end

        log.success(f"{len(chunks)} chunk(s) produced  (chunk_size={self.chunk_size} tok)")
        return ChunkArtifacts(
            chunks=chunks,
            cost_meta=self.estimate(input_tokens=chunk_tokens, output_tokens=chunk_tokens)
        )

    def estimate(self, input_tokens: float, output_tokens: float) -> EstimatePayload:  # type: ignore[override]
        # run_result = self.run(input_payload)
        return EstimatePayload(
            input_tokens=0.0,
            output_tokens=0.0,
            cost=0.0,
        )




# ── Manual smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Chunk a representative long generation and print each chunk with its
    index, char offsets, token count, and SHA-256 hash.

    Run with:
        uv run python -m lumiseval_graph.nodes.chunking
    """
    from pprint import pprint

    TEXT = (
        "The Transformer, introduced in the 2017 paper 'Attention Is All You Need' by Vaswani "
        "et al., is a deep learning architecture built entirely around the self-attention "
        "mechanism. Unlike recurrent neural networks (RNNs) and their variants — LSTMs and "
        "GRUs — the Transformer processes entire input sequences in parallel rather than "
        "sequentially, making it dramatically faster to train on modern hardware.\n\n"
        "At its core, a Transformer consists of an encoder and a decoder (though many modern "
        "models like BERT use only the encoder and GPT uses only the decoder). Each layer "
        "contains two sublayers: a multi-head self-attention mechanism and a position-wise "
        "feed-forward network, both wrapped with residual connections and layer normalization.\n\n"
        "Self-attention allows each token in the sequence to attend to every other token, "
        "computing a weighted sum of value vectors based on the compatibility of query and key "
        "vectors. Multi-head attention runs several attention operations in parallel, each in a "
        "lower-dimensional subspace, then concatenates and projects the results.\n\n"
        "Because there is no recurrence, the model has no inherent notion of position. "
        "Positional encodings — either fixed sinusoidal functions or learned embeddings — are "
        "added to the input embeddings to inject sequence order information."
    )

    node = ChunkExtractorNode(chunk_size=GENERATION_CHUNK_SIZE_TOKENS)
    print(repr(node))
    chunks = node.run(TEXT)

    print(f"\n{len(chunks)} chunk(s):\n")
    for c in chunks:
        tok = _count_tokens(c.text)
        print(f"  [{c.index}] chars={c.char_start}:{c.char_end}  tokens={tok}  sha256={c.sha256[:12]}…")
        pprint(c.text)
        print()
