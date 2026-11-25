# Retrieval-Augmented Generation

## Overview
Retrieval-augmented generation (RAG) answers questions from user-provided documents. Documents are split into chunks, the chunks are indexed, the most relevant chunks for a question are retrieved, and a language model writes an answer grounded in those passages with citations.

## Document processing
Uploaded PDF, HTML, Markdown and plain-text files are converted to text. Recursive chunking splits on paragraph breaks first, then lines, sentences, clauses and finally words, so that each chunk stays under the size limit while keeping natural boundaries. Consecutive chunks overlap by a few sentences so that facts spanning a boundary are not lost.

## Hybrid search
Sparse retrieval with BM25 scores chunks by exact term matches, weighted by inverse document frequency and normalised by chunk length. It excels at names, codes and rare keywords. Dense retrieval embeds the question and every chunk as vectors and finds nearest neighbours by cosine similarity, which captures paraphrases and synonyms. Hybrid search runs both and fuses the ranked lists. Reciprocal rank fusion gives every chunk a score of the sum of 1 / (k + rank) over the lists, with k typically 60, which needs no score calibration. Alpha fusion instead mixes min-max normalised scores with a weight alpha.

## HNSW
Exact nearest neighbour search compares the query with every vector. Hierarchical Navigable Small World graphs make this approximate and logarithmic. Each vector is inserted into a multi-layer graph; upper layers are sparse long-range links and layer zero contains every node. A search greedily descends through the layers and then runs a beam search of width ef on the bottom layer. The parameters M, ef_construction and ef_search trade memory and build time against recall.

## Reranking
First-stage retrieval scores the question and each chunk independently. A reranker, typically a cross-encoder such as BGE reranker, reads the question and passage together and produces a more accurate relevance score. Because it is expensive it only reorders the top 20 to 50 fused candidates, and the best few passages are placed in the prompt.
