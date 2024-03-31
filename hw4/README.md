



1) Take SberQuAD dataset:
https://huggingface.co/datasets/sberquad
2) Train model for searching documents with correct answer
3) Use all documents in corresponding dataset (test/val) for searching
4) Training example: SentenceTransformers:
https://www.sbert.net/docs/training/overview.html
5) Evaluate Recall@5, MRR, MAP, NDCG@10 on test dataset: 
https://lightning.ai/docs/torchmetrics/stable/retrieval/recall.html
https://lightning.ai/docs/torchmetrics/stable/retrieval/mrr.html
https://lightning.ai/docs/torchmetrics/stable/retrieval/map.html
https://lightning.ai/docs/torchmetrics/stable/retrieval/normalized_dcg.html