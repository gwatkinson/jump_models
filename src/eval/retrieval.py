# See https://torchmetrics.readthedocs.io/en/stable/retrieval/hit_rate.html

from torchmetrics import (
    RetrievalHitRate,
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalRecall,
    RetrievalRPrecision,
)

RETRIEVAL_METRICS = {
    "hit_rate": RetrievalHitRate,
    "mrr": RetrievalMRR,
    "ndcg": RetrievalNormalizedDCG,
    "precision": RetrievalPrecision,
    "recall": RetrievalRecall,
    "r-precision": RetrievalRPrecision,
}
