"""Token-weighted language-modeling metrics with explicit provenance."""
import math


class PerplexityAccumulator:
    def __init__(self):
        self.total_nll = 0.0
        self.valid_tokens = 0

    def add(self, mean_nll, valid_tokens):
        if not math.isfinite(mean_nll) or valid_tokens <= 0:
            raise ValueError("NLL must be finite and valid_tokens positive")
        self.total_nll += float(mean_nll) * int(valid_tokens)
        self.valid_tokens += int(valid_tokens)

    def result(self):
        if not self.valid_tokens:
            raise ValueError("No valid next-token targets were evaluated")
        return {"metric_version": "token_weighted_v1", "ppl": math.exp(self.total_nll / self.valid_tokens),
                "total_nll": self.total_nll, "valid_tokens": self.valid_tokens}
