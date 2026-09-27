import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


entropy = load_module("entropy_probe_cpu", "src/hip_attn/utils/entropy_probe.py")
metrics = load_module("ppl_metrics_cpu", "hip-research/src/hip_research/utils/ppl_metrics.py")
aggregate = load_module("aggregate_cpu", "aggregate_summaries.py")


class EntropyTests(unittest.TestCase):
    def test_future_key_cannot_change_earlier_query(self):
        q = torch.ones(1, 4, 2, 3)
        k = torch.zeros(1, 4, 1, 3)
        before = entropy.causal_probe_entropy(q, k)
        k[:, 3] = 100.
        after = entropy.causal_probe_entropy(q, k)
        torch.testing.assert_close(before[:, :3], after[:, :3])
        self.assertFalse(torch.equal(before[:, 3], after[:, 3]))

    def test_uniform_entropy_is_normalized_by_visible_prefix(self):
        result = entropy.causal_probe_entropy(torch.zeros(1, 4, 1, 2), torch.zeros(1, 4, 1, 2))
        torch.testing.assert_close(result[0, :, 0], torch.tensor([0., 1., 1., 1.]))

    def test_gqa_equals_explicit_head_repeat(self):
        q, k = torch.randn(2, 4, 4, 3), torch.randn(2, 6, 2, 3)
        torch.testing.assert_close(entropy.causal_probe_entropy(q, k), entropy.causal_probe_entropy(q, k.repeat_interleave(2, dim=2)))

    def test_single_token_decode_and_short_probe_are_finite(self):
        self.assertAlmostEqual(entropy.estimate_entropy_norm(torch.zeros(1, 1, 2, 3), torch.zeros(1, 8, 1, 3)), 1., places=6)
        result = entropy.causal_probe_entropy(torch.randn(1, 8, 2, 3), torch.randn(1, 8, 1, 3), probe_k=2)
        self.assertTrue(torch.isfinite(result).all())


class MetricTests(unittest.TestCase):
    def test_unequal_window_sizes_are_token_weighted(self):
        metric = metrics.PerplexityAccumulator()
        metric.add(math.log(2), 9)
        metric.add(math.log(8), 1)
        self.assertAlmostEqual(metric.result()["ppl"], math.exp((9*math.log(2)+math.log(8))/10))
        self.assertEqual(metric.result()["valid_tokens"], 10)

    def test_empty_and_nonfinite_metrics_fail(self):
        metric = metrics.PerplexityAccumulator()
        with self.assertRaises(ValueError): metric.result()
        with self.assertRaises(ValueError): metric.add(float("nan"), 10)

    def test_aggregate_requires_new_metric_and_does_not_average_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"ppl_pg19_adahip_llama_s32768_k512_a1p0_p2p0_20260928_000000.summary.txt"
            text = "step 1 PPL: 20.0, PPL_WST: nan, PPL_BST: nan, 2.0 sec\nstep 2 PPL: 10.0, PPL_WST: nan, PPL_BST: nan, 4.0 sec\n"
            path.write_text(text)
            self.assertIsNone(aggregate.parse_summary_file(path))
            import json
            text += "METRIC_JSON: " + json.dumps({"metric_version":"token_weighted_v1", "total_nll":math.log(3)*10, "valid_tokens":10})
            path.write_text(text)
            row=aggregate.parse_summary_file(path)
            self.assertAlmostEqual(row.ppl_mean, 3.)
            self.assertEqual(row.latency_mean_sec, 3.)

    def test_speedup_does_not_mix_model_names(self):
        a=aggregate.RunRow("pg19", "fa2", 32, None, None, None, 1., 4., model="a")
        b=aggregate.RunRow("pg19", "adahip", 32, 512, 1., 2., 1., 2., model="b")
        aggregate.compute_speedup([a,b])
        self.assertIsNone(b.speedup)


if __name__ == "__main__": unittest.main()
