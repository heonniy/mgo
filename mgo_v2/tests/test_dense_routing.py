"""Differential test for the opt-in decode routing weight kernel."""

import unittest
import torch

from mgo_v2.dense_routing import dense_decode_weights


class DenseRoutingTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_matches_scatter(self):
        for batch in (1, 16, 64):
            with self.subTest(batch=batch):
                gen = torch.Generator(device='cuda').manual_seed(812)
                selected = torch.randperm(128, device='cuda')[:8].repeat(batch, 1).contiguous()
                # Intentional collisions exercise substitution's FP32 accumulation.
                targets = torch.randint(0, 32, (128,), device='cuda', generator=gen)
                weights = torch.rand((batch, 8), device='cuda', dtype=torch.bfloat16, generator=gen)
                expected = torch.zeros((batch, 128), device='cuda', dtype=torch.float32)
                expected.scatter_add_(1, targets[selected], weights.float())
                expected = expected.to(weights.dtype)
                actual = dense_decode_weights(selected, weights, targets)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
