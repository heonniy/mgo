import unittest
import torch
from mgo_v2.qwen3_integration import attach_qwen3_runtime, detach_qwen3_runtime


class Qwen3MoeFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(4, 3, bias=False)
        self.top_k = 2
        self.layer_id = 0

    def forward(self, hidden_states):
        return hidden_states * 2, self.gate(hidden_states.flatten(0, 1))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 4)
        self.mlp = Qwen3MoeFixture()

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        return self.mlp(self.embedding(input_ids) if input_ids is not None else inputs_embeds)


class Runtime:
    def forward_layer(self, **kwargs):
        self.rows = kwargs["hidden_states"].shape[0]
        assert kwargs["selected_experts"].shape[0] == self.rows
        assert kwargs["full_router_probs"].shape[0] == self.rows
        return kwargs["hidden_states"] * 2


class TestQwen3Mask(unittest.TestCase):
    def test_padding_is_excluded_and_decode_mask_is_refreshed(self):
        model, runtime = Model(), Runtime()
        attach_qwen3_runtime(model, runtime)
        ids = torch.tensor([[0, 0, 2, 3], [0, 4, 5, 6]])
        mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
        expected = model.embedding(ids) * 2
        output, _ = model(input_ids=ids, attention_mask=mask)
        self.assertEqual(runtime.rows, 5)
        torch.testing.assert_close(output[mask.bool()], expected[mask.bool()])
        self.assertTrue(bool((output[~mask.bool()] == 0).all()))
        model(input_ids=ids[:, -1:], attention_mask=torch.cat((mask, torch.ones(2, 1, dtype=mask.dtype)), 1))
        self.assertEqual(runtime.rows, 2)
        model(input_ids=ids, attention_mask=torch.zeros_like(mask))
        self.assertEqual(runtime.rows, 0)
        self.assertEqual(detach_qwen3_runtime(model), 1)
        self.assertFalse(hasattr(model, "_mgo_v2_mask_hook"))
        torch.testing.assert_close(model(input_ids=ids, attention_mask=mask)[0], expected)

    def test_embeddings_and_positional_mask_arguments(self):
        model, runtime = Model(), Runtime()
        attach_qwen3_runtime(model, runtime)
        ids = torch.tensor([[1, 2, 3]])
        mask = torch.tensor([[0, 1, 1]])
        model(ids, mask)
        self.assertEqual(runtime.rows, 2)
        model(inputs_embeds=model.embedding(ids), attention_mask=mask)
        self.assertEqual(runtime.rows, 2)


if __name__ == "__main__":
    unittest.main()
