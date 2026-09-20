"""CPU-only layout regressions against the original, independently executed graph.

No checkpoints, Core ML conversion, accelerator devices, or timing are involved.
The body oracle is DecisionModel.forward, with a hook observing its final head.
"""

import pytest
import torch
from torch import nn

from experiments.ane_engineering.model import ChannelNorm, ConvAttention, ConvBody
from laya_coreml.torch_model import Attention, DecisionModel, HeadAttention


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_source():
    # Different full/local RoPE bases and a short window make layout mistakes
    # visible even though this model has only 32 hidden channels and 17 tokens.
    cfg = {
        "model_type": "modernbert",
        "hidden_size": 32,
        "intermediate_size": 48,
        "vocab_size": 53,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "local_attention": 4,
        "layer_types": ["full_attention", "sliding_attention", "full_attention"],
        "norm_bias": True,
        "attention_bias": True,
        "mlp_bias": True,
        "norm_eps": 3e-4,
        "rope_parameters": {
            "full_attention": {"rope_theta": 100.0},
            "sliding_attention": {"rope_theta": 10000.0},
        },
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260920)
        model = DecisionModel(cfg, {"head_layers": 2, "act_costs": {"defer": 0.5}}, 17)
        with torch.no_grad():
            # HeadAttention's packed projection parameters are normally loaded
            # from a checkpoint and otherwise have uninitialized storage.
            for name, parameter in model.named_parameters():
                if "in_proj" in name:
                    nn.init.normal_(parameter, std=0.04)
            for module in model.modules():
                if isinstance(module, nn.LayerNorm):
                    width = module.weight.numel()
                    module.weight.copy_(torch.linspace(0.4, 1.6, width))
                    module.bias.copy_(torch.linspace(-0.25, 0.35, width))
    return model.eval()


def example_batch():
    ids = torch.arange(3 * 17).reshape(3, 17) % 53
    valid = torch.arange(17)[None, :] < torch.tensor([17, 12, 9])[:, None]
    return {
        "input_ids": ids,
        "attention_mask": valid,
        # Deliberately unsorted, with different option counts and repeated pad
        # positions: a permutation error must not disappear in a sorted gather.
        "marker_pos": torch.tensor([[11, 4, 0, 0], [2, 9, 1, 0], [8, 5, 1, 3]]),
        "marker_mask": torch.tensor(
            [[True, True, False, False], [True, True, True, False], [True, True, True, True]]
        ),
        "qtype": torch.tensor([0, 1, 2]),
    }


def body_inputs(source, batch):
    """Construct the external BC1L contract directly in key/query coordinates."""
    ids, valid = batch["input_ids"], batch["attention_mask"]
    batch_size, length = ids.shape
    count = batch["marker_pos"].shape[1]
    full = torch.full((batch_size, length, 1, length), -1e4)
    local = torch.full_like(full, -1e4)
    selectors = torch.zeros(batch_size, length, 1, count)
    for row in range(batch_size):
        for key in range(length):
            for query in range(length):
                if valid[row, key]:
                    full[row, key, 0, query] = 0
                    if not valid[row, query] or abs(key - query) <= source.encoder.window // 2:
                        local[row, key, 0, query] = 0
        for slot, position in enumerate(batch["marker_pos"][row].tolist()):
            selectors[row, position, 0, slot] = 1
    return (
        source.encoder.embeddings.tok_embeddings(ids).transpose(1, 2).unsqueeze(2),
        full,
        local,
        source.type_emb(batch["qtype"])[:, :, None, None],
        selectors,
    )


def reference_outputs(source, batch):
    observed = {}

    def capture_head(_module, _args, output):
        observed["pooled"] = output[:, 0].detach().clone()

    handle = source.head.register_forward_hook(capture_head)
    try:
        logits, _ = source(**batch)
    finally:
        handle.remove()
    return logits, observed["pooled"]


def test_channel_norm_keeps_bias_after_scale_and_preserves_epsilon():
    source = nn.LayerNorm(8, eps=3e-4).double()
    with torch.no_grad():
        source.weight.copy_(torch.tensor([0.0, 0.25, 0.5, 2.0, -1.0, 1.5, -0.75, 0.8]))
        source.bias.copy_(torch.linspace(-0.3, 0.4, 8))
    # Small variance makes an accidentally changed epsilon observable. A zero
    # gamma also rejects bias/gamma remapping, while nonzero biases reject
    # (normalized + bias) * gamma.
    x = torch.arange(2 * 7 * 8, dtype=torch.float64).reshape(2, 7, 8).sin() * 0.01
    with torch.inference_mode():
        expected = source(x)
        actual = ChannelNorm(source)(x.transpose(1, 2).unsqueeze(2))
    torch.testing.assert_close(actual.squeeze(2).transpose(1, 2), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("layer_index", [0, 1], ids=["global_rope", "local_rope"])
def test_conv_attention_matches_original_across_heads_and_padding(tiny_source, layer_index):
    batch = example_batch()
    with torch.inference_mode():
        _, full, local, _, _ = body_inputs(tiny_source, batch)
        mask = full if layer_index == 0 else local
        reference_mask = (mask[:, :, 0, :] == 0).transpose(1, 2)[:, None]
        source = tiny_source.encoder.layers[layer_index].attn
        candidate = ConvAttention(source, rope=True, length=17).eval()
        generator = torch.Generator().manual_seed(31415)
        x = torch.randn(3, 17, 32, generator=generator)
        expected = source(x, reference_mask)
        actual = candidate(x.transpose(1, 2).unsqueeze(2), mask)
    torch.testing.assert_close(actual.squeeze(2).transpose(1, 2), expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("reference_attention", ["explicit", "sdpa"])
def test_conv_body_matches_original_logits_and_cls_for_types_and_padding(
    tiny_source, reference_attention
):
    for module in tiny_source.modules():
        if isinstance(module, (Attention, HeadAttention)):
            module.implementation = reference_attention
    candidate = ConvBody(tiny_source, length=17).eval()
    batch = example_batch()
    selected = batch["marker_mask"]
    baseline = None
    with torch.inference_mode():
        for replace_padding in (False, True):
            if replace_padding:
                batch["input_ids"][~batch["attention_mask"]] = 52
            expected_logits, expected_cls = reference_outputs(tiny_source, batch)
            logits, pooled = candidate(*body_inputs(tiny_source, batch))
            logits, pooled = logits[:, 0, 0, :], pooled[:, :, 0, 0]
            # Compare only valid option slots; marker masking belongs to the
            # caller, whereas ConvBody deliberately scores every supplied slot.
            torch.testing.assert_close(
                logits[selected], expected_logits[selected], atol=1e-5, rtol=5e-5
            )
            torch.testing.assert_close(pooled, expected_cls, atol=1e-5, rtol=5e-5)
            if baseline is None:
                baseline = logits[selected].clone(), pooled.clone()
            else:
                # Masked token values must not leak into valid decisions or CLS.
                torch.testing.assert_close(logits[selected], baseline[0], atol=1e-5, rtol=5e-5)
                torch.testing.assert_close(pooled, baseline[1], atol=1e-5, rtol=5e-5)
