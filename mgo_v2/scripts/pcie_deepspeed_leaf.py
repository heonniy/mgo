"""Untimed Qwen3 MoE setup for native ZeRO-3 parameter offload."""


def mark_qwen3_moe_leaves(model, budget_bytes):
    from deepspeed.utils import set_z3_leaf_modules
    from deepspeed.utils.z3_leaf_module import get_z3_leaf_modules

    modules = set_z3_leaf_modules(model, ['Qwen3MoeSparseMoeBlock'])
    expected = model.config.num_hidden_layers
    assert len(modules) == expected, (len(modules), expected)
    assert set(get_z3_leaf_modules(model)) == set(modules)
    rows = []
    names = {id(module): name for name, module in model.named_modules()}
    for module in modules:
        assert len(module.experts) == model.config.num_experts == 128
        parameters = list(module.parameters())
        nbytes = sum(getattr(p, 'ds_numel', p.numel()) * p.element_size() for p in parameters)
        assert 0 < nbytes <= budget_bytes, (nbytes, budget_bytes)
        rows.append(dict(module=names[id(module)], experts=len(module.experts),
                         full_parameter_bytes=nbytes, parameter_tensors=len(parameters)))
    return dict(status='PASS', leaf_class='Qwen3MoeSparseMoeBlock', blocks=len(modules),
                per_rank_parameter_budget_bytes=budget_bytes, leaves=rows,
                adaptation='Native ZeRO-3 leaf gathers all expert and router parameters before rank-local conditional expert execution; computation and routing unchanged',
                scope='Setup validation only; actual collective progress, numerical validity and resident budget require GPU calibration/warmup')
