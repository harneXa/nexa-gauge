from ng_graph.llm.pricing import get_model_pricing


def test_get_model_pricing_prefixed_openai_key() -> None:
    pricing = get_model_pricing("openai/gpt-4o-mini")
    assert pricing.input_per_1k == 0.000150
    assert pricing.output_per_1k == 0.000600


def test_get_model_pricing_prefixed_gemini_key() -> None:
    pricing = get_model_pricing("gemini/gemini-2.5-flash")
    assert pricing.input_per_1k == 0.000300
    assert pricing.output_per_1k == 0.002500


def test_get_model_pricing_prefixed_gemini_flash_lite_key() -> None:
    pricing = get_model_pricing("gemini/gemini-2.5-flash-lite")
    assert pricing.input_per_1k == 0.000100
    assert pricing.output_per_1k == 0.000400


def test_get_model_pricing_prefixed_gemini_2_0_flash_lite_key() -> None:
    pricing = get_model_pricing("gemini/gemini-2.0-flash-lite")
    assert pricing.input_per_1k == 0.000075
    assert pricing.output_per_1k == 0.000300


def test_get_model_pricing_bare_name_backward_compatible() -> None:
    pricing = get_model_pricing("gpt-4o-mini")
    assert pricing.input_per_1k == 0.000150
    assert pricing.output_per_1k == 0.000600
