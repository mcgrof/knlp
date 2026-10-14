"""Describe which cache writes an attention call actually reads."""

FRESH = "fresh_current_qdq_v1"
WRITE_FIRST = "write_before_attention_qdq_v1"


def recipe(name=FRESH):
    if name not in (FRESH, WRITE_FIRST):
        raise ValueError(f"unknown cache recipe: {name}")
    return {
        "id": name,
        "prefill": "fresh" if name == FRESH else "rounded_new_and_stored",
        "chunked_prefill": (
            "fresh_new_rounded_stored" if name == FRESH else "rounded_new_and_stored"
        ),
        "current_token_decode": "fresh" if name == FRESH else "rounded",
        "stored_tokens": "rounded_once_on_write",
        "scale_lifetime": "frozen_per_layer_separate_k_v",
        "scale_arithmetic_dtype": "float32",
        "format": "fp8_e4m3fn",
        "rounding": "nearest_ties_even",
        "clipping": "finite_symmetric_448",
        "dequantization": "float32_product_cast_to_activation_dtype",
        "physical_storage": "activation_dtype",
    }


def manifest_recipe(manifest):
    value = manifest.get("cache_recipe", recipe())
    if value != recipe(value["id"]):
        raise ValueError("cache recipe fields do not match its implementation")
    return value
