"""A frozen final delimiter must not recover answers from open reasoning."""

from tools.kv.native.completion import final_text


def test_default_normalization_preserves_existing_behavior():
    assert final_text(' "amber birch cobalt."<|im_end|>suffix') == "amber birch cobalt"


def test_final_channel_excludes_reasoning():
    text = "The archive might say jade.\n</think>\n amber birch cobalt."
    assert final_text(text, "</think>") == "amber birch cobalt"


def test_missing_final_marker_cannot_rescue_correct_words():
    assert final_text("amber birch cobalt", "</think>") == ""
    assert (
        final_text("I found amber birch cobalt but am still thinking", "</think>") == ""
    )
