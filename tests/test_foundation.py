from surya.foundation import FoundationPredictor


def test_foundation_flash2():
    try:
        f = FoundationPredictor(None, None, None, "flash_attention_2")
        assert f.model.decoder.config._attn_implementation == "flash_attention_2"
        assert f.model.vision_encoder.config._attn_implementation == "flash_attention_2"
    except Exception as e:
        assert False, (
            f"FoundationPredictor with flash_attention_2 raised an exception: {e}"
        )
