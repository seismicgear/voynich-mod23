import pytest

from decoder import decode_word


def test_decode_word_success():
    """decode_word should translate known glyphs via modular inverses."""
    assert decode_word("qo") == "AM"


def test_decode_word_unknown_glyph():
    """decode_word should raise when it encounters an unmapped glyph."""
    with pytest.raises(ValueError, match=r"Unknown glyph '\*'"):
        decode_word("q*o")
