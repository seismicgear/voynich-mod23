import pytest
from experiment.decode import Mod23Decoder

def test_decoder_initialization():
    decoder = Mod23Decoder()
    assert decoder.MOD == 23
    assert len(decoder.ALPHABET) == 23

def test_tokenize_eva_simple():
    decoder = Mod23Decoder()
    tokens = decoder.tokenize_eva("qokeedy")
    assert tokens == ["qokeedy"]

def test_tokenize_eva_composite():
    decoder = Mod23Decoder()
    # 'qo' is not in default map as a single unit if 'q' and 'o' are separate,
    # but 'qokedy' is.
    # Let's test "qokedy" + "dal" (if dal exists) or simple "q" "o"
    # q -> 1, o -> 2.
    tokens = decoder.tokenize_eva("qo")
    assert tokens == ["q", "o"]

def test_tokenize_greedy_order():
    # If we have 'daiin' (7) and 'd' (5), 'a' (7), 'i' (8), 'n' (15)
    # tokenize_eva('daiin') should return ['daiin'], not ['d', 'a', 'i', 'i', 'n']
    decoder = Mod23Decoder()
    tokens = decoder.tokenize_eva("daiin")
    assert tokens == ["daiin"]

def test_decode_word_known():
    decoder = Mod23Decoder()
    # 'q' -> 1 -> inv(1)=1 -> 'A'
    assert decoder.decode_word("q") == "A"

    # 'o' -> 2 -> inv(2)=12 -> 'M' (since 12*2 = 24 = 1 mod 23)
    # Wait, let's check the logic.
    # 2 * 12 = 24 = 1 mod 23. Correct.
    # num_to_latin[1] = 'A'
    # num_to_latin[12] = 'M' (A=1, B=2, C=3, D=4, E=5, F=6, G=7, H=8, I=9, K=10, L=11, M=12)
    # Wait, ALPHABET = "ABCDEFGHIKLMNOPQRSTVXYZ"
    # 1=A, 2=B, 3=C, 4=D, 5=E, 6=F, 7=G, 8=H, 9=I, 10=K, 11=L, 12=M
    assert decoder.decode_word("o") == "M"

def test_decode_word_unknown_glyph():
    decoder = Mod23Decoder()
    with pytest.raises(ValueError, match="Unknown glyph"):
        decoder.decode_word("q!o")

def test_mod_inverse_safe():
    decoder = Mod23Decoder()
    # 23 mod 23 is 0, not invertible. Should return 23.
    # But our map usually handles 1..23.
    # If we had a glyph mapping to 23 (z -> 23)
    # inv(23, 23) -> 23
    # 23 maps to 'Z' (last letter)
    # Let's check 'z'
    assert decoder.decode_word("z") == "Z"
