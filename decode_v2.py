"""
decode_v2.py
Positional Decoder: Different mappings for Start, Body, and End of lines.
"""

class PositionalDecoder:
    def __init__(self, map_start: dict, map_body: dict, map_end: dict):
        self.map_start = map_start
        self.map_body = map_body
        self.map_end = map_end
        self.default_char = '?'

    def decode_line(self, tokens: list[str]) -> str:
        if not tokens:
            return ""

        decoded_chars = []
        for i, token in enumerate(tokens):
            # 1. Determine State
            if i == 0:
                mapping = self.map_start
            elif i == len(tokens) - 1:
                mapping = self.map_end
            else:
                mapping = self.map_body

            # 2. Decode Token
            # If the token isn't in our map (rare), skip or use '?'
            decoded_chars.append(mapping.get(token, self.default_char))

        return "".join(decoded_chars)

    def decode_text(self, lines: list[list[str]]) -> str:
        return "".join(self.decode_line(line) for line in lines)
