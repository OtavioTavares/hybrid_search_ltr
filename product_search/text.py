import re

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text) -> list[str]:
    tokens = _TOKEN.findall(str(text).lower())
    return [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t for t in tokens]
