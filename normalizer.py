
import re

LEET_MAP = {
    "0": "o", "1": "i", "3": "e", "4": "a",
    "5": "s", "6": "g", "7": "t", "8": "b",
    "9": "g", "@": "a", "$": "s", "!": "i",
    "+": "t", "(": "c", "#": "h", "%": "x"
}

SLANG_MAP = {
    "kys": "kill yourself", "kms": "kill myself",
    "gtfo": "get out", "stfu": "shut up",
    "foh": "get out of here", "pos": "piece of trash",
    "sob": "son of a bad person", "wth": "what the heck",
    "wtf": "what the heck", "af": "very",
    "fml": "my life is ruined", "smh": "shaking my head",
    "istg": "i swear", "ngl": "not going to lie",
    "omg": "oh my god", "omfg": "oh my god",
    "lmao": "laughing", "lmfao": "laughing",
    "rofl": "laughing", "idk": "i do not know",
    "imo": "in my opinion", "tbh": "to be honest",
    "irl": "in real life", "dm": "direct message",
    "fyi": "for your information", "btw": "by the way",
}

def decode_leet(text):
    return " ".join("".join(LEET_MAP.get(c, c) for c in w.lower()) for w in text.split())

def decode_slang(text):
    return " ".join(SLANG_MAP.get(w, w) for w in text.lower().split())

def remove_repeated_chars(text):
    return re.sub(r"(.)\1{2,}", r"\1\1", text)

def remove_special_separators(text):
    return re.sub(r"\b(\w)([.\-_](\w)){2,}\b",
                  lambda m: m.group(0).replace(".","").replace("-","").replace("_",""), text)

def normalize_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@(\w+)", r"\1", text)
    text = remove_special_separators(text)
    text = decode_leet(text)
    text = decode_slang(text)
    text = remove_repeated_chars(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
