import re
from flask import Blueprint, request, jsonify

duolingo_sort_bp = Blueprint("duolingo_sort", __name__)


_ROMAN_RE = re.compile(r"^(M{0,3})(CM|CD|D?C{0,3})"
                       r"(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$", re.I)
_ARABIC_RE = re.compile(r"^\d+$")

_CHN_NUM_CHARS = set("零〇一二三四五六七八九十百千萬万億亿点兩两").union(set("壹貳叁肆伍陸柒捌玖拾佰仟"))  

def is_roman(s: str) -> bool:
    t = s.strip().upper()
    return bool(_ROMAN_RE.match(t)) and t != ""

def is_arabic(s: str) -> bool:
    return bool(_ARABIC_RE.match(s.strip()))

def is_chinese(s: str) -> bool:
    return any(ch in _CHN_NUM_CHARS for ch in s)

def is_german_word(s: str) -> bool:
    t = s.strip().lower()
    return bool(re.fullmatch(r"[a-zäöüß\- ]+", t)) and any(
        key in t for key in [
            "null","eins","ein","zwei","drei","vier","fünf","funf","sechs","sieben","acht","neun",
            "zehn","elf","zwölf","zwolf","dreizehn","vierzehn","fünfzehn","funfzehn","sechzehn",
            "siebzehn","achtzehn","neunzehn","zwanzig","dreißig","dreissig","vierzig","fünfzig",
            "funfzig","sechzig","siebzig","achtzig","neunzig","hundert","tausend","million","milliarde"
        ]
    )

def is_english_word(s: str) -> bool:
    t = s.strip().lower()
    return bool(re.fullmatch(r"[a-z\- ]+", t)) and any(
        key in t for key in [
            "zero","one","two","three","four","five","six","seven","eight","nine",
            "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen",
            "eighteen","nineteen","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety",
            "hundred","thousand","million","billion","trillion","and"
        ]
    )


def roman_to_int(s: str) -> int:
    vals = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':250//1*2, 'M':1000}  
    vals['D'] = 500
    s = s.upper()
    total = 0
    prev = 0
    for ch in reversed(s):
        v = vals[ch]
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    return total

_EN_UNITS = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,
    "ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,
    "seventeen":17,"eighteen":18,"nineteen":19
}
_EN_TENS = {
    "twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90
}
_EN_SCALES = {"hundred":100, "thousand":1_000, "million":1_000_000, "billion":1_000_000_000, "trillion":1_000_000_000_000}

def english_to_int(text: str) -> int:
    t = text.lower().replace("-", " ")
    t = re.sub(r"\band\b", " ", t)
    parts = [p for p in t.split() if p]
    if not parts:
        raise ValueError("empty english")
    total = 0
    curr = 0
    for w in parts:
        if w in _EN_UNITS:
            curr += _EN_UNITS[w]
        elif w in _EN_TENS:
            curr += _EN_TENS[w]
        elif w == "hundred":
            if curr == 0: curr = 1
            curr *= 100
        elif w in _EN_SCALES and w != "hundred":
            if curr == 0: curr = 1
            total += curr * _EN_SCALES[w]
            curr = 0
        else:
            raise ValueError(f"unknown english token: {w}")
    return total + curr

_CHN_DIGITS = {
    "零":0,"〇":0,"〇":0,"一":1,"二":2,"兩":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,
    "壹":1,"貳":2,"叁":3,"肆":4,"伍":5,"陸":6,"柒":7,"捌":8,"玖":9
}
_CHN_UNITS = {"十":10,"拾":10,"百":100,"佰":100,"千":1000,"仟":1000}
_CHN_BIG = {"萬":10_000,"万":10_000,"億":100_000_000,"亿":100_000_000}

def chinese_to_int(text: str) -> int:
    s = text.strip()
    if s in ("零","〇"):
        return 0

    def parse_section(sub: str) -> int:
        val = 0; num = 0
        for ch in sub:
            if ch in _CHN_DIGITS:
                num = _CHN_DIGITS[ch]
            elif ch in _CHN_UNITS:
                unit = _CHN_UNITS[ch]
                val += (num if num != 0 else 1) * unit
                num = 0
            else:
                pass
        return val + num

    total = 0
    last = s
    for big_char, big_val in [("億",100_000_000),("亿",100_000_000),("萬",10_000),("万",10_000)]:
        if big_char in last:
            parts = last.split(big_char)
            left = parse_section(parts[0]) if parts[0] else 1
            right = parse_section(parts[1]) if len(parts) > 1 else 0
            total += left * big_val
            last = parts[1] if len(parts) > 1 else ""
    return total + parse_section(last)

_DE_UNITS = {
    "null":0,"ein":1,"eins":1,"eine":1,"einen":1,"zwei":2,"drei":3,"vier":4,"fuenf":5,"fünf":5,
    "sechs":6,"sieben":7,"acht":8,"neun":9,"zehn":10,"elf":11,"zwoelf":12,"zwölf":12
}
_DE_TEENS = {
    "dreizehn":13,"vierzehn":14,"fuenfzehn":15,"fünfzehn":15,"sechzehn":16,"siebzehn":17,
    "achtzehn":18,"neunzehn":19
}
_DE_TENS = {
    "zwanzig":20,"dreissig":30,"dreißig":30,"vierzig":40,"fuenfzig":50,"fünfzig":50,
    "sechzig":60,"siebzig":70,"achtzig":80,"neunzig":90
}
_DE_SCALES = {"hundert":100, "tausend":1000, "million":1_000_000, "millionen":1_000_000,
              "milliarde":1_000_000_000, "milliarden":1_000_000_000}

def _normalize_german(t: str) -> str:
    t = t.lower()
    t = re.sub(r"(hundert)", r" \1 ", t)
    t = re.sub(r"(tausend)", r" \1 ", t)
    t = re.sub(r"(millionen|million|milliarden|milliarde)", r" \1 ", t)
    t = t.replace("-", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def german_to_int(text: str) -> int:
    t = _normalize_german(text)  
    parts = [p for p in t.split() if p and p != "und"]  

    total = 0
    curr = 0
    i = 0
    while i < len(parts):
        w = parts[i]

        if w in ("million","millionen","milliarde","milliarden","tausend"):
            scale = _DE_SCALES[w]
            if curr == 0: curr = 1
            total += curr * scale
            curr = 0
            i += 1
            continue

        if w == "hundert":
            if curr == 0: curr = 1
            curr *= 100
            i += 1
            continue

        if w in _DE_UNITS:
            curr += _DE_UNITS[w]
            i += 1
            continue
        if w in _DE_TEENS:
            curr += _DE_TEENS[w]
            i += 1
            continue
        if w in _DE_TENS:
            curr += _DE_TENS[w]
            i += 1
            continue

        m = re.fullmatch(r"([a-zäöüß]+)und([a-zäöüß]+)", w)
        if m:
            left, right = m.group(1), m.group(2)
            if left in _DE_UNITS and right in _DE_TENS:
                curr += _DE_UNITS[left] + _DE_TENS[right]
                i += 1
                continue

        raise ValueError(f"unknown german token: {w}")

    return total + curr


LANG_PRIORITY = {
    "roman": 0,
    "english": 1,
    "zh_trad": 2,
    "zh_simp": 3,
    "german": 4,
    "arabic": 5,
}

def detect_and_parse(s: str) -> tuple[int, str]:
    """Return (value, lang_tag). lang_tag used for tie-breaking priority."""
    t = s.strip()
    if is_arabic(t):
        return int(t), "arabic"
    if is_roman(t):
        return roman_to_int(t), "roman"
    if is_chinese(t):
        val = chinese_to_int(t)
        if any(ch in t for ch in ("万","亿")):
            tag = "zh_simp"
        elif any(ch in t for ch in ("萬","億")):
            tag = "zh_trad"
        else:
            tag = "zh_trad"
        return val, tag
    if is_english_word(t):
        return english_to_int(t), "english"
    if is_german_word(t):
        return german_to_int(t), "german"
    try:
        return int(t), "arabic"
    except:
        raise ValueError(f"Unrecognized number format: {s}")


@duolingo_sort_bp.route("/duolingo-sort", methods=["POST"])
def duolingo_sort():
    try:
        payload = request.get_json(force=True, silent=False)
        if not isinstance(payload, dict):
            return jsonify({"error": "Body must be a JSON object"}), 400
        part = payload.get("part")
        ch_input = (payload.get("challengeInput") or {})
        unsorted_list = ch_input.get("unsortedList")
        if part not in ("ONE", "TWO"):
            return jsonify({"error": "part must be 'ONE' or 'TWO'"}), 400
        if not isinstance(unsorted_list, list):
            return jsonify({"error": "challengeInput.unsortedList must be a list"}), 400

        if part == "ONE":
            vals = []
            for s in unsorted_list:
                ss = str(s).strip()
                if is_arabic(ss):
                    vals.append(int(ss))
                elif is_roman(ss):
                    vals.append(roman_to_int(ss))
                else:
                    return jsonify({"error": f"Invalid item for Part ONE (only Roman/Arabic allowed): {s}"}), 400
            vals.sort()
            return jsonify({"sortedList": [str(v) for v in vals]})

        enriched = []
        for s in unsorted_list:
            val, tag = detect_and_parse(str(s))
            enriched.append((val, LANG_PRIORITY[tag], tag, s))

        enriched.sort(key=lambda x: (x[0], x[1]))
        sorted_strings = [orig for (_, _, _, orig) in enriched]
        return jsonify({"sortedList": sorted_strings})

    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 400
