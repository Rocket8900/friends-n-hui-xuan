import re
import math
from typing import Dict, Any
from flask import Blueprint, request, jsonify

trading_formula_bp = Blueprint("trading_formula", __name__)

_GREEK = {
    r"\alpha": "alpha",
    r"\beta": "beta",
    r"\gamma": "gamma",
    r"\delta": "delta",
    r"\epsilon": "epsilon",
    r"\zeta": "zeta",
    r"\eta": "eta",
    r"\theta": "theta",
    r"\iota": "iota",
    r"\kappa": "kappa",
    r"\lambda": "lambda",
    r"\mu": "mu",
    r"\nu": "nu",
    r"\xi": "xi",
    r"\pi": "pi",
    r"\rho": "rho",
    r"\sigma": "sigma",
    r"\tau": "tau",
    r"\upsilon": "upsilon",
    r"\phi": "phi",
    r"\chi": "chi",
    r"\psi": "psi",
    r"\omega": "omega",
}

def _strip_math_delims(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\$+\s*|\s*\$+$", "", s)
    s = re.sub(r"^\\\(|\\\)$", "", s)
    s = re.sub(r"^\\\[|\\\]$", "", s)
    return s.strip()

def _replace_greek(s: str) -> str:
    for k, v in _GREEK.items():
        s = s.replace(k, v)
    return s

def _replace_text_vars(s: str) -> str:
    return re.sub(r"\\text\{([^}]+)\}", r"\1", s)

def _replace_expectation_brackets(s: str) -> str:
    if "=" in s:
        s = s.split("=", 1)[1]
    s = re.sub(r"([A-Za-z]+)\[([^\]]+)\]", lambda m: f"{m.group(1)}_{re.sub(r'[^0-9A-Za-z_]+','_', m.group(2))}", s)
    return s

def _normalize_ops(s: str) -> str:
    s = s.replace(r"\times", "*").replace(r"\cdot", "*")
    s = s.replace("−", "-")  
    s = s.replace("^", "**")
    s = s.replace(r"\max", "max").replace(r"\min", "min")
    s = s.replace(r"\log", "log").replace(r"\ln", "log")
    s = s.replace(" ", "")
    return s

def _replace_frac(s: str) -> str:
    def replace_once(text: str):
        i = 0
        while True:
            i = text.find(r"\frac{", i)
            if i == -1:
                return text, False
            j = i + len(r"\frac")  
            if j >= len(text) or text[j] != "{":
                i += 1
                continue

            def read_brace(start: int):
                assert text[start] == "{"
                depth = 0
                k = start
                out = []
                while k < len(text):
                    ch = text[k]
                    if ch == "{":
                        depth += 1
                        if depth > 1:
                            out.append(ch)
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            return k + 1, "".join(out)
                        else:
                            out.append(ch)
                    else:
                        out.append(ch)
                    k += 1
                raise ValueError("Unbalanced braces in \\frac")

            num_end, num = read_brace(j)
            if num_end >= len(text) or text[num_end] != "{":
                i += 1
                continue
            den_end, den = read_brace(num_end)

            before = text[:i]
            after = text[den_end:]
            repl = f"({num})/({den})"
            return before + repl + after, True

    changed = True
    while changed:
        s, changed = replace_once(s)
    return s

def _replace_exponentials(s: str) -> str:
    s = re.sub(r"e\^\{([^}]+)\}", r"exp(\1)", s)
    s = re.sub(r"e\^([A-Za-z0-9_\.]+)", r"exp(\1)", s)
    return s

def _replace_summations(s: str) -> str:
    while True:
        m = re.search(r"\\sum_\{([A-Za-z])=([^}]+)\}\^\{([^}]+)\}", s)
        if not m:
            break
        var, start, end = m.group(1), m.group(2), m.group(3)
        head_end = m.end()
        if head_end < len(s) and s[head_end] == "{":
            depth = 0
            k = head_end
            expr = []
            while k < len(s):
                ch = s[k]
                if ch == "{":
                    depth += 1
                    if depth > 1: expr.append(ch)
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        k += 1
                        break
                    else:
                        expr.append(ch)
                else:
                    expr.append(ch)
                k += 1
            inner = "".join(expr)
            inner_py = _latex_to_python(inner)
            py = f"(sum(({inner_py}) for {var} in range(int({start}), int({end})+1)))"
            s = s[:m.start()] + py + s[k:]
        else:
            k = head_end
            paren = 0
            body = []
            while k < len(s):
                ch = s[k]
                if ch in "({":
                    paren += 1
                    body.append(ch)
                elif ch in ")}":
                    paren -= 1
                    body.append(ch)
                elif paren == 0 and ch in "+-" :
                    break
                else:
                    body.append(ch)
                k += 1
            inner = "".join(body).strip()
            inner_py = _latex_to_python(inner)
            py = f"(sum(({inner_py}) for {var} in range(int({start}), int({end})+1)))"
            s = s[:m.start()] + py + s[k:]
    return s

def _latex_to_python(s: str) -> str:
    s = _strip_math_delims(s)
    s = _replace_text_vars(s)
    s = _replace_greek(s)
    s = _replace_expectation_brackets(s)
    s = _replace_frac(s)
    s = _replace_exponentials(s)
    s = _replace_summations(s)
    s = _normalize_ops(s)
    s = re.sub(r"\\_", "_", s)  
    s = re.sub(r"_\{([^}]+)\}", r"_\1", s) 
    s = _insert_implicit_mul(s) 
    return s

def _safe_eval(expr: str, variables: Dict[str, Any]) -> float:
    safe_env = {
        "max": max,
        "min": min,
        "log": math.log,
        "exp": math.exp,
        "sum": sum,
        "abs": abs,
        "e": math.e,
        "pi": math.pi,
        "__builtins__": None,
    }
    safe_env.update(variables)
    return float(eval(expr, safe_env, {}))


def _insert_implicit_mul(s: str) -> str:
    """
    Insert '*' for implicit multiplication:
      - var( … ), num( … ), )(, )var, )num, varnum  -> insert '*'
    but DO NOT insert before known function calls: max( … ), min( … ), log( … ), exp( … ), sum( … ), abs( … ).
    """
    KNOWN_FUNCS = {"max", "min", "log", "exp", "sum", "abs"}

    tok_re = re.compile(
        r"""
        ([A-Za-z_][A-Za-z0-9_]*) |     
        (\d+(?:\.\d*)?|\.\d+)    |     
        (\() | (\)) | ([+\-*/])        
        """,
        re.VERBOSE,
    )

    tokens = []
    pos = 0
    for m in tok_re.finditer(s):
        if m.start() > pos:
            tokens.append(s[pos:m.start()])
        tokens.append(m.group(0))
        pos = m.end()
    if pos < len(s):
        tokens.append(s[pos:])

    def tok_type(t: str) -> str:
        if not t:
            return "OTHER"
        if t[0].isalpha() or t[0] == "_":
            return "IDENT"
        if t[0].isdigit() or (t[0] == "." and len(t) > 1 and t[1].isdigit()):
            return "NUMBER"
        if t == "(":
            return "LP"
        if t == ")":
            return "RP"
        if t in "+-*/":
            return "OP"
        return "OTHER"

    out = []
    for i, t in enumerate(tokens):
        if i > 0:
            prev = tokens[i - 1]
            ttype, ptype = tok_type(t), tok_type(prev)

            if ttype == "LP" and ptype == "IDENT" and prev in KNOWN_FUNCS:
                pass
            else:
                if (
                    (ptype in {"IDENT", "NUMBER", "RP"} and ttype in {"IDENT", "NUMBER", "LP"})
                ):
                    out.append("*")
        out.append(t)

    return "".join(out)



@trading_formula_bp.route("/trading-formula", methods=["POST"])
def trading_formula():
    try:
        data = request.get_json(force=True, silent=False)
        if not isinstance(data, list):
            return jsonify({"error": "Input must be a JSON array"}), 400

        results = []
        for case in data:
            formula = case.get("formula", "")
            variables = case.get("variables", {}) or {}
            normalized_vars = {}
            for k, v in variables.items():
                k_norm = _replace_greek(k)
                normalized_vars[k_norm] = v
            expr = _latex_to_python(formula)
            value = _safe_eval(expr, normalized_vars)
            results.append({"result": round(value + 0.0000000001, 4)})  

        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 400
