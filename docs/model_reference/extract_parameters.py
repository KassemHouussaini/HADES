"""
extract_parameters.py
=====================
Parameter table of the HADES model, extracted from the code itself so that
the documentation cannot drift away from what the model computes
(decision log A4).

What is extracted (Python `ast` + `tokenize`, no code is executed):
  1. module constants: UPPER_CASE names assigned at module level;
  2. SystemContext fields with a default value (system_builder.py);
  3. numbers hard-coded inside functions: an assignment of a numeric literal
     to a name, e.g. `PSA_CH4_recovery = 0.97`.
For each: file, line, name, value, the comment block directly above it and
the comment on the same line, and a source status:
  cited        the comments contain a citation pattern (a year 19xx/20xx,
               "et al", doi, Table, Eq., a reference number [n], a named
               dataset or regulation, or a decision log entry);
  assumption   no citation, but the comment says it is an assumption or a
               model default / author choice;
  no source    neither. These are the gaps to fill before the paper.
The status is a text pattern, not a judgement of the source's quality.

Outputs: docs/model_reference/parameters.csv and parameters.md.

    python docs/model_reference/extract_parameters.py
"""
import ast
import csv
import io
import os
import re
import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REPO_URL = "https://github.com/KassemHouussaini/HADES/blob/paper2-dev"

# model files (plotting, Brightway lookup checks and ML training scripts excluded)
FILES = [
    "ADM1/ADM1.py", "ADM1/system_builder.py", "ADM1/heat_integration.py",
    "ADM1/hydrochar_enduse.py", "ADM1/ad_characterization.py", "ADM1/lca_contexts.py",
    "ADM1/feedstock_sampler.py", "ADM1/sweep_engine.py", "ADM1/lca_contributions.py",
    "ML training scripts and models/htc_surrogate.py",
]

CITED = re.compile(
    r"\b(19|20)\d\d\b|et al|doi|\bTable\b|\bEq\.|\beqs?\.|\[\d+\]|decision log|ecoinvent|Directive|NRCan|"
    r"ECCC|BSM2|Batstone|IEA|EPA|MWPS|ASABE|Fisgativa|Kleerebezem|Chandler|Aitchison|IUPAC|ADM1 \(|"
    r"\b[A-Z]\d{1,2}[a-e]?\b",              # decision log entries such as C1d, F15
    re.I)
CITED_EXTRA = None
ASSUMED = re.compile(r"assum|model default|author|user'?s? (choice|decision|range)|set by|placeholder", re.I)

# names that are plumbing, not model parameters
SKIP = re.compile(r"^(HERE|ROOT|_HERE|ML_MODULE|FG_DB|FG_ROLES|DATASET|AMBIENT_CSV|REPO_URL|FILES|"
                  r"META|AD|KEY|PARTS|SKIP|CITED|ASSUMED|META_KEYS|HEAT_CONSUMERS|DATASET_LABELS|"
                  r"OWN_CORRELATION|DEFAULT_PRESET|NG_BOILER|HEAT_CONFIGS|FEED_MOISTURE_FACTOR|"
                  r"NRCAN_TABLE1_PJ|REGIONAL_HEAT_MIX|_.*)$")     # names, dataset labels, derived tables


def _comments(path):
    """{line: comment text} for every comment token in the file."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for tok in tokenize.generate_tokens(f.readline):
            if tok.type == tokenize.COMMENT:
                out[tok.start[0]] = tok.string.lstrip("#").strip()
    return out


def _block_above(lines, comments, lineno):
    """Contiguous comment lines directly above `lineno` (stops at code or blank)."""
    block, i = [], lineno - 1
    while i >= 1:
        s = lines[i - 1].strip()
        if s.startswith("#") and i in comments:
            block.append(comments[i])
            i -= 1
        else:
            break
    return " ".join(reversed(block))


def _value(node, src):
    v = ast.get_source_segment(src, node) or ""
    v = " ".join(v.split())
    return v if len(v) <= 160 else v[:157] + "..."


# physical and chemical constants (molar masses, gas constant, reference
# temperatures, acid dissociation and water constants, gas densities): standard
# values, listed for completeness, not model choices
PHYSICAL = re.compile(r"^(MW|MW_\w+|N_KG_PER_KMOL|R|T_base|T_ref|T_REF|Kw|kw|Ka_\w+|RHO_\w+|COD_CH4|CP_KJ_PER_KG_K|LHV_\w+)$")
# numerical settings of solvers and loops, not physical parameters
NUMERICAL = re.compile(r"^(tol|maxIter|eps|MIN_BLOCKS|max_iter|n_iter|dt|atol|rtol)$")


def _status(text, name="", nearby=""):
    if PHYSICAL.match(name):
        return "physical constant"
    if NUMERICAL.match(name):
        return "numerical setting"
    if CITED.search(text):
        return "cited"
    if ASSUMED.search(text):
        return "assumption"
    if nearby and CITED.search(nearby):
        return "cited nearby"        # a citation in a comment block within 25 lines above
    return "no source"


def _is_number(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _is_number(node.operand)
    if isinstance(node, ast.BinOp):                      # e.g. 0.06/14
        return _is_number(node.left) and _is_number(node.right)
    return False


def extract(relpath):
    path = os.path.join(ROOT, relpath)
    src = open(path, encoding="utf-8").read()
    lines = src.splitlines()
    comments = _comments(path)
    tree = ast.parse(src)
    rows = []

    def add(kind, name, node, value_node, scope=""):
        ln = node.lineno
        above = _block_above(lines, comments, ln)
        inline = comments.get(ln, "") if not lines[ln - 1].strip().startswith("#") else ""
        # comments on the continuation lines of a multi-line assignment
        for k in range(ln + 1, (node.end_lineno or ln) + 1):
            if k in comments:
                inline = (inline + " " + comments[k]).strip()
        text = " ".join(t for t in (above, inline) if t)
        nearby = " ".join(comments[k] for k in range(max(1, ln - 25), ln) if k in comments)
        rows.append(dict(file=relpath, line=ln, kind=kind, scope=scope, name=name,
                         value=_value(value_node, src), comment=text, status=_status(text, name, nearby),
                         link=f"{REPO_URL}/{relpath.replace(' ', '%20')}#L{ln}"))

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            n = node.targets[0].id
            seg = ast.get_source_segment(src, node.value) or ""
            # model parameters carry numbers; lists of names and key maps are structure
            if n.isupper() and not SKIP.match(n) and re.search(r"\d", seg) and not re.match(r"^(STATE_NAMES|IDX)", n):
                add("module constant", n, node, node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            n = node.target.id
            if n.isupper() and not SKIP.match(n):
                add("module constant", n, node, node.value)
        elif isinstance(node, ast.ClassDef) and node.name == "SystemContext":
            for b in node.body:
                if isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name) and b.value is not None:
                    if _is_number(b.value):           # numeric defaults only (None / "" are plumbing)
                        add("SystemContext field", b.target.id, b, b.value, scope="SystemContext")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Assign) and len(sub.targets) == 1
                        and isinstance(sub.targets[0], ast.Name) and _is_number(sub.value)):
                    n = sub.targets[0].id
                    if n in ("i", "j", "k", "n", "t", "total", "left", "used", "out") or n.startswith("_"):
                        continue
                    if isinstance(sub.value, ast.Constant) and sub.value.value in (0, 0.0, 1, 1.0):
                        continue                          # counters and initializations
                    add("number in function", n, sub, sub.value, scope=node.name)
    return rows


def main():
    rows = [r for f in FILES for r in extract(f)]
    out_dir = HERE
    with open(os.path.join(out_dir, "parameters.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    STATUSES = ("cited", "cited nearby", "assumption", "physical constant", "numerical setting", "no source")
    n = {s: sum(r["status"] == s for r in rows) for s in STATUSES}
    md = ["# HADES parameter table", "",
          "Generated by `docs/model_reference/extract_parameters.py` from the code on `paper2-dev`; "
          "do not edit by hand, rerun the script. Status is a text pattern on the code comments: "
          "*cited* (a reference, dataset, regulation or decision log entry), *assumption* "
          "(stated as an assumption or model default), *no source* (neither; to fill).", "",
          "*cited nearby*: no citation next to the value, but one in a comment within 25 lines above "
          "(check it applies). *physical constant* and *numerical setting* are standard values and "
          "solver settings.", "",
          "**" + f"{len(rows)} entries: " + ", ".join(f"{v} {k}" for k, v in n.items()) + ".**", ""]
    by_file = {}
    for r in rows:
        by_file.setdefault(r["file"], []).append(r)
    for f, rs in by_file.items():
        k = {s: sum(r["status"] == s for r in rs) for s in STATUSES}
        md += [f"## {f}", "",
               f"{len(rs)} entries: " + ", ".join(f"{v} {s_}" for s_, v in k.items() if v) + ".", "",
               "| Name | Where | Value | Status | Comment in the code |", "|---|---|---|---|---|"]
        for r in rs:
            where = f"[L{r['line']}]({r['link']})" + (f" `{r['scope']}`" if r["scope"] else "")
            com = r["comment"].replace("|", "\\|")
            com = com if len(com) <= 300 else com[:297] + "..."
            val = r["value"].replace("|", "\\|")
            flag = "**no source**" if r["status"] == "no source" else r["status"]
            md.append(f"| `{r['name']}` | {where} | `{val}` | {flag} | {com} |")
        md.append("")
    with open(os.path.join(out_dir, "parameters.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"{len(rows)} entries: {n}")
    for f, rs in by_file.items():
        print(f"  {f}: {len(rs)} ({sum(r['status'] == 'no source' for r in rs)} without source)")


if __name__ == "__main__":
    main()
