r"""
erlotinib_planner_agent.py

Implements exactly this architecture:

    Current state + target + route history
                    |
                    v
            AI AGENT / PLANNER
        "What chemistry should happen?"
              /              \
    Known template       No template?
       library           generate candidate
              \              /
                    v
                  RDKit
              execute + validate
              /              \
         failure            success
            |                  |
     feedback to AI       score candidate
            |                  |
      (back to planner)   beam search
                                |
                    (back to top, next iteration)

One Claude call ("the planner") both decides what chemistry should happen
AND decides, per proposed step, whether it matches a known template or
needs a freshly generated SMARTS -- there is no second "author SMARTS"
call. RDKit is the only thing that can accept or reject a step. A
rejected step's error is fed straight back to the planner for a bounded,
step-scoped retry (it does not restart the whole iteration). Accepted
steps are scored and go into the beam; the beam's survivors become next
iteration's "current state".

Requires:
    pip install anthropic rdkit
    export ANTHROPIC_API_KEY=...

Run:
    python erlotinib_planner_agent.py --max-depth 4 --beam-width 5 --k 4
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import anthropic

# --------------------------------------------------------------------------
# 1. Endpoints & chemistry ground truth
# --------------------------------------------------------------------------

START_SMILES = "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1"          # erlotinib
TARGET_SMILES = "COc1cc2ncnc(Nc3cccc(C#N)c3)c2cc1OCCN"                # target
CORE_SMARTS = Chem.MolFromSmarts("c1ccc2ncnc(Nc3ccccc3)c2c1")

_target_mol = Chem.MolFromSmiles(TARGET_SMILES)
_target_fp = AllChem.GetMorganFingerprintAsBitVect(_target_mol, 2, 2048)
_target_canon = Chem.MolToSmiles(_target_mol)


def fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)


def tanimoto_to_target(mol) -> float:
    return DataStructs.TanimotoSimilarity(fp(mol), _target_fp)


def core_intact(mol) -> bool:
    return mol.HasSubstructMatch(CORE_SMARTS)


def is_exact_target(mol) -> bool:
    return Chem.MolToSmiles(mol) == _target_canon


def try_apply_smarts(mol, smarts: str):
    """Compile + run a reaction SMARTS. Returns (best_product_mol, error)."""
    try:
        rxn = AllChem.ReactionFromSmarts(smarts)
        if rxn is None:
            return None, "SMARTS did not compile."
    except Exception as e:
        return None, f"SMARTS compile error: {e}"

    try:
        outcomes = rxn.RunReactants((mol,))
    except Exception as e:
        return None, f"RunReactants error: {e}"

    if not outcomes:
        return None, "no matching site on current molecule"

    products, seen = [], set()
    for outcome in outcomes:
        try:
            combined = outcome[0]
            for extra in outcome[1:]:
                combined = Chem.CombineMols(combined, extra)
            Chem.SanitizeMol(combined)
        except Exception:
            continue
        smi = Chem.MolToSmiles(combined)
        if smi in seen:
            continue
        seen.add(smi)
        products.append(combined)

    if not products:
        return None, "product(s) failed sanitization"

    products.sort(key=tanimoto_to_target, reverse=True)
    return products[0], None


# --------------------------------------------------------------------------
# 2. Known template library ("known template library" branch)
# --------------------------------------------------------------------------

TEMPLATE_LIBRARY = {
    "alkyne_to_nitrile": dict(
        smarts="[c:1][C:2]#[CH]>>[c:1][C:2]#N",
        family="aniline_edit", confidence=0.90,
        literature_basis="tBuONO/N-oxide or NIS/TMSN3 late-stage ArC#CH -> ArC#N.",
    ),
    "mono_demethylate_terminal_ether": dict(
        smarts="[c:1][O:2][CH2:3][CH2:4][O:5][CH3:6]>>[c:1][O:2][CH2:3][CH2:4][O:5][H]",
        family="arm_differentiate", confidence=0.85,
        literature_basis="Sub-stoichiometric BBr3, -55C: cleaves only the terminal methyl ether.",
    ),
    "exhaustive_dealkylate_to_phenol": dict(
        smarts="[c:1][O:2][CH2:3][CH2:4][O:5][CH3:6]>>[c:1][O:2][H]",
        family="arm_shorten", confidence=0.75,
        literature_basis="Forcing BBr3/HBr cleaves the whole 2-carbon arm off, leaving a phenol.",
    ),
    "O_methylate_phenol": dict(
        smarts="[c:1][OH:2]>>[c:1][O:2][CH3]",
        family="arm_shorten", confidence=0.90,
        literature_basis="MeI or Me2SO4 / K2CO3, standard phenol O-methylation.",
    ),
    "primary_alcohol_to_amine": dict(
        smarts="[CH2:1][OH:2]>>[CH2:1][NH2:2]",
        family="arm_lengthen", confidence=0.80,
        literature_basis="Lumped activate/azide-displace/reduce (or Mitsunobu-Gabriel).",
    ),
    "mitsunobu_direct_aminoethyl_etherification": dict(
        smarts="[c:1][OH:2]>>[c:1][O:2][CH2][CH2][NH2]",
        family="arm_install_convergent", confidence=0.75,
        literature_basis="Mitsunobu of the phenol with N-Boc-ethanolamine, then Boc removal.",
    ),
}

TEMPLATE_MENU = "\n".join(
    f"- {name}: {t['literature_basis']}" for name, t in TEMPLATE_LIBRARY.items()
)

# --------------------------------------------------------------------------
# 3. The planner ("AI AGENT / PLANNER" box)
# --------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = f"""You are the planner in a retrosynthesis search. \
Given the current molecule, the target, and the route so far, decide what \
chemistry should happen next.

Known template library (prefer these when they genuinely fit -- they are \
pre-verified reaction SMARTS, more reliable than anything generated fresh):
{TEMPLATE_MENU}

For each candidate step, either:
(a) reference a template by exact name in "template_name" (leave \
"reaction_smarts" null), or
(b) if nothing in the library fits, set "template_name" to null and write \
your own "reaction_smarts" (RDKit-compatible reaction SMARTS).

Hard constraints:
- Never propose a step that breaks the quinazoline ring or the exocyclic \
C(quinazoline)-N(aniline) bond.
- Make candidates genuinely different bond edits, not reagent variants of \
the same disconnection.
- If truly no productive step exists from this molecule, return an empty array.

Respond with ONLY a JSON array (no markdown fences, no prose outside it), \
each element:
{{
  "transform_label": "short_snake_case_name",
  "family": "one of: aniline_edit | arm_shorten | arm_lengthen | "
            "arm_differentiate | arm_install_convergent | other",
  "template_name": "name from the library, or null",
  "reaction_smarts": "only if template_name is null",
  "literature_basis": "short justification",
  "confidence": 0.0-1.0
}}
"""

FIX_SYSTEM_PROMPT = """You are the planner in a retrosynthesis search. Your \
previous proposal for the NEXT step was rejected by the chemistry engine \
(RDKit). Given the error, propose a corrected replacement for that ONE step \
-- either a different template_name from the library, or a fixed \
reaction_smarts. Respond with ONLY a single JSON object, same schema as \
before (transform_label, family, template_name, reaction_smarts, \
literature_basis, confidence)."""


def call_messages_create(client, **kwargs):
    """
    Thin wrapper around client.messages.create() that degrades gracefully if
    the installed anthropic SDK's signature doesn't accept 'temperature'
    (seen with some older/mismatched SDK versions -- run `pip install -U
    anthropic` to fix properly). Falls back to calling without it so the
    search can still run; a warning is printed once.
    """
    try:
        return client.messages.create(**kwargs)
    except TypeError as e:
        if "temperature" in str(e) and "temperature" in kwargs:
            if not getattr(call_messages_create, "_warned", False):
                print("    [warn] installed anthropic SDK rejected "
                      "'temperature' -- calling without it. "
                      "Run `pip install -U anthropic` to fix this properly.",
                      file=sys.stderr)
                call_messages_create._warned = True
            kwargs = {k: v for k, v in kwargs.items() if k != "temperature"}
            return client.messages.create(**kwargs)
        raise


def extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


def resolve_smarts(candidate: dict):
    """Known-template-vs-generate branch: deterministic, not the AI's call."""
    name = candidate.get("template_name")
    if name and name in TEMPLATE_LIBRARY:
        return TEMPLATE_LIBRARY[name]["smarts"], "template", TEMPLATE_LIBRARY[name]["confidence"]
    smarts = candidate.get("reaction_smarts")
    conf = float(candidate.get("confidence", 0.5))
    return smarts, "generated", conf


# --------------------------------------------------------------------------
# 4. Node / Step
# --------------------------------------------------------------------------

@dataclass
class Step:
    transform_label: str
    family: str
    smarts: str
    source: str            # "template" or "generated"
    literature_basis: str
    confidence: float


@dataclass
class Node:
    smiles: str
    mol: object = field(repr=False)
    path: tuple[Step, ...] = field(default_factory=tuple)
    depth: int = 0

    @property
    def route_confidence(self) -> float:
        c = 1.0
        for s in self.path:
            c *= s.confidence
        return c

    def score(self) -> float:
        s = tanimoto_to_target(self.mol) - 0.03 * self.depth
        s *= (0.5 + 0.5 * self.route_confidence)
        if is_exact_target(self.mol):
            s += 1.0
        return s


def history_text(node: Node) -> str:
    if not node.path:
        return "none (this is the start)"
    return ", ".join(s.transform_label for s in node.path)


def planner_propose(client, model, node: Node, k: int, temperature: float) -> list[dict]:
    """AI AGENT / PLANNER: 'what chemistry should happen?' -- k candidates."""
    prompt = (
        f"Target molecule (SMILES): {TARGET_SMILES}\n"
        f"Current molecule (SMILES): {node.smiles}\n"
        f"Current Tanimoto similarity to target: {tanimoto_to_target(node.mol):.3f}\n"
        f"Route so far: {history_text(node)}\n\n"
        f"Propose up to {k} diverse candidate next steps as a JSON array."
    )
    resp = call_messages_create(
        client, model=model, max_tokens=1200, temperature=temperature,
        system=PLANNER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    try:
        return extract_json(text)[:k]
    except Exception as e:
        print(f"    [warn] planner returned unparseable JSON ({e})", file=sys.stderr)
        return []


def planner_fix(client, model, node: Node, bad_candidate: dict, error: str, temperature: float) -> dict | None:
    """'feedback to AI' loop: bounded, scoped to this one rejected step."""
    prompt = (
        f"Target molecule (SMILES): {TARGET_SMILES}\n"
        f"Current molecule (SMILES): {node.smiles}\n"
        f"Your rejected proposal: {json.dumps(bad_candidate)}\n"
        f"RDKit error: {error}\n"
        f"Propose a corrected replacement as a single JSON object."
    )
    resp = call_messages_create(
        client, model=model, max_tokens=600, temperature=temperature,
        system=FIX_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    try:
        data = extract_json(text)
        return data[0] if isinstance(data, list) else data
    except Exception as e:
        print(f"    [warn] fix response unparseable ({e})", file=sys.stderr)
        return None


# --------------------------------------------------------------------------
# 5. RDKit execute + validate, with the failure -> feedback -> retry loop
# --------------------------------------------------------------------------

def execute_and_validate(client, model, node: Node, candidate: dict,
                          max_retries: int, temperature: float) -> Node | None:
    """RDKit execute+validate box, including the bounded 'feedback to AI' loop."""
    for attempt in range(max_retries + 1):
        smarts, source, conf = resolve_smarts(candidate)
        if not smarts:
            error = "no template_name and no reaction_smarts supplied"
        else:
            product, error = try_apply_smarts(node.mol, smarts)
            if product is not None and not core_intact(product):
                product, error = None, "product loses the required quinazoline core"
            if product is not None:
                step = Step(
                    transform_label=candidate.get("transform_label", "unnamed"),
                    family=candidate.get("family", "other"),
                    smarts=smarts, source=source,
                    literature_basis=candidate.get("literature_basis", ""),
                    confidence=conf,
                )
                return Node(smiles=Chem.MolToSmiles(product), mol=product,
                            path=node.path + (step,), depth=node.depth + 1)

        # failure -> feedback to AI (bounded)
        if attempt < max_retries:
            print(f"      [retry {attempt+1}/{max_retries}] "
                  f"{candidate.get('transform_label')}: {error}", file=sys.stderr)
            fixed = planner_fix(client, model, node, candidate, error, temperature)
            if fixed is None:
                break
            candidate = fixed
        else:
            print(f"      [give up] {candidate.get('transform_label')}: {error}", file=sys.stderr)

    return None


# --------------------------------------------------------------------------
# 6. Score candidate + beam search (loop back to top)
# --------------------------------------------------------------------------

def run_iteration(client, model, frontier: list[Node], k: int, beam_width: int,
                   max_retries: int, temperature: float,
                   visited: set[str], prune_below: float) -> tuple[list[Node], list[Node]]:
    pool: dict[str, Node] = {}
    hits: list[Node] = []

    for node in frontier:
        print(f"  planner: what chemistry should happen from {node.smiles} "
              f"(depth {node.depth})", file=sys.stderr)
        candidates = planner_propose(client, model, node, k, temperature)

        for candidate in candidates:
            child = execute_and_validate(client, model, node, candidate, max_retries, temperature)
            if child is None:
                continue
            if child.smiles in visited:
                continue
            sc = child.score()
            if sc < prune_below:
                continue
            if is_exact_target(child.mol):
                hits.append(child)
            existing = pool.get(child.smiles)
            if existing is None or sc > existing.score():
                pool[child.smiles] = child

    next_frontier = sorted(pool.values(), key=lambda n: n.score(), reverse=True)[:beam_width]
    return next_frontier, hits


def search(client, model, max_depth: int, beam_width: int, k: int,
           max_retries: int, temperature: float, prune_below: float):
    root = Node(smiles=Chem.MolToSmiles(Chem.MolFromSmiles(START_SMILES)),
                mol=Chem.MolFromSmiles(START_SMILES))
    frontier = [root]
    visited = {root.smiles}
    all_hits: list[Node] = []

    for depth in range(1, max_depth + 1):
        print(f"--- iteration {depth}: {len(frontier)} node(s) in beam ---", file=sys.stderr)
        frontier, hits = run_iteration(client, model, frontier, k, beam_width,
                                        max_retries, temperature, visited, prune_below)
        visited.update(n.smiles for n in frontier)
        all_hits.extend(hits)
        if not frontier:
            print("    beam collapsed to empty; stopping.", file=sys.stderr)
            break

    return all_hits, frontier


# --------------------------------------------------------------------------
# 7. Cluster into route families + report
# --------------------------------------------------------------------------

def family_signature(node: Node) -> tuple:
    return tuple(sorted({s.family for s in node.path}))


def cluster(nodes: list[Node], top_n=10) -> list[Node]:
    best: dict[tuple, Node] = {}
    for n in nodes:
        sig = family_signature(n)
        cur = best.get(sig)
        if cur is None or n.score() > cur.score():
            best[sig] = n
    return sorted(best.values(), key=lambda n: n.score(), reverse=True)[:top_n]


def print_report(hits: list[Node], frontier: list[Node]):
    print("=" * 78)
    print("PLANNER-AGENT route search: erlotinib -> target")
    print("=" * 78)
    pool = hits if hits else frontier
    label = "EXACT-MATCH ROUTE FAMILIES" if hits else \
            "no exact match within max_depth -- best candidates so far"
    print(f"--- {label} ---\n")

    families = cluster(pool)
    if not families:
        print("No surviving routes -- widen --beam-width / --k / --max-depth.")
        return

    for i, node in enumerate(families, 1):
        print(f"[Route family {i}]  score={node.score():.3f}  steps={node.depth}  "
              f"route_confidence={node.route_confidence:.2f}  "
              f"exact_match={is_exact_target(node.mol)}")
        for j, s in enumerate(node.path, 1):
            print(f"    {j}. {s.transform_label}  ({s.family}, source={s.source}, "
                  f"conf={s.confidence:.2f})")
            print(f"       basis: {s.literature_basis}")
            print(f"       smarts: {s.smarts}")
        print(f"    final SMILES: {node.smiles}")
        print()


# --------------------------------------------------------------------------
# 8. CLI
# --------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--beam-width", type=int, default=5)
    ap.add_argument("--k", type=int, default=4, help="candidates proposed per node per iteration")
    ap.add_argument("--max-retries", type=int, default=2, help="bounded feedback-to-AI retries per step")
    ap.add_argument("--model", type=str, default="claude-sonnet-5")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--prune-below", type=float, default=0.05)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment first.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic()
    hits, frontier = search(
        client, args.model,
        max_depth=args.max_depth, beam_width=args.beam_width, k=args.k,
        max_retries=args.max_retries, temperature=args.temperature,
        prune_below=args.prune_below,
    )
    print_report(hits, frontier)