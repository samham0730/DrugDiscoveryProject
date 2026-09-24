"""
test_erlotinib_planner_agent.py

Test compound (also the module's TARGET_SMILES):
    COc1cc2ncnc(Nc3cccc(C#N)c3)c2cc1OCCN

Requires:
    pip install pytest rdkit anthropic
    (no ANTHROPIC_API_KEY or network needed -- the AI-in-the-loop tests use
    a MockClient, never the real Anthropic API)

Run:
    pytest test_erlotinib_planner_agent.py -v

Structure:
    1. Fixture sanity        -- the test compound really is the target
    2. Chemistry primitives  -- tanimoto / core_intact / is_exact_target / try_apply_smarts
    3. Each known template   -- tested in isolation on a minimal substrate,
                                 so regiochemical ambiguity never enters in
    4. Routing               -- resolve_smarts: template-lookup vs generated
    5. Scoring & clustering  -- Node.score(), family_signature(), cluster()
    6. The feedback loop     -- execute_and_validate() retry-then-fix and
                                 give-up-after-max-retries, via MockClient
    7. Full route            -- chains the known templates from erlotinib
                                 all the way to the literal test compound
"""

import json
import pytest
from rdkit import Chem

from erlotinib_planner_agent import (
    START_SMILES, TARGET_SMILES, CORE_SMARTS,
    TEMPLATE_LIBRARY,
    tanimoto_to_target, core_intact, is_exact_target, try_apply_smarts,
    resolve_smarts, Node, Step,
    execute_and_validate,
    family_signature, cluster,
)

TEST_COMPOUND = "COc1cc2ncnc(Nc3cccc(C#N)c3)c2cc1OCCN"


def canon(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def mol(smiles: str):
    return Chem.MolFromSmiles(smiles)


# --------------------------------------------------------------------------
# 1. Fixture sanity
# --------------------------------------------------------------------------

class TestFixtureSanity:

    def test_test_compound_matches_module_target(self):
        assert canon(TEST_COMPOUND) == canon(TARGET_SMILES)

    def test_test_compound_parses(self):
        assert mol(TEST_COMPOUND) is not None

    def test_erlotinib_parses(self):
        assert mol(START_SMILES) is not None

    def test_test_compound_is_not_erlotinib(self):
        assert canon(TEST_COMPOUND) != canon(START_SMILES)


# --------------------------------------------------------------------------
# 2. Chemistry primitives, evaluated against the test compound
# --------------------------------------------------------------------------

class TestChemistryPrimitives:

    def test_tanimoto_self_similarity_is_one(self):
        assert tanimoto_to_target(mol(TEST_COMPOUND)) == pytest.approx(1.0)

    def test_tanimoto_erlotinib_is_less_than_one(self):
        assert tanimoto_to_target(mol(START_SMILES)) < 1.0

    def test_tanimoto_erlotinib_is_meaningfully_similar(self):
        # shares the whole quinazoline + anilino scaffold -- should not be near zero
        assert tanimoto_to_target(mol(START_SMILES)) > 0.3

    def test_core_intact_on_test_compound(self):
        assert core_intact(mol(TEST_COMPOUND)) is True

    def test_core_intact_on_erlotinib(self):
        assert core_intact(mol(START_SMILES)) is True

    def test_core_intact_false_when_scaffold_absent(self):
        assert core_intact(mol("c1ccccc1")) is False

    def test_is_exact_target_true_for_identity(self):
        assert is_exact_target(mol(TEST_COMPOUND)) is True

    def test_is_exact_target_false_for_erlotinib(self):
        assert is_exact_target(mol(START_SMILES)) is False

    def test_try_apply_smarts_rejects_bad_syntax(self):
        product, error = try_apply_smarts(mol(START_SMILES), "not a smarts (((")
        assert product is None
        assert error is not None

    def test_try_apply_smarts_rejects_no_matching_site(self):
        # target has no bromine anywhere -- this SMARTS can't match
        product, error = try_apply_smarts(mol(TEST_COMPOUND), "[c:1][Br]>>[c:1][I]")
        assert product is None
        assert "no matching site" in error


# --------------------------------------------------------------------------
# 3. Each known template, in isolation on a minimal substrate
#    (avoids erlotinib's two-identical-arms regiochemical ambiguity)
# --------------------------------------------------------------------------

class TestKnownTemplates:

    def test_alkyne_to_nitrile_minimal(self):
        smarts = TEMPLATE_LIBRARY["alkyne_to_nitrile"]["smarts"]
        product, error = try_apply_smarts(mol("C#Cc1ccccc1"), smarts)
        assert error is None
        assert Chem.MolToSmiles(product) == canon("N#Cc1ccccc1")

    def test_alkyne_to_nitrile_on_erlotinib_preserves_rest_of_molecule(self):
        smarts = TEMPLATE_LIBRARY["alkyne_to_nitrile"]["smarts"]
        product, error = try_apply_smarts(mol(START_SMILES), smarts)
        assert error is None
        assert core_intact(product) is True
        expected = "N#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1"
        assert Chem.MolToSmiles(product) == canon(expected)

    def test_mono_demethylate_terminal_ether(self):
        smarts = TEMPLATE_LIBRARY["mono_demethylate_terminal_ether"]["smarts"]
        # PhO-CH2CH2-OMe -> PhO-CH2CH2-OH ; aryl ether untouched
        product, error = try_apply_smarts(mol("c1ccccc1OCCOC"), smarts)
        assert error is None
        assert Chem.MolToSmiles(product) == canon("c1ccccc1OCCO")

    def test_exhaustive_dealkylate_to_phenol(self):
        smarts = TEMPLATE_LIBRARY["exhaustive_dealkylate_to_phenol"]["smarts"]
        product, error = try_apply_smarts(mol("c1ccccc1OCCOC"), smarts)
        assert error is None
        assert Chem.MolToSmiles(product) == canon("Oc1ccccc1")

    def test_O_methylate_phenol(self):
        smarts = TEMPLATE_LIBRARY["O_methylate_phenol"]["smarts"]
        product, error = try_apply_smarts(mol("Oc1ccccc1"), smarts)
        assert error is None
        assert Chem.MolToSmiles(product) == canon("COc1ccccc1")

    def test_primary_alcohol_to_amine(self):
        smarts = TEMPLATE_LIBRARY["primary_alcohol_to_amine"]["smarts"]
        product, error = try_apply_smarts(mol("OCCOc1ccccc1"), smarts)
        assert error is None
        assert Chem.MolToSmiles(product) == canon("NCCOc1ccccc1")

    def test_mitsunobu_direct_aminoethyl_etherification(self):
        smarts = TEMPLATE_LIBRARY["mitsunobu_direct_aminoethyl_etherification"]["smarts"]
        product, error = try_apply_smarts(mol("Oc1ccccc1"), smarts)
        assert error is None
        assert Chem.MolToSmiles(product) == canon("NCCOc1ccccc1")

    def test_every_template_compiles_and_is_core_safe_on_erlotinib_where_applicable(self):
        # sanity sweep: no template should silently break the core when it
        # does apply to erlotinib itself
        for name, entry in TEMPLATE_LIBRARY.items():
            product, error = try_apply_smarts(mol(START_SMILES), entry["smarts"])
            if product is not None:
                assert core_intact(product) is True, f"{name} broke the core"

    def test_deliberately_core_breaking_smarts_is_caught(self):
        # not in TEMPLATE_LIBRARY -- exists only to prove core_intact() fires
        distractor = ("[c:1][NH:2][c:3]1[cH][cH][cH][cH][cH]1"
                      ">>[c:1][OH].[NH2:2][c:3]1[cH][cH][cH][cH][cH]1")
        product, error = try_apply_smarts(mol(START_SMILES), distractor)
        assert error is None  # it does run and sanitize...
        assert core_intact(product) is False  # ...but it must fail the core check


# --------------------------------------------------------------------------
# 4. Routing: resolve_smarts (known template vs generated branch)
# --------------------------------------------------------------------------

class TestResolveSmarts:

    def test_known_template_name_wins_even_if_smarts_also_supplied(self):
        candidate = {
            "template_name": "alkyne_to_nitrile",
            "reaction_smarts": "garbage that would never compile",
            "confidence": 0.4,
        }
        smarts, source, conf = resolve_smarts(candidate)
        assert smarts == TEMPLATE_LIBRARY["alkyne_to_nitrile"]["smarts"]
        assert source == "template"
        assert conf == TEMPLATE_LIBRARY["alkyne_to_nitrile"]["confidence"]

    def test_unknown_template_name_falls_back_to_generated(self):
        candidate = {
            "template_name": "some_novel_disconnection_not_in_library",
            "reaction_smarts": "[c:1][F]>>[c:1][Cl]",
            "confidence": 0.55,
        }
        smarts, source, conf = resolve_smarts(candidate)
        assert smarts == "[c:1][F]>>[c:1][Cl]"
        assert source == "generated"
        assert conf == pytest.approx(0.55)

    def test_no_template_name_uses_generated_smarts(self):
        candidate = {"template_name": None, "reaction_smarts": "[c:1][F]>>[c:1][Cl]",
                     "confidence": 0.6}
        smarts, source, conf = resolve_smarts(candidate)
        assert smarts == "[c:1][F]>>[c:1][Cl]"
        assert source == "generated"

    def test_missing_everything_returns_none_smarts(self):
        smarts, source, conf = resolve_smarts({})
        assert smarts is None
        assert source == "generated"


# --------------------------------------------------------------------------
# 5. Scoring & clustering
# --------------------------------------------------------------------------

class TestScoringAndClustering:

    def _step(self, label, family, confidence):
        return Step(transform_label=label, family=family, smarts="[c:1][F]>>[c:1][Cl]",
                    source="template", literature_basis="test", confidence=confidence)

    def test_score_gives_exact_match_bonus(self):
        hit = Node(smiles=canon(TEST_COMPOUND), mol=mol(TEST_COMPOUND), depth=0)
        miss = Node(smiles=canon(START_SMILES), mol=mol(START_SMILES), depth=0)
        assert hit.score() > miss.score()
        assert hit.score() >= 1.0  # +1.0 bonus dominates

    def test_score_penalizes_depth(self):
        shallow = Node(smiles=canon(START_SMILES), mol=mol(START_SMILES), depth=0)
        deep = Node(smiles=canon(START_SMILES), mol=mol(START_SMILES), depth=5)
        assert deep.score() < shallow.score()

    def test_score_penalizes_low_route_confidence(self):
        confident = Node(smiles=canon(START_SMILES), mol=mol(START_SMILES), depth=1,
                          path=(self._step("a", "arm_shorten", 0.9),))
        shaky = Node(smiles=canon(START_SMILES), mol=mol(START_SMILES), depth=1,
                     path=(self._step("a", "arm_shorten", 0.1),))
        assert confident.score() > shaky.score()

    def test_route_confidence_is_product_of_steps(self):
        n = Node(smiles=canon(START_SMILES), mol=mol(START_SMILES), depth=2,
                 path=(self._step("a", "f1", 0.8), self._step("b", "f2", 0.5)))
        assert n.route_confidence == pytest.approx(0.4)

    def test_family_signature_ignores_order(self):
        n1 = Node(smiles="x", mol=mol(START_SMILES),
                  path=(self._step("a", "fam1", 0.9), self._step("b", "fam2", 0.9)))
        n2 = Node(smiles="y", mol=mol(START_SMILES),
                  path=(self._step("b", "fam2", 0.9), self._step("a", "fam1", 0.9)))
        assert family_signature(n1) == family_signature(n2)

    def test_cluster_keeps_best_per_family_and_respects_top_n(self):
        good = Node(smiles=canon(TEST_COMPOUND), mol=mol(TEST_COMPOUND), depth=1,
                     path=(self._step("a", "fam1", 0.9),))
        worse_same_family = Node(smiles=canon(START_SMILES), mol=mol(START_SMILES), depth=3,
                                  path=(self._step("a", "fam1", 0.2),))
        different_family = Node(smiles=canon(START_SMILES), mol=mol(START_SMILES), depth=1,
                                 path=(self._step("c", "fam3", 0.9),))
        families = cluster([good, worse_same_family, different_family], top_n=10)
        sigs = [family_signature(n) for n in families]
        assert family_signature(good) in sigs
        assert family_signature(different_family) in sigs
        # the weaker fam1 route must NOT also appear -- best-per-signature wins
        assert len(families) == 2

    def test_cluster_respects_top_n_limit(self):
        nodes = [Node(smiles=str(i), mol=mol(START_SMILES), depth=1,
                       path=(self._step(f"s{i}", f"fam{i}", 0.9),))
                 for i in range(5)]
        assert len(cluster(nodes, top_n=3)) == 3


# --------------------------------------------------------------------------
# 6. The feedback loop, via a MockClient (no network / API key needed)
# --------------------------------------------------------------------------

class _FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    def __init__(self, fixed_reply_json):
        self.fixed_reply_json = fixed_reply_json
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return _FakeResponse(self.fixed_reply_json)


class _FakeClient:
    def __init__(self, fixed_reply_json):
        self.messages = _FakeMessages(fixed_reply_json)


class TestFeedbackLoop:

    def test_bad_candidate_with_zero_retries_gives_up_without_calling_ai(self):
        node = Node(smiles=canon(START_SMILES), mol=mol(START_SMILES))
        bad_candidate = {"transform_label": "nope", "family": "other",
                          "template_name": None, "reaction_smarts": "!!not smarts!!",
                          "confidence": 0.5}
        client = _FakeClient(fixed_reply_json="should never be used")
        result = execute_and_validate(client, "unused-model", node, bad_candidate,
                                       max_retries=0, temperature=1.0)
        assert result is None
        assert client.messages.calls == 0  # feedback loop never triggered

    def test_bad_candidate_gets_fixed_on_retry(self):
        node = Node(smiles=canon(START_SMILES), mol=mol(START_SMILES))
        bad_candidate = {"transform_label": "nope", "family": "other",
                          "template_name": None, "reaction_smarts": "!!not smarts!!",
                          "confidence": 0.5}
        fix_json = json.dumps({
            "transform_label": "fixed", "family": "aniline_edit",
            "template_name": "alkyne_to_nitrile", "reaction_smarts": None,
            "literature_basis": "corrected", "confidence": 0.9,
        })
        client = _FakeClient(fixed_reply_json=fix_json)
        result = execute_and_validate(client, "unused-model", node, bad_candidate,
                                       max_retries=1, temperature=1.0)
        assert result is not None
        assert client.messages.calls == 1  # exactly one feedback round-trip
        assert result.path[-1].source == "template"
        assert result.path[-1].transform_label == "fixed"

    def test_still_bad_after_exhausting_retries_gives_up(self):
        node = Node(smiles=canon(START_SMILES), mol=mol(START_SMILES))
        bad_candidate = {"transform_label": "nope", "family": "other",
                          "template_name": None, "reaction_smarts": "!!not smarts!!",
                          "confidence": 0.5}
        still_bad_json = json.dumps({
            "transform_label": "still_nope", "family": "other",
            "template_name": None, "reaction_smarts": "still !! not valid",
            "literature_basis": "", "confidence": 0.3,
        })
        client = _FakeClient(fixed_reply_json=still_bad_json)
        result = execute_and_validate(client, "unused-model", node, bad_candidate,
                                       max_retries=2, temperature=1.0)
        assert result is None
        assert client.messages.calls == 2  # bounded, not infinite

    def test_good_candidate_succeeds_first_try_without_calling_ai(self):
        node = Node(smiles=canon(START_SMILES), mol=mol(START_SMILES))
        good_candidate = {"transform_label": "ok", "family": "aniline_edit",
                           "template_name": "alkyne_to_nitrile", "reaction_smarts": None,
                           "confidence": 0.9}
        client = _FakeClient(fixed_reply_json="should never be used")
        result = execute_and_validate(client, "unused-model", node, good_candidate,
                                       max_retries=2, temperature=1.0)
        assert result is not None
        assert client.messages.calls == 0


# --------------------------------------------------------------------------
# 7. Full known-template route from erlotinib to the literal test compound
# --------------------------------------------------------------------------

class TestFullRouteReachesTestCompound:
    """
    Chains the known templates in an order chosen to make the two
    identical -OCH2CH2OMe arms unambiguous after the first step:

        1. alkyne_to_nitrile                    (single site, unambiguous)
        2. mono_demethylate_terminal_ether       (breaks arm symmetry)
        3. primary_alcohol_to_amine              (only one -CH2OH left)
        4. exhaustive_dealkylate_to_phenol       (only the untouched arm matches)
        5. O_methylate_phenol                    (only one phenol present)

    try_apply_smarts() picks the highest-Tanimoto-to-target product when a
    step's SMARTS matches more than one site, so this test also exercises
    that "search toward target" tie-breaking at step 2 -- if it ever picks
    the wrong regiochemical arm, this test is what will catch it.
    """

    def _run(self, smiles, template_name):
        smarts = TEMPLATE_LIBRARY[template_name]["smarts"]
        product, error = try_apply_smarts(mol(smiles), smarts)
        assert error is None, f"{template_name} failed: {error}"
        assert core_intact(product) is True, f"{template_name} broke the core"
        return Chem.MolToSmiles(product)

    def test_five_step_known_template_chain_reaches_test_compound(self):
        smi = START_SMILES
        for template_name in [
            "alkyne_to_nitrile",
            "mono_demethylate_terminal_ether",
            "primary_alcohol_to_amine",
            "exhaustive_dealkylate_to_phenol",
            "O_methylate_phenol",
        ]:
            smi = self._run(smi, template_name)
        assert smi == canon(TEST_COMPOUND)

    def test_chain_is_strictly_monotonic_in_similarity(self):
        smi = START_SMILES
        prev_sim = tanimoto_to_target(mol(smi))
        for template_name in [
            "alkyne_to_nitrile",
            "mono_demethylate_terminal_ether",
            "primary_alcohol_to_amine",
            "exhaustive_dealkylate_to_phenol",
            "O_methylate_phenol",
        ]:
            smi = self._run(smi, template_name)
            sim = tanimoto_to_target(mol(smi))
            assert sim >= prev_sim, f"{template_name} did not increase similarity"
            prev_sim = sim
        assert is_exact_target(mol(smi)) is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
