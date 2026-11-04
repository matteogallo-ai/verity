"""verity.eval.perturb: a supported claim becomes a false but plausible one, deterministically."""

from __future__ import annotations

import pytest

from verity.eval.perturb import Kind, perturb

EVIDENCE = (
    "Total revenue for fiscal year 2024 was $482.0 million, an increase of 18.4% compared "
    "to $407.1 million in fiscal year 2023. Revenue from Enterprise Cloud grew; revenue "
    "from Managed Services was $128.6 million. The policy took effect in March 2024 and "
    "is reviewed by the Security Council."
)


def test_number_changes_one_figure_to_a_value_absent_from_the_evidence() -> None:
    p = perturb("Total revenue in FY2024 was $482.0 million.", Kind.NUMBER, EVIDENCE)
    assert p is not None
    assert p.old == "482.0" and p.new not in EVIDENCE
    assert p.perturbed == f"Total revenue in FY2024 was ${p.new} million."
    assert len(p.new.split(".")[1]) == 1  # same precision as the original figure


def test_number_skips_years_and_ids_glued_to_letters() -> None:
    # 2024 is a year (date kind) and FY2024 an identifier: neither is "a number".
    assert perturb("The FY2024 plan was adopted in 2024.", Kind.NUMBER, EVIDENCE) is None


def test_number_keeps_thousands_separators() -> None:
    p = perturb("The fund holds 1,200 accounts.", Kind.NUMBER, EVIDENCE)
    assert p is not None and p.old == "1,200"
    assert "," in p.new and p.new != p.old


def test_date_shifts_the_month_before_the_year() -> None:
    p = perturb("The policy took effect in March 2024.", Kind.DATE, EVIDENCE)
    assert p is not None and p.old == "March" and p.new not in ("March", "")
    assert p.perturbed == f"The policy took effect in {p.new} 2024."


def test_date_shifts_a_year_to_one_the_evidence_does_not_hold() -> None:
    p = perturb("Revenue grew in fiscal year 2024.", Kind.DATE, EVIDENCE)
    assert p is not None and p.old == "2024" and p.new not in EVIDENCE


def test_negation_inserts_not_after_the_first_auxiliary() -> None:
    p = perturb("Revenue was up 18.4% year over year.", Kind.NEGATION, EVIDENCE)
    assert p is not None and p.perturbed == "Revenue was not up 18.4% year over year."
    p = perturb("Customers can export their data.", Kind.NEGATION, EVIDENCE)
    assert p is not None and p.perturbed == "Customers cannot export their data."


def test_negation_leaves_an_already_negative_claim_alone() -> None:
    # Removing a negation is not reliably grammatical; double negation is not plausible.
    assert perturb("Backups are not encrypted.", Kind.NEGATION, EVIDENCE) is None
    assert perturb("Revenue never fell.", Kind.NEGATION, EVIDENCE) is None


ENTITIES = {
    "person": ["Dana Whitfield", "Robert Lang", "Lena Park"],
    "company": ["Acme", "Northwind"],
    "city": ["London", "Paris", "Frankfurt"],
    "segment": ["Enterprise Cloud", "Managed Services"],
    "edition": ["Enterprise", "Professional"],
}


def test_entity_swaps_a_name_for_one_of_the_same_type_from_the_evidence() -> None:
    p = perturb("Revenue from Enterprise Cloud fell.", Kind.ENTITY, EVIDENCE, ENTITIES)
    # "Enterprise Cloud" is a segment, not the edition "Enterprise": longest name first.
    assert p is not None and p.old == "Enterprise Cloud" and p.new == "Managed Services"


def test_entity_prefers_a_distractor_the_evidence_holds() -> None:
    evidence = "The limit for Paris is US$300; the limit for London is US$325."
    p = perturb("The hotel limit for London is US$325.", Kind.ENTITY, evidence, ENTITIES)
    assert p is not None and p.new == "Paris"  # Frankfurt is absent from the evidence


def test_entity_never_crosses_types() -> None:
    p = perturb("Dana Whitfield succeeded Robert Lang.", Kind.ENTITY, "", ENTITIES)
    assert p is not None and p.new == "Lena Park"  # the only person not in the claim


def test_entity_keeps_the_possessive_and_whole_words() -> None:
    p = perturb("Northwind's order form was signed.", Kind.ENTITY, "", ENTITIES)
    assert p is not None and p.perturbed == "Acme's order form was signed."
    assert perturb("Acmeville hosts the event.", Kind.ENTITY, "", ENTITIES) is None


def test_entity_needs_a_known_name_in_the_claim() -> None:
    assert perturb("Revenue grew by a lot.", Kind.ENTITY, EVIDENCE, ENTITIES) is None
    assert perturb("Northwind signed.", Kind.ENTITY, EVIDENCE) is None  # no list given


@pytest.mark.parametrize("kind", list(Kind))
def test_empty_claim_has_no_perturbation(kind: Kind) -> None:
    assert perturb("", kind, EVIDENCE) is None


@pytest.mark.parametrize("kind", list(Kind))
def test_perturbation_is_deterministic(kind: Kind) -> None:
    claim = "Enterprise Cloud revenue was $312.7 million in March 2024."
    assert perturb(claim, kind, EVIDENCE, ENTITIES) == perturb(claim, kind, EVIDENCE, ENTITIES)


def test_number_keeps_a_percentage_at_most_100() -> None:
    p = perturb("Gross margin was 71.8%.", Kind.NUMBER, EVIDENCE)
    assert p is not None and float(p.new) <= 100


def test_number_leaves_a_figure_that_restates_a_number_word() -> None:
    # "three (3) years" -> "three (9) years" contradicts itself: not plausible.
    assert perturb("Obligations survive for three (3) years.", Kind.NUMBER, EVIDENCE) is None


def test_number_leaves_plan_names_and_days_of_the_month() -> None:
    assert perturb("Acme matches the 401(k) plan.", Kind.NUMBER, EVIDENCE) is None
    assert perturb("The quarter ended on March 31, 2025.", Kind.NUMBER, EVIDENCE) is None


def test_entity_leaves_a_name_that_is_part_of_a_longer_name() -> None:
    # "Northwind Logistics" and "Standard Contractual Clauses" are other names.
    assert perturb("Northwind Logistics subscribes.", Kind.ENTITY, "", ENTITIES) is None
    assert perturb("Transfers use the Enterprise Cloud Clauses.", Kind.ENTITY, "", ENTITIES) is None
    assert perturb("It is the Big Enterprise plan.", Kind.ENTITY, "", ENTITIES) is None


def test_entity_after_a_capitalised_function_word_is_still_a_name() -> None:
    p = perturb("If Northwind objects, The Board decides.", Kind.ENTITY, "", ENTITIES)
    assert p is not None and p.perturbed == "If Acme objects, The Board decides."


@pytest.mark.parametrize(
    "claim",
    [
        "Employees have fifteen days of paid leave.",  # "fifteen" is not a participle
        "The company has children's benefits.",
        "Staff in at-will roles receive notice.",  # "will" inside a hyphenated word
        "Managers need to do a quarterly review.",  # "do" is the main verb after "to"
    ],
)
def test_negation_skips_words_that_are_not_auxiliaries_here(claim: str) -> None:
    assert perturb(claim, Kind.NEGATION, EVIDENCE) is None


def test_negation_keeps_a_perfect_auxiliary_before_a_participle() -> None:
    p = perturb("The policy has been updated.", Kind.NEGATION, EVIDENCE)
    assert p is not None and p.perturbed == "The policy has not been updated."


ROLES = {
    # An edition is listed with its noun: "Business" alone is a common word.
    "edition": ["Business edition", "Enterprise edition", "Standard edition"],
    "role": ["Director", "Vice President"],
}


@pytest.mark.parametrize(
    "claim",
    [
        "Business days exclude holidays.",  # a common word, not the edition
        "Standard shipping applies.",
        "The Director of Finance approves.",  # a title, not the role alone
    ],
)
def test_entity_skips_common_words_and_titles(claim: str) -> None:
    assert perturb(claim, Kind.ENTITY, "", ROLES) is None


def test_entity_swaps_an_edition_named_as_one() -> None:
    p = perturb("The Business edition costs more.", Kind.ENTITY, "", ROLES)
    assert p is not None and p.perturbed == "The Enterprise edition costs more."


@pytest.mark.parametrize(
    "claim", ["See Section 4.2 for details.", "Call 555-0100 for support.", "Article 12 applies."]
)
def test_number_leaves_section_numbers_and_phone_numbers(claim: str) -> None:
    assert perturb(claim, Kind.NUMBER, EVIDENCE) is None
