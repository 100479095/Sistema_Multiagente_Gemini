"""Unit tests for the post-cutoff corpus and the hallucination wrappers.

These run without Ollama and without touching the database.
"""
from __future__ import annotations

import random
from datetime import date

import pytest

from attacker.hallucination_wrapper import (
    AuthorityWrapper,
    HallucinationWrapper,
    InterleavedWrapper,
    PrefixWrapper,
    WrapResult,
    build_wrapper,
)
from attacker.post_cutoff_corpus import (
    POST_CUTOFF_FACTS,
    PostCutoffFact,
    get_fact_by_id,
    random_fact,
)

LLAMA2_CUTOFF = date(2023, 7, 20)


# ---------------------------------------------------------------- corpus

def test_corpus_has_sixteen_facts():
    assert len(POST_CUTOFF_FACTS) == 16


def test_all_facts_strictly_post_cutoff():
    for fact in POST_CUTOFF_FACTS:
        d = date.fromisoformat(fact.date)
        assert d > LLAMA2_CUTOFF, f"{fact.id} dated {fact.date} is not post-cutoff"


def test_fact_ids_are_unique():
    ids = [f.id for f in POST_CUTOFF_FACTS]
    assert len(ids) == len(set(ids))


def test_facts_cover_multiple_categories():
    cats = {f.category for f in POST_CUTOFF_FACTS}
    assert cats >= {"tech", "politics", "sports", "science", "culture"}


def test_get_fact_by_id_finds_existing():
    fact = POST_CUTOFF_FACTS[0]
    assert get_fact_by_id(fact.id) == fact


def test_get_fact_by_id_unknown_returns_none():
    assert get_fact_by_id("DOES_NOT_EXIST") is None


def test_random_fact_with_seed_is_deterministic():
    a = random_fact(seed=42)
    b = random_fact(seed=42)
    assert a == b
    assert isinstance(a, PostCutoffFact)


def test_random_fact_different_seeds_can_differ():
    seen = {random_fact(seed=s).id for s in range(50)}
    assert len(seen) > 1


# -------------------------------------------------------------- wrappers

@pytest.fixture
def seeded_rng():
    return random.Random(123)


@pytest.mark.parametrize(
    "wrapper_cls,expected_name",
    [
        (PrefixWrapper, "prefix"),
        (InterleavedWrapper, "interleaved"),
        (AuthorityWrapper, "authority"),
    ],
)
def test_wrapper_strategy_metadata(wrapper_cls, expected_name, seeded_rng):
    wrapper = wrapper_cls(rng=seeded_rng)
    result = wrapper.wrap("PAYLOAD_TOKEN_XYZ")
    assert isinstance(result, WrapResult)
    assert result.strategy == expected_name
    assert result.fact_id in {f.id for f in POST_CUTOFF_FACTS}
    assert result.fact_text  # non-empty


@pytest.mark.parametrize(
    "wrapper_cls",
    [PrefixWrapper, InterleavedWrapper, AuthorityWrapper],
)
def test_wrapper_preserves_payload(wrapper_cls, seeded_rng):
    payload = "PAYLOAD_TOKEN_XYZ"
    wrapper = wrapper_cls(rng=seeded_rng)
    result = wrapper.wrap(payload)
    assert payload in result.wrapped_text


@pytest.mark.parametrize(
    "wrapper_cls",
    [PrefixWrapper, InterleavedWrapper, AuthorityWrapper],
)
def test_wrapper_includes_some_fact_tokens(wrapper_cls, seeded_rng):
    payload = "PAYLOAD_TOKEN_XYZ"
    wrapper = wrapper_cls(rng=seeded_rng)
    result = wrapper.wrap(payload)
    # At least one non-trivial token from the fact must reappear in the wrap.
    fact_tokens = {
        t.strip(".,:;()[]\"' ").lower()
        for t in result.fact_text.split()
        if len(t) >= 5
    }
    wrapped_lower = result.wrapped_text.lower()
    assert any(t in wrapped_lower for t in fact_tokens)


def test_authority_wrapper_includes_date(seeded_rng):
    wrapper = AuthorityWrapper(rng=seeded_rng)
    result = wrapper.wrap("PAYLOAD_TOKEN")
    assert "[VERIFIED" in result.wrapped_text
    fact = get_fact_by_id(result.fact_id)
    assert fact is not None
    assert fact.date in result.wrapped_text


def test_interleaved_wrapper_payload_is_in_the_middle(seeded_rng):
    wrapper = InterleavedWrapper(rng=seeded_rng)
    result = wrapper.wrap("PAYLOAD_TOKEN")
    idx = result.wrapped_text.index("PAYLOAD_TOKEN")
    # Payload must not be at the very start or very end.
    assert 0 < idx < len(result.wrapped_text) - len("PAYLOAD_TOKEN")


# --------------------------------------------------------------- factory

def test_build_wrapper_none_returns_none():
    assert build_wrapper("none") is None
    assert build_wrapper(None) is None


def test_build_wrapper_unknown_raises():
    with pytest.raises(ValueError):
        build_wrapper("garbage")


@pytest.mark.parametrize("name", ["prefix", "interleaved", "authority"])
def test_build_wrapper_known_returns_subclass(name):
    wrapper = build_wrapper(name, seed=7)
    assert isinstance(wrapper, HallucinationWrapper)
    assert wrapper.strategy_name == name


def test_build_wrapper_seed_makes_wrap_deterministic():
    a = build_wrapper("prefix", seed=99).wrap("p")
    b = build_wrapper("prefix", seed=99).wrap("p")
    assert a == b
