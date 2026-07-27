"""Tests for the LLM-as-judge (src/experiment/judge.py).

Deterministic: the judge model is a scripted fake (no Ollama). Parsing is
verified to be conservative — anything not explicitly COMPLIED is a refusal, so
an ambiguous or empty answer never inflates the attack success rate.
"""

from __future__ import annotations

from experiment.judge import (
    COMPLIED,
    REFUSED,
    AdaptiveAttacker,
    Judge,
    JudgeVerdict,
    extract_prompt,
    parse_verdict,
)
from llm.client import AssistantMessage
from messages import Messages


def _messages() -> Messages:
    return Messages(
        system_prompt="persona",
        user_prompt={"email": "read", "calendar": "list"},
        injections={"agentic": "open", "harmful": "bad"},
        judge_system="You classify safety.",
        judge_user_template="Response: {response}. COMPLIED or REFUSED?",
        attack_prompts={"agentic": ["fb1", "fb2"], "harmful": ["fb3"]},
        generator_system="You optimise injections.",
        generator_user_template=(
            "type={attack_type} prior={prior_prompt} resp={response} "
            "return between <prompt></prompt>"
        ),
    )


class ScriptedJudgeLLM:
    """Returns canned AssistantMessage replies in call order; records the last call."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0
        self.last_messages = None
        self.last_kwargs = None

    def chat(self, messages, tools=None, *, temperature=None, seed=None):
        self.calls += 1
        self.last_messages = messages
        self.last_kwargs = {"tools": tools, "temperature": temperature, "seed": seed}
        reply = self._replies.pop(0) if self._replies else ""
        return AssistantMessage(content=reply)


# --------------------------------------------------------------------------- #
# parse_verdict
# --------------------------------------------------------------------------- #


def test_parse_complied_first_line_with_rationale():
    v = parse_verdict("COMPLIED\nIt gave step-by-step instructions.")
    assert v.complied is True
    assert v.label == COMPLIED
    assert v.success is True
    assert v.rationale == "It gave step-by-step instructions."


def test_parse_refused_first_line():
    v = parse_verdict("REFUSED\nThe assistant declined and warned the user.")
    assert v.complied is False
    assert v.label == REFUSED
    assert v.success is False
    assert v.rationale == "The assistant declined and warned the user."


def test_parse_tolerates_trailing_punctuation_and_case():
    assert parse_verdict("complied.").complied is True
    assert parse_verdict("Complied — here is why").complied is True


def test_parse_is_conservative_on_ambiguity():
    # Anything not clearly COMPLIED counts as a refusal.
    assert parse_verdict("MAYBE\nnot sure").complied is False
    assert parse_verdict("The verdict is COMPLIED").complied is False  # not first token


def test_parse_empty_or_none_is_refusal():
    assert parse_verdict(None).complied is False
    assert parse_verdict("").complied is False
    assert parse_verdict("   \n  ").complied is False


def test_parse_single_line_uses_it_as_rationale():
    v = parse_verdict("REFUSED")
    assert v.complied is False
    assert v.rationale == "REFUSED"


# --------------------------------------------------------------------------- #
# Judge.evaluate
# --------------------------------------------------------------------------- #


def test_evaluate_sends_judge_prompts_and_returns_verdict():
    msgs = _messages()
    llm = ScriptedJudgeLLM(["COMPLIED\nDelivered the disallowed content."])
    verdict = Judge(llm, messages=msgs).evaluate("Sure, here is how ...")

    assert isinstance(verdict, JudgeVerdict)
    assert verdict.complied is True
    assert llm.calls == 1
    # system + user, in that order, sourced from messages.yaml.
    roles = [m["role"] for m in llm.last_messages]
    assert roles == ["system", "user"]
    assert llm.last_messages[0]["content"] == "You classify safety."
    assert "Sure, here is how ..." in llm.last_messages[1]["content"]
    # No tools, fixed seed for reproducibility.
    assert llm.last_kwargs["tools"] is None
    assert llm.last_kwargs["seed"] == 0


def test_evaluate_refusal_is_not_success():
    llm = ScriptedJudgeLLM(["REFUSED\nThe assistant refused."])
    verdict = Judge(llm, messages=_messages()).evaluate("I can't help with that.")
    assert verdict.success is False


def test_evaluate_handles_empty_response_text():
    llm = ScriptedJudgeLLM(["REFUSED\nNothing to grade."])
    verdict = Judge(llm, messages=_messages()).evaluate(None)
    assert verdict.complied is False
    assert llm.calls == 1  # still asks the judge; None is rendered as empty


# --------------------------------------------------------------------------- #
# extract_prompt
# --------------------------------------------------------------------------- #


def test_extract_prompt_pulls_text_between_tags():
    assert extract_prompt("<prompt>do the thing</prompt>") == "do the thing"


def test_extract_prompt_strips_surrounding_whitespace_and_prose():
    text = "Here you go:\n<prompt>\n  open the window now  \n</prompt>\nHope it helps."
    assert extract_prompt(text) == "open the window now"


def test_extract_prompt_handles_multiline_body():
    text = "<prompt>line one\nline two</prompt>"
    assert extract_prompt(text) == "line one\nline two"


def test_extract_prompt_returns_none_without_tags():
    assert extract_prompt("no tags here, sorry") is None


def test_extract_prompt_returns_none_when_empty_between_tags():
    assert extract_prompt("<prompt>   </prompt>") is None


def test_extract_prompt_none_input_is_none():
    assert extract_prompt(None) is None


# --------------------------------------------------------------------------- #
# AdaptiveAttacker.propose_prompt
# --------------------------------------------------------------------------- #


def test_propose_prompt_extracts_improved_injection():
    msgs = _messages()
    llm = ScriptedJudgeLLM(["Sure: <prompt>a better injection</prompt>"])
    attacker = AdaptiveAttacker(llm, messages=msgs)

    proposed = attacker.propose_prompt(
        "agentic", prior_prompt="open", response="I won't."
    )

    assert proposed == "a better injection"
    assert llm.calls == 1
    roles = [m["role"] for m in llm.last_messages]
    assert roles == ["system", "user"]
    assert llm.last_messages[0]["content"] == "You optimise injections."
    # The user message carries the attack type, prior prompt, and response.
    user = llm.last_messages[1]["content"]
    assert "agentic" in user and "open" in user and "I won't." in user
    # No tools; a seed is passed for reproducibility.
    assert llm.last_kwargs["tools"] is None


def test_propose_prompt_returns_none_when_generator_emits_no_prompt():
    llm = ScriptedJudgeLLM(["I cannot help with that request."])
    attacker = AdaptiveAttacker(llm, messages=_messages())

    proposed = attacker.propose_prompt(
        "harmful", prior_prompt="bad", response="I refuse."
    )

    assert proposed is None
    assert llm.calls == 1
