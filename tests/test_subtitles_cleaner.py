"""GPT-постобработка субтитров (сеть замокана)."""

from __future__ import annotations

import openai
import pytest
from openai.error import AuthenticationError

from analysis import subtitles_cleaner as sc


class FakeCompletion:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        item = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(item, Exception):
            raise item
        return {"choices": [{"message": {"content": item}}]}


def install(monkeypatch, answers):
    fake = FakeCompletion(answers)
    monkeypatch.setattr(openai.ChatCompletion, "create", staticmethod(fake))
    monkeypatch.setattr(sc.time, "sleep", lambda *_: None)
    return fake


class TestCleanAndCorrectText:
    def test_empty_input_short_circuits(self, monkeypatch):
        fake = install(monkeypatch, ["не должно вызваться"])
        assert sc.clean_and_correct_text("", api_key="k") == ""
        assert fake.calls == []

    def test_result_is_uppercased_without_punctuation(self, monkeypatch):
        install(monkeypatch, ["привет, мир!"])

        result = sc.clean_and_correct_text("привет мир", api_key="k")

        assert result == "ПРИВЕТ МИР"

    def test_highlight_tags_survive(self, monkeypatch):
        install(monkeypatch, ["это <hl>важно</hl> очень"])

        result = sc.clean_and_correct_text("это важно очень", api_key="k")

        assert "<hl>ВАЖНО</hl>" in result

    def test_h1_tags_are_normalised(self, monkeypatch):
        install(monkeypatch, ["это <h1>ключ</h1>"])

        assert "<hl>КЛЮЧ</hl>" in sc.clean_and_correct_text("x", api_key="k")

    def test_emoji_is_preserved(self, monkeypatch):
        install(monkeypatch, ["огонь 🔥"])

        assert "🔥" in sc.clean_and_correct_text("огонь", api_key="k")

    def test_custom_prompt_receives_text(self, monkeypatch):
        fake = install(monkeypatch, ["ОК"])

        sc.clean_and_correct_text("исходник", api_key="k", prompt_template="Правь: {text}")

        assert "Правь: исходник" in fake.calls[0]["messages"][0]["content"]

    def test_language_instruction_is_injected(self, monkeypatch):
        fake = install(monkeypatch, ["ОК"])

        sc.clean_and_correct_text(
            "text", api_key="k", language="ru",
            prompt_template="{language_instructions}\n{text}",
        )

        assert "ru" in fake.calls[0]["messages"][0]["content"]

    def test_model_name_is_passed(self, monkeypatch):
        fake = install(monkeypatch, ["ОК"])

        sc.clean_and_correct_text("text", api_key="k", model="my-model")

        assert fake.calls[0]["model"] == "my-model"

    def test_meta_ad_mode_requires_template(self, monkeypatch):
        install(monkeypatch, ["ОК"])

        with pytest.raises(ValueError, match="meta_ad"):
            sc.clean_and_correct_text("text", api_key="k", subtitle_mode="meta_ad")

    def test_meta_ad_mode_uses_its_template(self, monkeypatch):
        fake = install(monkeypatch, ["ОК"])

        sc.clean_and_correct_text(
            "text", api_key="k", subtitle_mode="meta_ad",
            meta_ad_prompt_template="МЕТА: {text}",
        )

        assert fake.calls[0]["messages"][0]["content"].startswith("МЕТА:")

    def test_auth_error_is_explicit(self, monkeypatch):
        install(monkeypatch, [AuthenticationError("bad")])

        with pytest.raises(ValueError, match="авторизации"):
            sc.clean_and_correct_text("text", api_key="k")

    def test_retries_then_gives_up(self, monkeypatch):
        fake = install(monkeypatch, [RuntimeError("сеть упала")])

        with pytest.raises(RuntimeError):
            sc.clean_and_correct_text("text", api_key="k", max_retries=2)

        assert len(fake.calls) == 2
