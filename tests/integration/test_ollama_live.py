"""Prompt adjustment against the real local model (opt in: pytest -m llm).

Needs Ollama running on LlmConfig().ollama_url with LlmConfig().model pulled. Each sample goes
through the same system prompt, cleanup and fidelity check the app uses, and must pass the
check: the model may restructure, but not drop or add information, nor translate.
"""

from __future__ import annotations

import pytest

from app.contracts import LlmConfig
from app.llm.ollama import OllamaClient
from app.llm.postprocess import check_fidelity, clean_output
from app.llm.prompt_text import SYSTEM_PROMPT, build_user_message, detect_language

pytestmark = pytest.mark.llm

SAMPLES = {
    "english": (
        "i need a python script that reads all the csv files in data/raw, merges them on the user_id column, "
        "drops rows where age is missing and writes the result to data/clean/users.parquet. it should log how "
        "many rows were dropped. use pandas 2.x, no polars. if possible make it run in under 10 seconds for 1 "
        "million rows"
    ),
    "romanian": (
        "vreau sa imi faci o functie care primeste o lista de comenzi si calculeaza totalul cu TVA de 19%, "
        "rotunjit la 2 zecimale. comenzile anulate nu se pun la socoteala. daca lista e goala sa returneze 0. "
        "numele functiei sa fie calculeaza_total si sa aiba teste"
    ),
    "mixed": (
        "am o problema cu build-ul: cand rulez build.ps1 pe windows imi da eroarea ModuleNotFoundError pentru "
        "pefile. am incercat pip install -r requirements-dev.txt dar tot nu merge. vreau sa stiu de ce si cum "
        "fixez, si sa nu schimb versiunea de pyinstaller 6.22.3"
    ),
    "with_code": (
        "why does this return None instead of the config? it worked before i added the cache\n"
        "```python\n"
        "def load(path):\n"
        "    if path in _cache:\n"
        "        _cache[path]\n"
        "    cfg = parse(path)\n"
        "    _cache[path] = cfg\n"
        "    return cfg\n"
        "```\n"
        "keep the function signature the same"
    ),
    "short": "fix the typo in README line 12 pls",
    "long_list": (
        "we are building a todo api in fastapi. endpoints: create todo, list todos with pagination (page size 20), "
        "mark done, delete. store in postgres 16 using sqlalchemy 2. auth with jwt, tokens expire after 1h. dont use "
        "alembic yet. tests with pytest, coverage at least 80%. deploy target is a single docker container, python "
        "3.12. also maybe add rate limiting 100 req/min per user if its not too much work"
    ),
    "romanian_long": (
        "am nevoie de un script care sa ia toate pozele din folderul Poze/2024, sa le redimensioneze la maxim 1920px "
        "pe latura lunga, sa pastreze datele EXIF si sa le salveze in Poze/2024_mici. fisierele care nu sunt jpg sau "
        "png sa fie ignorate. foloseste Pillow. nu suprascrie fisierele care exista deja"
    ),
}


@pytest.fixture(scope="module")
def client():
    cfg = LlmConfig()
    c = OllamaClient(cfg.ollama_url, cfg.model, keep_alive="5m", timeout_s=300)
    health = c.health()
    if not health.ok:
        pytest.skip(health.message)
    return c


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_rewrite_keeps_the_information(client, name):
    draft = SAMPLES[name]
    raw = client.chat(SYSTEM_PROMPT, build_user_message(draft)).text
    out = clean_output(raw, draft)
    report = check_fidelity(draft, out)
    assert report.ok, f"{report.summary()}\n--- output ---\n{out}"
    assert not out.startswith("```")
    if name == "short":
        assert "##" not in out
    if detect_language(draft) == "ro":  # headings follow the draft's language
        assert "## Goal" not in out and "## Requirements" not in out
    else:
        assert "## Scop" not in out and "## Cerințe" not in out
