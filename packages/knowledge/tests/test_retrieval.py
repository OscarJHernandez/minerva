from minerva_knowledge import KnowledgeBase, SearchQuery


def ids(kb: KnowledgeBase, **kw) -> list[str]:  # type: ignore[no-untyped-def]
    return [e.id for e in kb.search(SearchQuery(**kw))]


def test_text_search_finds_gaussian_first(kb: KnowledgeBase) -> None:
    assert ids(kb, text="gaussian integral")[0] == "int.gaussian"


def test_stemming_matches_exponential(kb: KnowledgeBase) -> None:
    assert "trig.euler" in ids(kb, text="euler exponentials")


def test_structural_features_retrieve_log_rules(kb: KnowledgeBase) -> None:
    found = ids(kb, features={"f:log"}, rules_only=True)
    assert {"log.product", "log.quotient", "log.power"} <= set(found)
    assert all(kb.get(i).rule is not None for i in found)  # type: ignore[union-attr]


def test_filters(kb: KnowledgeBase) -> None:
    assert all(
        kb.get(i).chapter == "physics.relativity" for i in ids(kb, chapter="physics.relativity")
    )  # type: ignore[union-attr]
    assert all(kb.get(i).kind == "series" for i in ids(kb, kind="series", limit=50))  # type: ignore[union-attr]


def test_hostile_query_text_is_harmless(kb: KnowledgeBase) -> None:
    for q in ['"; DROP TABLE entries; --', "NEAR(", "* OR 1=1", "'''", "a" * 5000]:
        ids(kb, text=q)  # must not raise
    assert ids(kb, text="") != []  # empty query lists entries


def test_no_match_for_nonsense(kb: KnowledgeBase) -> None:
    assert ids(kb, text="zzzzqqqq") == []
