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


# --- queries made of common words must not be answered with entries that only share "of" -------


def test_a_query_with_one_real_word_and_filler_matches_nothing_irrelevant(kb: KnowledgeBase) -> None:
    # The handbook has nothing about the brachistochrone; "of" is in more than half of the entries.
    assert ids(kb, text="curve of fastest descent") == []
    assert ids(kb, text="brachistochrone") == []


def test_words_that_are_in_most_entries_are_not_a_query(kb: KnowledgeBase) -> None:
    assert ids(kb, text="of the") == []
    assert ids(kb, text="'''") == []  # nothing in it can be searched for


def test_a_title_or_keyword_match_is_enough_for_two_words(kb: KnowledgeBase) -> None:
    found = ids(kb, text="Kepler Lagrangian")
    assert "method.mech.euler_lagrange" in found  # Lagrangian is one of its keywords
    assert "series.basel" not in ids(kb, text="Kepler problem Lagrangian")


def test_three_words_need_two_of_them(kb: KnowledgeBase) -> None:
    assert ids(kb, text="Kepler problem Lagrangian") == []  # each entry matches only one word
    assert ids(kb, text="euler lagrange equations")[0].startswith("method.mech.euler_lagrange")


def test_a_single_word_that_is_common_is_still_a_query(kb: KnowledgeBase) -> None:
    assert ids(kb, text="derivative") != []
    assert ids(kb, text="variational")[0].startswith("method.mech.")


def test_filler_does_not_hide_the_real_words(kb: KnowledgeBase) -> None:
    assert ids(kb, text="the calculus of variations")[:3] == ids(kb, text="calculus variations")[:3]
    assert ids(kb, text="Basel problem") == ["series.basel"]


def test_structural_hits_are_kept_whatever_the_words_say(kb: KnowledgeBase) -> None:
    # Identity retrieval passes text and features: a rule that fits by structure is a candidate even
    # when none of the words matches it.
    with_features = [
        e.id for e in kb.candidate_rules({"f:log"}, text="zzzz of the", limit=50)
    ]
    assert {"log.product", "log.quotient", "log.power"} <= set(with_features)
    assert kb.candidate_rules(set(), text="of the") == []
