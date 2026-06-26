import math

from app.eval import (
    attribute_channels,
    average_precision,
    bootstrap_ci,
    evaluate,
    ndcg_at_k,
    precision_at_k,
    ranked_docs,
    recall_at_k,
    score_results,
)

RANKED = ["a", "b", "c"]
QREL = {"a": 1, "c": 1}  # a and c relevant, b not


def test_recall_and_precision():
    assert recall_at_k(RANKED, QREL, 2) == 0.5  # {a,b} ∩ {a,c} = {a}
    assert recall_at_k(RANKED, QREL, 3) == 1.0
    assert precision_at_k(RANKED, QREL, 2) == 0.5


def test_ndcg_matches_hand_computation():
    # gains [1,0,1] -> dcg 1 + 0 + 1/log2(4)=0.5 -> 1.5; idcg [1,1] -> 1.6309
    assert math.isclose(ndcg_at_k(RANKED, QREL, 3), 1.5 / 1.6309297535714573, abs_tol=1e-6)


def test_average_precision():
    # hits at ranks 1 and 3 -> (1/1 + 2/3) / 2
    assert math.isclose(average_precision(RANKED, QREL), (1.0 + 2 / 3) / 2, abs_tol=1e-9)


def test_metrics_handle_no_relevant():
    assert recall_at_k(["a"], {}, 5) == 0.0
    assert average_precision(["a"], {}) == 0.0
    assert ndcg_at_k(["a"], {}, 5) == 0.0


def test_bootstrap_ci_is_deterministic_and_brackets_mean():
    a = bootstrap_ci([0.0, 1.0, 0.5, 0.8, 0.2], seed=0)
    b = bootstrap_ci([0.0, 1.0, 0.5, 0.8, 0.2], seed=0)
    assert a == b  # same seed -> reproducible
    mean, lo, hi = a
    assert math.isclose(mean, 0.5, abs_tol=1e-9)
    assert lo <= mean <= hi


def test_bootstrap_ci_edge_cases():
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)
    assert bootstrap_ci([0.7]) == (0.7, 0.7, 0.7)
    mean, lo, hi = bootstrap_ci([0.5] * 8)
    assert (mean, lo, hi) == (0.5, 0.5, 0.5)  # zero variance -> tight interval


def test_ranked_docs_dedups_chunks_keeping_order():
    results = [
        {"document_id": "d1"},
        {"document_id": "d2"},
        {"document_id": "d1"},  # another chunk of d1
        {"document_id": "d3"},
    ]
    assert ranked_docs(results) == ["d1", "d2", "d3"]


def test_score_results_aggregates_with_cis():
    results = {"q1": [{"document_id": "a"}, {"document_id": "b"}, {"document_id": "c"}]}
    qrels = {"q1": {"a": 1, "c": 1}}
    out = score_results(results, qrels, ks=(2, 3))
    assert out["n_queries"] == 1
    assert math.isclose(out["metrics"]["Recall@2"]["mean"], 0.5, abs_tol=1e-9)
    assert math.isclose(out["metrics"]["Recall@3"]["mean"], 1.0, abs_tol=1e-9)
    # single query -> degenerate CI equal to the mean
    assert out["metrics"]["Recall@2"]["ci95"] == [0.5, 0.5]


def test_attribution_credits_unique_channel_recovery():
    results = {
        "q1": [
            {"document_id": "a", "channels": ["content", "fulltext"]},  # relevant
            {"document_id": "b", "channels": ["sparse"]},  # relevant, sparse-only
            {"document_id": "x", "channels": ["content"]},  # not relevant
        ]
    }
    qrels = {"q1": {"a": 1, "b": 1, "c": 1}}  # c relevant but not retrieved
    attr = attribute_channels(results, qrels)
    assert attr["gold_hits_retrieved"] == 2
    assert attr["by_channel"] == {"content": 1, "fulltext": 1, "sparse": 1}
    # b was surfaced ONLY by sparse -> lost without that channel
    assert attr["unique_to_channel"] == {"sparse": 1}


def test_attribution_respects_top_k_cut():
    results = {
        "q1": [
            {"document_id": "a", "channels": ["content"]},
            {"document_id": "b", "channels": ["sparse"]},
        ]
    }
    qrels = {"q1": {"a": 1, "b": 1}}
    # only the first result counts at k=1
    attr = attribute_channels(results, qrels, k=1)
    assert attr["gold_hits_retrieved"] == 1
    assert attr["by_channel"] == {"content": 1}


def test_evaluate_combines_metrics_and_attribution():
    results = {"q1": [{"document_id": "a", "channels": ["content", "sparse"]}]}
    qrels = {"q1": {"a": 1}}
    out = evaluate(results, qrels, ks=(10,))
    assert out["n_queries"] == 1
    assert "nDCG@10" in out["metrics"]
    assert out["attribution"]["gold_hits_retrieved"] == 1


async def test_run_evaluation_calls_search_per_case(monkeypatch):
    from app import eval as eval_mod
    from app import search as search_mod

    canned = {
        "alpha": [{"document_id": "a", "channels": ["content", "sparse"]}],
        "beta": [{"document_id": "x", "channels": ["content"]}],  # misses gold "b"
    }

    async def fake_search(store, http, query, top_k=50, tuning=None):
        return canned[query]

    monkeypatch.setattr(search_mod, "search", fake_search)

    report = await eval_mod.run_evaluation(
        None, None,
        golden={"q1": {"a": 1}, "q2": {"b": 1}},
        queries={"q1": "alpha", "q2": "beta"},
        ks=(10,),
    )
    assert report["n_queries"] == 2
    assert math.isclose(report["metrics"]["Recall@10"]["mean"], 0.5, abs_tol=1e-9)
    # only doc "a" (q1) was a recovered gold hit; q2 missed
    assert report["attribution"]["gold_hits_retrieved"] == 1
    assert report["attribution"]["by_channel"] == {"content": 1, "sparse": 1}
