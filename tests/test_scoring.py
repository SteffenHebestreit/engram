import math

from app.scoring import (
    median_proximity_scores,
    recency_blend,
    recency_decay,
    sparse_dot,
    sparse_scores,
)


def test_recency_decay_halves_at_one_half_life():
    assert recency_decay(0.0, 100.0) == 1.0  # brand new
    assert math.isclose(recency_decay(100.0, 100.0), 0.5)  # one half-life
    assert math.isclose(recency_decay(200.0, 100.0), 0.25)  # two half-lives
    assert recency_decay(1e9, 100.0) < 1e-6  # ancient -> ~0


def test_recency_decay_disabled_returns_one():
    assert recency_decay(500.0, 0.0) == 1.0  # half_life<=0 disables decay


def test_recency_blend_weight_zero_is_pure_relevance():
    # weight 0 -> min-max normalized relevance only (recency ignored)
    out = recency_blend([3.0, 1.0, 2.0], [0.0, 1.0, 1.0], 0.0)
    assert out[0] > out[2] > out[1]  # 3 > 2 > 1 preserved


def test_recency_blend_weight_one_is_pure_recency():
    out = recency_blend([3.0, 1.0, 2.0], [0.1, 0.9, 0.5], 1.0)
    assert out == [0.1, 0.9, 0.5]  # relevance ignored, recency passed through


def test_recency_blend_tilts_toward_recent():
    # item 0 is most relevant but old; item 1 is nearly as relevant and brand new.
    # relevance min-max normalizes to [1.0, 0.95, 0.0]; with a 0.5 blend the newer
    # near-top item (0.5·0.95 + 0.5·1.0 = 0.975) overtakes the stale top
    # (0.5·1.0 + 0.5·0.0 = 0.5).
    out = recency_blend([3.0, 2.9, 1.0], [0.0, 1.0, 0.0], 0.5)
    assert out[1] > out[0] > out[2]


def test_empty():
    assert median_proximity_scores([]) == []


def test_single_result_scores_one():
    assert median_proximity_scores([[0.3, 0.4]]) == [1.0]


def test_outlier_scores_lower_than_cohesive_results():
    scores = median_proximity_scores(
        [
            [1.0, 0.0, 0.0],
            [0.95, 0.05, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0],  # outlier
        ]
    )
    cohesive, outlier = scores[:3], scores[3]
    assert all(c > 0.9 for c in cohesive)
    assert outlier < min(cohesive)


def test_scores_bounded_zero_one():
    scores = median_proximity_scores([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_identical_vectors_all_score_one():
    scores = median_proximity_scores([[0.5, 0.5]] * 4)
    assert all(math.isclose(s, 1.0) for s in scores)


def test_zero_median_degrades_gracefully():
    # opposite vectors -> median is the zero vector
    scores = median_proximity_scores([[1.0, 0.0], [-1.0, 0.0]])
    assert scores == [0.5, 0.5]


def test_sparse_dot_sums_shared_terms_only():
    a = {"cat": 1.0, "dog": 2.0, "fish": 0.5}
    b = {"dog": 3.0, "fish": 4.0, "bird": 9.0}
    # dog: 2*3=6, fish: 0.5*4=2 ; cat/bird not shared
    assert sparse_dot(a, b) == 8.0


def test_sparse_dot_disjoint_or_empty_is_zero():
    assert sparse_dot({"a": 1.0}, {"b": 1.0}) == 0.0
    assert sparse_dot({}, {"a": 1.0}) == 0.0
    assert sparse_dot({"a": 1.0}, {}) == 0.0


def test_sparse_dot_is_symmetric_regardless_of_map_size():
    big = {str(i): 1.0 for i in range(100)}
    big["x"] = 2.0
    small = {"x": 3.0}
    assert sparse_dot(big, small) == sparse_dot(small, big) == 6.0


def test_sparse_scores_normalize_to_top_candidate():
    q = {"cat": 1.0, "dog": 1.0}
    cands = [
        {"cat": 2.0, "dog": 2.0},  # dot 4 -> top -> 1.0
        {"cat": 1.0},  # dot 1 -> 0.25
        {"bird": 5.0},  # dot 0 -> 0.0
        None,  # missing sparse -> 0.0
    ]
    assert sparse_scores(q, cands) == [1.0, 0.25, 0.0, 0.0]


def test_sparse_scores_without_query_is_all_zero():
    assert sparse_scores(None, [{"a": 1.0}, {"b": 2.0}]) == [0.0, 0.0]
    assert sparse_scores({}, [{"a": 1.0}]) == [0.0]


def test_sparse_scores_all_zero_when_no_overlap():
    assert sparse_scores({"q": 1.0}, [{"a": 1.0}, None]) == [0.0, 0.0]
