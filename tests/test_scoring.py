import math

from app.scoring import median_proximity_scores, sparse_dot, sparse_scores


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
