import math

from app.scoring import median_proximity_scores


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
