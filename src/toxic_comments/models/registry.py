from toxic_comments.models.baseline import build_dummy_baseline
from toxic_comments.models.tfidf_logreg import build_tfidf_logistic_regression


def build_models(max_features: int = 50_000):
    return {
        "dummy_most_frequent": build_dummy_baseline(),
        "tfidf_logistic_regression": build_tfidf_logistic_regression(
            max_features=max_features
        )
    }