import pytest


def test_pipeline_fit_and_predict(sample_pipeline, preprocessed_booking_data):
    # Arrange
    X, y = preprocessed_booking_data

    search_space = {"n_estimators": [10], "max_depth": [3]}

    # Act
    sample_pipeline.fit(X, y, search_space)
    preds = sample_pipeline.predict(X)
    probs = sample_pipeline.predict_proba(X)

    # Assert
    assert len(preds) == len(X)
    assert probs.shape == (len(X), 2)


def test_get_logged_params(sample_pipeline, preprocessed_booking_data):
    # Arrange
    X, y = preprocessed_booking_data

    search_space = {"n_estimators": [10], "max_depth": [3]}

    # Act
    sample_pipeline.fit(X, y, search_space)
    params = sample_pipeline.get_logged_params()

    # Assert
    assert params["model"] == "RandomForestClassifier"
    assert "model_param_n_estimators" in params
    assert "best_model_val_score" in params


def test_predict_before_fit_raises(sample_pipeline, preprocessed_booking_data):
    X, _ = preprocessed_booking_data
    with pytest.raises(AttributeError, match="Pipeline is not trained"):
        sample_pipeline.predict(X)


def test_get_logged_params_before_fit_raises(sample_pipeline):
    with pytest.raises(AttributeError, match="Model is not trained"):
        sample_pipeline.get_logged_params()
