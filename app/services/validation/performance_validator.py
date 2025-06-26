from base import ModelValidator


class PerformanceValidator(ModelValidator):
    """
    Validates model perofrmance metrics, e.g., AUC, F1 score, etc.
    """

    def __init__(self, min_auc: float = 0.85):
        pass
