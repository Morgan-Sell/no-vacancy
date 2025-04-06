from sqlalchemy import Column, Integer, String, Float, Date
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class TrainResults(Base):
    __tablename__ = "train_results"

    booking_id = Column(String, primary_key=True, nullable=False)
    prediction = Column(Integer, nullable=False)
    probability_not_canceled = Column(Float, nullable=False)
    probability_canceled = Column(Float, nullable=False)  
    model_version = Column(String, nullable=False, default=__model_version__)  # Store the model version used for prediction
    created_at = Column(Date, nullable=False)

    def __repr__(self):
        return (
            f"<Results("
            f"booking_id={self.booking_id}, "
            f"prediction={self.prediction}, "
            f"probability_canceled={self.probability_canceled}, "
            f"created_at={self.created_at})>"
        )


class ValidationResult(Base):
    __tablename__ = "validation_results"

    booking_id = Column(String, primary_key=True, nullable=False)
    prediction = Column(Integer, nullable=False)
    probability_not_canceled = Column(Float, nullable=False)
    probability_canceled = Column(Float, nullable=False)  
    model_version = Column(String, nullable=False, default=__model_version__)  # Store the model version used for prediction
    created_at = Column(Date, nullable=False)

    def __repr__(self):
        return (
            f"<Results("
            f"booking_id={self.booking_id}, "
            f"prediction={self.prediction}, "
            f"probability_canceled={self.probability_canceled}, "
            f"created_at={self.created_at})>"
        )


class TestResults(Base):
    __tablename__ = "test_results"
    
    booking_id = Column(String, primary_key=True, nullable=False)
    prediction = Column(Integer, nullable=False)
    probability_not_canceled = Column(Float, nullable=False)
    probability_canceled = Column(Float, nullable=False)  
    model_version = Column(String, nullable=False, default=__model_version__)  # Store the model version used for prediction
    created_at = Column(Date, nullable=False)

    def __repr__(self):
        return (
            f"<Results("
            f"booking_id={self.booking_id}, "
            f"prediction={self.prediction}, "
            f"probability_canceled={self.probability_canceled}, "
            f"created_at={self.created_at})>"
        )