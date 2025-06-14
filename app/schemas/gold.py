from config import __model_version__
from sqlalchemy import Column, Date, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    Base class for SQLAlchemy models.
    This approach makes Base a class, which can be inherited by other models.
    """

    pass


class Predictions(Base):
    __tablename__ = "predictions"

    booking_id = Column(String, primary_key=True, nullable=False)
    prediction = Column(Integer, nullable=False)
    probability_not_canceled = Column(Float, nullable=False)
    probability_canceled = Column(Float, nullable=False)
    model_version = Column(
        String, nullable=False, default=__model_version__
    )  # Store the model version used for prediction
    created_at = Column(Date, nullable=False)

    def __repr__(self):
        return (
            f"<Results("
            f"booking_id={self.booking_id}, "
            f"prediction={self.prediction}, "
            f"probability_canceled={self.probability_canceled}, "
            f"created_at={self.created_at})>"
        )
