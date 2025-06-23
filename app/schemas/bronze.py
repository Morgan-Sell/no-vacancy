from sqlalchemy import Column, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    Base class for SQLAlchemy models.
    This approach makes Base a class, which can be inherited by other models.
    """

    pass


class RawData(Base):
    """
    Bronze table for raw data storage.
    """

    __tablename__ = "raw_data"

    booking_id = Column(String, primary_key=True)
    number_of_adults = Column(Integer, nullable=True)
    number_of_children = Column(Integer, nullable=True)
    number_of_weekend_nights = Column(Integer, nullable=True)
    number_of_week_nights = Column(Integer, nullable=True)
    type_of_meal = Column(String, nullable=True)
    car_parking_space = Column(Integer, nullable=True)
    room_type = Column(String, nullable=True)
    lead_time = Column(Integer, nullable=True)  # in days
    market_segment_type = Column(String, nullable=True)
    is_repeat_guest = Column(Integer, nullable=True)  # 1 if repeat guest, 0 otherwise
    num_previous_cancellations = Column(Integer, nullable=True)
    num_previous_bookings_not_canceled = Column(Integer, nullable=True)
    average_price = Column(Float, nullable=True)
    special_requests = Column(Integer, nullable=True)
    date_of_reservation = Column(String, nullable=True)
    booking_status = Column(String, nullable=False)
