from sqlalchemy import Column, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    Base class for SQLAlchemy models.
    This approach makes Base a class, which can be inherited by other models.
    """

    pass


class TrainValidationData(Base):
    __tablename__ = "train_validation_data"

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
    is_repeat_guest = Column(Integer, nullable=True)
    num_previous_cancellations = Column(Integer, nullable=True)
    num_previous_bookings_not_canceled = Column(Integer, nullable=True)
    special_requests = Column(Integer, nullable=True)
    average_price = Column(Float, nullable=True)
    month_of_reservation = Column(String, nullable=True)
    day_of_week = Column(String, nullable=True)
    # is_type_of_meal_meal_plan_1 = Column(Integer, nullable=False)
    # is_type_of_meal_meal_plan_2 = Column(Integer, nullable=False)
    # is_type_of_meal_meal_plan_3 = Column(Integer, nullable=False)
    # is_room_type_room_type_1 = Column(Integer, nullable=False)
    # is_room_type_room_type_2 = Column(Integer, nullable=False)
    # is_room_type_room_type_3 = Column(Integer, nullable=False)
    # is_room_type_room_type_4 = Column(Integer, nullable=False)
    # is_room_type_room_type_5 = Column(Integer, nullable=False)
    # is_room_type_room_type_6 = Column(Integer, nullable=False)
    # is_room_type_room_type_7 = Column(Integer, nullable=False)
    # is_market_segment_type_online = Column(Integer, nullable=False)
    # is_market_segment_type_corporate = Column(Integer, nullable=False)
    # is_market_segment_type_complementary = Column(Integer, nullable=False)
    # is_market_segment_type_aviation = Column(Integer, nullable=False)
    # is_month_of_reservation_nov = Column(Integer, nullable=False)
    # is_month_of_reservation_dec = Column(Integer, nullable=False)
    # is_month_of_reservation_jun = Column(Integer, nullable=False)
    # is_month_of_reservation_jan = Column(Integer, nullable=False)
    # is_month_of_reservation_oct = Column(Integer, nullable=False)
    # is_month_of_reservation_may = Column(Integer, nullable=False)
    # is_month_of_reservation_apr = Column(Integer, nullable=False)
    # is_month_of_reservation_aug = Column(Integer, nullable=False)
    # is_month_of_reservation_mar = Column(Integer, nullable=False)
    # is_month_of_reservation_feb = Column(Integer, nullable=False)
    # is_month_of_reservation_jul = Column(Integer, nullable=False)
    # is_day_of_week_monday = Column(Integer, nullable=False)
    # is_day_of_week_friday = Column(Integer, nullable=False)
    # is_day_of_week_wednesday = Column(Integer, nullable=False)
    # is_day_of_week_thursday = Column(Integer, nullable=False)
    # is_day_of_week_saturday = Column(Integer, nullable=False)
    # is_day_of_week_tuesday = Column(Integer, nullable=False)
    is_cancellation = Column(Integer, nullable=False)


class TestData(Base):
    __tablename__ = "test_data"

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
    is_repeat_guest = Column(Integer, nullable=True)
    num_previous_cancellations = Column(Integer, nullable=True)
    num_previous_bookings_not_canceled = Column(Integer, nullable=True)
    special_requests = Column(Integer, nullable=True)
    average_price = Column(Float, nullable=True)
    month_of_reservation = Column(String, nullable=True)
    day_of_week = Column(String, nullable=True)
    # is_type_of_meal_meal_plan_1 = Column(Integer, nullable=False)
    # is_type_of_meal_meal_plan_2 = Column(Integer, nullable=False)
    # is_type_of_meal_meal_plan_3 = Column(Integer, nullable=False)
    # is_room_type_room_type_1 = Column(Integer, nullable=False)
    # is_room_type_room_type_2 = Column(Integer, nullable=False)
    # is_room_type_room_type_3 = Column(Integer, nullable=False)
    # is_room_type_room_type_4 = Column(Integer, nullable=False)
    # is_room_type_room_type_5 = Column(Integer, nullable=False)
    # is_room_type_room_type_6 = Column(Integer, nullable=False)
    # is_room_type_room_type_7 = Column(Integer, nullable=False)
    # is_market_segment_type_online = Column(Integer, nullable=False)
    # is_market_segment_type_corporate = Column(Integer, nullable=False)
    # is_market_segment_type_complementary = Column(Integer, nullable=False)
    # is_market_segment_type_aviation = Column(Integer, nullable=False)
    # is_month_of_reservation_nov = Column(Integer, nullable=False)
    # is_month_of_reservation_dec = Column(Integer, nullable=False)
    # is_month_of_reservation_jun = Column(Integer, nullable=False)
    # is_month_of_reservation_jan = Column(Integer, nullable=False)
    # is_month_of_reservation_oct = Column(Integer, nullable=False)
    # is_month_of_reservation_may = Column(Integer, nullable=False)
    # is_month_of_reservation_apr = Column(Integer, nullable=False)
    # is_month_of_reservation_aug = Column(Integer, nullable=False)
    # is_month_of_reservation_mar = Column(Integer, nullable=False)
    # is_month_of_reservation_feb = Column(Integer, nullable=False)
    # is_month_of_reservation_jul = Column(Integer, nullable=False)
    # is_day_of_week_monday = Column(Integer, nullable=False)
    # is_day_of_week_friday = Column(Integer, nullable=False)
    # is_day_of_week_wednesday = Column(Integer, nullable=False)
    # is_day_of_week_thursday = Column(Integer, nullable=False)
    # is_day_of_week_saturday = Column(Integer, nullable=False)
    # is_day_of_week_tuesday = Column(Integer, nullable=False)
    is_cancellation = Column(Integer, nullable=False)
