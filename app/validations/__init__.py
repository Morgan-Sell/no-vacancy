# ------------------------------------------
# Global Variables for the validations module
# ------------------------------------------


# -- Categorical Variables --
EXPECTED_BOOKING_STATUS = (["Canceled", "Not_Canceled"],)
EXPECTED_MEAL_TYPES = (["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"],)
EXPECTED_ROOM_TYPES = [
    "Room_Type 1",
    "Room_Type 2",
    "Room_Type 3",
    "Room_Type 4",
    "Room_Type 5",
    "Room_Type 6",
    "Room_Type 7",
]
EXPECTED_MARKET_SEGMENTS = [
    "Online",
    "Offline",
    "Corporate",
    "Aviation",
    "Complementary",
]


# -- Business Logic --
MIN_AVERAGE_PRICE = 0
MAX_AVERAGE_PRICE = 1000
MIN_LEAD_TIME = 0
MAX_LEAD_TIME = 730  # 2 years
