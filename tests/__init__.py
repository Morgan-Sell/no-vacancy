NECESSARY_BINARY_VARIABLES = {
    # Add only fields defined in TrainData or ValidationTestData
    "type_of_meal": ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3"],
    "room_type": [
        "Room_Type 1",
        "Room_Type 2",
        "Room_Type 3",
        "Room_Type 4",
        "Room_Type 5",
        "Room_Type 6",
        "Room_Type 7",
    ],
    "market_segment_type": ["Online", "Corporate", "Complementary", "Aviation"],
    "month_of_reservation": [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Oct",
        "Nov",
        "Dec",
    ],
    "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
}

BOOKING_DATA_VARS_TO_DROP = [
    "type_of_meal",
    "room_type",
    "market_segment_type",
    "month_of_reservation",
    "day_of_week",
]

BOOKING_DATA_RENAME_MAP = {
    "repeated": "is_repeat_guest",
    "pc": "num_previous_cancellations",
    "pnot_c": "num_previous_bookings_not_canceled",
    "type of meal": "type_of_meal",
    "car parking space": "car_parking_space",
    "room type": "room_type",
    "market segment type": "market_segment_type",
}
