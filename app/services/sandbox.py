if __name__ == "__main__":


    train_data_enc = set(['number_of_adults', 'number_of_children', 'number_of_weekend_nights',
        'number_of_week_nights', 'car_parking_space', 'lead_time',
        'is_repeat_guest', 'p_c', 'p_not_c', 'average_price',
        'special_requests', 'type_of_meal_Meal Plan 1',
        'type_of_meal_Not Selected', 'type_of_meal_Meal Plan 2',
        'type_of_meal_Meal Plan 3', 'room_type_Room_Type 1',
        'room_type_Room_Type 4', 'room_type_Room_Type 5',
        'room_type_Room_Type 2', 'room_type_Room_Type 6',
        'room_type_Room_Type 7', 'room_type_Room_Type 3',
        'market_segment_type_Offline', 'market_segment_type_Online',
        'market_segment_type_Complementary', 'market_segment_type_Corporate',
        'market_segment_type_Aviation', 'month_of_reservation_Apr',
        'month_of_reservation_Dec', 'month_of_reservation_Oct',
        'month_of_reservation_Aug', 'month_of_reservation_Sep',
        'month_of_reservation_Jun', 'month_of_reservation_Nov',
        'month_of_reservation_Feb', 'month_of_reservation_Mar',
        'month_of_reservation_May', 'month_of_reservation_Jul',
        'month_of_reservation_Jan', 'day_of_week_Sunday',
        'day_of_week_Saturday', 'day_of_week_Friday', 'day_of_week_Monday',
        'day_of_week_Thursday', 'day_of_week_Wednesday', 'day_of_week_Tuesday'])



    test_data_enc = set(['number_of_adults', 'number_of_children', 'number_of_weekend_nights',
        'number_of_week_nights', 'car_parking_space', 'lead_time',
        'is_repeat_guest', 'p_c', 'p_not_c', 'average_price',
        'special_requests', 'type_of_meal_Meal Plan 1',
        'type_of_meal_Not Selected', 'type_of_meal_Meal Plan 2',
        'type_of_meal_Meal Plan 3', 'room_type_Room_Type 1',
        'room_type_Room_Type 4', 'room_type_Room_Type 5',
        'room_type_Room_Type 2', 'room_type_Room_Type 6',
        'room_type_Room_Type 7', 'room_type_Room_Type 3',
        'market_segment_type_Offline', 'market_segment_type_Online',
        'market_segment_type_Complementary', 'market_segment_type_Corporate',
        'market_segment_type_Aviation', 'month_of_reservation_Apr',
        'month_of_reservation_Dec', 'month_of_reservation_Oct',
        'month_of_reservation_Aug', 'month_of_reservation_Sep',
        'month_of_reservation_Jun', 'month_of_reservation_Nov',
        'month_of_reservation_Feb', 'month_of_reservation_Mar',
        'month_of_reservation_May', 'month_of_reservation_Jul',
        'month_of_reservation_Jan', 'day_of_week_Sunday',
        'day_of_week_Saturday', 'day_of_week_Friday', 'day_of_week_Monday',
        'day_of_week_Thursday', 'day_of_week_Wednesday', 'day_of_week_Tuesday'])
    
    print("Elements in either train_data_enc or test_data_enc but not both:", 
          train_data_enc ^ test_data_enc)