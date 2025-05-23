# NoVacancy Feature Branch History
A ML application that predicts whether someone will cancel their hotel reservation.

## Overview
Includes the installation steps for each feature branch.


## Installation & Running the App

### `postgres`
Implements medallion architecture and reads `/data/bookings_raw.csv` into the `raw_data` table in the Bronze database.

1. Clone the repo.
   ```
    git clone https://github.com/Morgan-Sell/no-vacancy.git
   ```

2. Switch to `postgres` feature branch.
   ```
   git checkout postgres
   ```

3. Create a `.env` file in the project root directory with the following variables:
   - DB_USER
   - DB_PASSWORD
   - DB_PORT
   - BRONZE_DB_HOST
   - BRONZE_DB
   - SILVER_DB_HOST
   - SILVER_DB
   - GOLD_DB_HOST
   - GOLD_DB
   - TEST_DB_USER
   - TEST_DB_PASSWORD
   - TEST_DB_HOST
   - TEST_DB
   - TEST_DB_PORT

4. Build Docker image defined in `docker-compose.yaml`.
   ```
   docker compose build
   ```

5. Start all docker services in detached mode.
   ```
   docker compose up -d
   ```

6. Enter `http://127.0.0.1:8000/` into your web browser. Your web browser should generate the below text.
   ```
   {"message":"Welcome to the No Vacancy API!"}
   ```

7. Identify the container ID by running
   ```
   docker ps -a
   ```

8. Once you've identified the container ID associated with the image called `no-vacancy-app` enter the following to access the application:
   ```
   docker exec -it <container_id> /bin/bash
   ```

9. Now that you're in the application you can query the data from `raw_data` table. Enter the code below in your command line to access PostgreSQL and see the first 10 lines of the `raw_data` table.
    ```
    docker exec -it bronze-db psql -U <db_user_from_dotenv> -d <bronze_db_from_dotenv>

    SELECT * FROM raw_data LIMIT 10;
    ```
    
10. Since the `raw_data` table has been populated, you can process the data and save the data to `novacancy-silver` database and train the model using the `NoVacancyPipeline` class. Once you're inside the Docker container (by following instruction #8), execute the following code:
    ```
    python services/trainer.py
    ```
   
   If the application successfully runs, you should see some like the following:
   ```
   2025-05-14 00:43:16,688 - __main__ - INFO -train_pipeline:116 - ✅ Loaded raw data
   2025-05-14 00:43:19,010 - __main__ - INFO -train_pipeline:126 - ✅ Saved preprocessed data to Silver database
   Fitting 3 folds for each of 20 candidates, totalling 60 fits
   2025-05-14 00:43:34,575 - services.pipeline_management - INFO -save_pipeline:39 - ✅ Pipeline and processor successfully saved at models/no_vacancy_pipeline.pkl
   2025-05-14 00:43:34,740 - __main__ - INFO -evaluate_model:98 - 0.0.3 - AUC: 1.0
   ```


### `web-server`
FastAPI web server comprised of the routers and the services required to process the data, perform feature engineering, train the model, save model artifacts and produce predictions.

1. Clone the repo.
   ```
    git clone https://github.com/Morgan-Sell/no-vacancy.git
   ```

2. Switch to `web-server` feature branch.
   ```
   git checkout web-server
   ```

3. Build the Docker image (replace <docker_username> with your Docker Hub username). Make sure Docker is running on your local PC.
   ```
   docker build -t <docker_username>/no-vacancy:v1 . 
   ```

4. Run the container.
   ```
   docker run -it -p 8000:8000 <docker_username>/no-vacancy:v1
   ```

5. Test the API by going to `http://0.0.0.0:8000/docs` in your browser.




