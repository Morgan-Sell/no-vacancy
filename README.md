# NoVacancy - WebServer Feature Branch
A ML application that predicts whether someone will cancel their hotel reservation.

## Overview
This feature branch includes:
- Data preprocessing
- Feature engineering
- Model training & hyperparameter tuning
- Pipeline management (save, load, delete)
- Prediction service

The web server is built using FastAPI and is containerized with Docker.

## Installation & Running the App

1. Clone the repo.
   ```
    git clone https://github.com/Morgan-Sell/no-vacancy.git
   ```

2. Switch to `web-server` feature branch.
   ```
   git checkout web-server
   ```

3. Build the Docker image (replace <docker_username> with your Docker Hub username).
   ```
   docker build -t <docker_username>/no-vacancy:v1 . 
   ```

4. Run the container.
   ```
   docker run -it -p 8000:8000 <docker_username>/no-vacancy:v1
   ```

5. Test the API by going to `http://0.0.0.0:8000/docs` in your browser.




