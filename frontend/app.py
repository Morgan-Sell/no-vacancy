"""
NoVacancy Frontend Application.

A thin Flask layer that serves the UI and proxies requests
to FastAPI (predictions) and Airflow (training).
"""

import uuid

import requests
from config import Config
from flask import Flask, jsonify, render_template, request


def create_app() -> Flask:
    """Application factory patterns."""
    app = Flask(__name__)
    app.config.from_object(Config)

    register_routes(app)

    return app


def register_routes(app: Flask) -> None:
    """Register all application routes."""

    @app.route("/")
    def index():
        """Render the main booking form."""
        return render_template("index.html")

    @app.route("/health")
    def health():
        """Health check endpoint for container orchestration."""
        return jsonify({"status": "healthy", "service": "novacancy-frontend"})

    # -------------------------------------------------------------------------
    # Prediction API Proxy
    # -------------------------------------------------------------------------
    @app.route("/api/predict", methods=["POST"])
    def proxy_predict():
        """Forward prediction request to FastAPI"""
        try:
            data = request.get_json()

            # Generate Booking_ID for each record (backend is system of record)
            for record in data.get("data", []):
                if "Booking_ID" not in record:
                    record["Booking_ID"] = f"WEB{uuid.uuid4().hex[:8].upper()}"
                # Ensure booking status is set for the model
                record["booking status"] = "Not_Canceled"

            response = requests.post(
                f"{app.config['FASTAPI_URL']}/predict/",
                json=data,
                timeout=app.config["PREDICT_TIMEOUT"],
            )
            response.raise_for_status()
            return jsonify(response.json())

        except requests.exceptions.ConnectionError:
            return jsonify({"error": "Prediction service unavailable"}), 503
        except requests.exceptions.Timeout:
            return jsonify({"error": "Prediction request timed out"}), 504
        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    @app.route("/api/train/status/<path:dag_run_id>")
    def get_trainig_status(dag_run_id: str):
        """Get DAG run status for progress tracking"""
        try:
            dag_id = app.config["TRAINING_DAG_ID"]
            auth = (
                app.config["AIRFLOW_USERNAME"],
                app.config["AIRFLOW_PASSWORD"],
            )

            response = requests.get(
                f"{app.config['AIRFLOW_URL']}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}",
                auth=auth,
                timeout=app.config["AIRFLOW_TIMEOUT"],
            )
            response.raise_for_status()
            result = response.json()

            return jsonify(
                {
                    "dag_run_id": dag_run_id,
                    "state": result.get("state"),
                    "start_date": result.get("start_date"),
                    "end_date": result.get("end_date"),
                }
            )

        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Failed to get status: {str(e)}"}), 500

    @app.route("/api/train/tasks/<path:dag_run_id>")
    def get_task_status(dag_run_id: str):
        """Get individual task states for granular progress."""
        try:
            dag_id = app.config["TRAINING_DAG_ID"]
            auth = (
                app.config["AIRFLOW_USERNAME"],
                app.config["AIRFLOW_PASSWORD"],
            )

            response = requests.get(
                f"{app.config['AIRFLOW_URL']}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances",
                auth=auth,
                timeout=app.config["AIRFLOW_TIMEOUT"],
            )
            response.raise_for_status()
            tasks = response.json().get("task_instances", [])

            return jsonify(
                {
                    "dag_run_id": dag_run_id,
                    "tasks": [
                        {
                            "task_id": t.get("task_id"),
                            "state": t.get("state"),
                            "start_date": t.get("start_date"),
                            "end_date": t.get("end_date"),
                        }
                        for t in tasks
                    ],
                }
            )

        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Failed to get tasks: {str(e)}"}), 500

    # -------------------------------------------------------------------------
    # Training API Proxy (Airflow)
    # -------------------------------------------------------------------------
    @app.route("/api/train", methods=["POST"])
    def trigger_training():
        """Trigger Airflow DAG for model training."""
        try:
            dag_id = app.config["TRAINING_DAG_ID"]
            auth = (
                app.config["AIRFLOW_USERNAME"],
                app.config["AIRFLOW_PASSWORD"],
            )

            response = requests.post(
                f"{app.config['AIRFLOW_URL']}/api/v1/dags/{dag_id}/dagRuns",
                auth=auth,
                json={"conf": {}},
                timeout=app.config["AIRFLOW_TIMEOUT"],
            )
            response.raise_for_status()
            result = response.json()

            return jsonify(
                {
                    "status": "triggered",
                    "dag_run_id": result.get("dag_run_id"),
                    "state": result.get("state"),
                }
            )

        except requests.exceptions.ConnectionError:
            return jsonify({"error": "Training service unavailable"}), 503
        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Failed to trigger training: {str(e)}"}), 500


# Create an app instance for gunicorn
# Use gunicorn b/c Flask app is solely a synchronous proxy layer.
app = create_app()


if __name__ == "__main__":
    # Local development only
    app.run(host="0.0.0.0", port=5050, debug=True)
