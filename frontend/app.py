"""
NoVacancy Frontend Application.

A thin Flask layer that serves the UI and proxies requests
to FastAPI (predictions) and Airflow (training).
"""

from config import Config
from flask import Flask, jsonify, render_template


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


# Create an app instance for gunicorn
# Use gunicorn b/c Flask app is solely a synchronous proxy layer.
app = create_app()


if __name__ == "__main__":
    # Local development only
    app.run(host="0.0.0.0", port=5050, debug=True)
