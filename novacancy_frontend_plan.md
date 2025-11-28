# NoVacancy Frontend Implementation Plan

> *"We see your departure before you do"*

## Executive Summary

This plan outlines the implementation of a Flask-based frontend for the NoVacancy MLOps application. The frontend provides an elegant, noir-inspired interface for hotel staff to predict booking cancellations using the existing FastAPI inference backend and Airflow-orchestrated training pipeline.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            NoVacancy System                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐       ┌─────────────────┐       ┌─────────────────┐      │
│   │   Flask     │ JSON  │    FastAPI      │       │    MLflow       │      │
│   │   Frontend  │──────▶│    Inference    │──────▶│    Registry     │      │
│   │   :5050     │       │    :8000        │       │    :5001        │      │
│   └─────────────┘       └─────────────────┘       └─────────────────┘      │
│         │                                                                   │
│         │ REST API (trigger + poll status)                                  │
│         ▼                                                                   │
│   ┌─────────────┐       ┌─────────────────┐       ┌─────────────────┐      │
│   │   Airflow   │       │    Training     │       │   PostgreSQL    │      │
│   │   Webserver │──────▶│    Container    │──────▶│   Bronze/Silver │      │
│   │   :8080     │       │                 │       │   /Gold DBs     │      │
│   └─────────────┘       └─────────────────┘       └─────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Prediction Flow**: User submits form → Flask collects JSON → POST to FastAPI `/predict/` → Model loaded from MLflow → Probability returned → Displayed in UI

2. **Training Flow**: User clicks "Train Model" → Flask triggers Airflow DAG (`training_pipeline`) via REST API → Frontend polls DAG run status → Progress displayed in real-time → Completion notification shown

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Airflow DAG** | `training_pipeline` | Matches existing DAG in `app/dags/training_pipeline_dag.py` |
| **Authentication** | None | Internal-only application |
| **Booking ID** | Backend-generated | System of record owns ID assignment; prevents collisions; maintains referential integrity |
| **Training Feedback** | Real-time polling | Better UX; users see task progress without refreshing |

---

## Branch Strategy

### Branch 1: `feature/flask-foundation`

**Purpose**: Establish the Flask application skeleton with noir design system.

**Files to Create**:

| File | Description |
|------|-------------|
| `frontend/__init__.py` | Package marker |
| `frontend/app.py` | Flask application factory and routes |
| `frontend/config.py` | Configuration management (API URLs, Airflow credentials) |
| `frontend/templates/base.html` | Base template with noir CSS design system |
| `frontend/templates/index.html` | Main page (extends base) |
| `frontend/static/css/style.css` | Custom styles and animations |
| `Dockerfile.frontend` | Container definition |
| `requirements/frontend.txt` | Flask dependencies |

**docker-compose.yml Addition**:
```yaml
frontend:
  build:
    context: .
    dockerfile: Dockerfile.frontend
  container_name: novacancy-frontend
  ports:
    - "5050:5050"
  environment:
    FASTAPI_URL: http://inference-container:8000
    AIRFLOW_URL: http://airflow-webserver:8080
    AIRFLOW_USERNAME: homer
    AIRFLOW_PASSWORD: waffles
  depends_on:
    - inference-container
  profiles:
    - airflow
  restart: unless-stopped
```

**Design System (CSS Variables)**:
```css
:root {
  /* Noir palette inspired by White Lotus */
  --bg-primary: #0d0d0d;
  --bg-secondary: #1a1a1a;
  --bg-card: #262626;
  --bg-input: #333333;

  /* Text colors */
  --text-primary: #f5f5dc;      /* Cream */
  --text-secondary: #a0a0a0;
  --text-muted: #666666;

  /* Accent colors */
  --accent-gold: #c9a227;
  --accent-gold-hover: #d4af37;
  --accent-danger: #8b0000;
  --accent-success: #2e5a2e;

  /* Typography */
  --font-display: 'Playfair Display', Georgia, serif;
  --font-body: 'Inter', -apple-system, sans-serif;

  /* Spacing */
  --spacing-xs: 0.25rem;
  --spacing-sm: 0.5rem;
  --spacing-md: 1rem;
  --spacing-lg: 2rem;
  --spacing-xl: 4rem;

  /* Borders */
  --border-radius: 4px;
  --border-color: #404040;
}
```

**Acceptance Criteria**:
- [ ] Flask app runs on port 5050
- [ ] Base template renders with noir styling
- [ ] Health check endpoint responds at `/health`
- [ ] Container builds successfully
- [ ] Service integrates with docker-compose (airflow profile)

---

### Branch 2: `feature/booking-form`

**Purpose**: Build the reservation form matching the expected data schema.

**Form Field Mapping**:

| CSV Column | Internal Name | Form Label | Input Type | Validation |
|------------|---------------|------------|------------|------------|
| `Booking_ID` | `Booking_ID` | *(backend-generated)* | — | Not in form |
| `number of adults` | `number of adults` | Number of Adults | number | min=1, max=10 |
| `number of children` | `number of children` | Number of Children | number | min=0, max=10 |
| `number of weekend nights` | `number of weekend nights` | Weekend Nights | number | min=0, max=14 |
| `number of week nights` | `number of week nights` | Weekday Nights | number | min=0, max=30 |
| `type of meal` | `type of meal` | Meal Plan | select | options below |
| `car parking space` | `car parking space` | Parking Space | toggle | 0 or 1 |
| `room type` | `room type` | Room Type | select | options below |
| `lead time` | `lead time` | Days Until Arrival | number | min=0, max=365 |
| `market segment type` | `market segment type` | Booking Channel | select | options below |
| `repeated` | `repeated` | Returning Guest | toggle | 0 or 1 |
| `P-C` | `P-C` | Previous Cancellations | number | min=0 |
| `P-not-C` | `P-not-C` | Previous Completed Stays | number | min=0 |
| `average price` | `average price` | Nightly Rate ($) | number | min=0, step=0.01 |
| `special requests` | `special requests` | Special Requests | number | min=0, max=5 |
| `date of reservation` | `date of reservation` | Reservation Date | date | default=today |
| `booking status` | `booking status` | *(not in form)* | — | Set to "Not_Canceled" |

**Note on Field Names**: The JSON payload uses the original CSV column names (with spaces) to match the existing FastAPI `/predict/` endpoint expectations. The backend's `predictor.py` handles the transformation.

**Select Options**:

```python
MEAL_PLANS = [
    ("Meal Plan 1", "Meal Plan 1"),
    ("Meal Plan 2", "Meal Plan 2"),
    ("Meal Plan 3", "Meal Plan 3"),
    ("Not Selected", "No Meal Plan"),
]

ROOM_TYPES = [
    ("Room_Type 1", "Standard Room"),
    ("Room_Type 2", "Deluxe Room"),
    ("Room_Type 3", "Suite"),
    ("Room_Type 4", "Premium Suite"),
    ("Room_Type 5", "Penthouse"),
    ("Room_Type 6", "Villa"),
    ("Room_Type 7", "Presidential"),
]

MARKET_SEGMENTS = [
    ("Online", "Online"),
    ("Offline", "Offline"),
    ("Corporate", "Corporate"),
    ("Complementary", "Complementary"),
    ("Aviation", "Aviation"),
]
```

**Form Layout** (Two-column responsive grid):

```
┌─────────────────────────────────────────────────────────────┐
│                      N O V A C A N C Y                      │
│              "We see your departure before you do"          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─── Guest Information ───┐  ┌─── Stay Details ─────────┐ │
│  │ Number of Adults    [2] │  │ Weekend Nights       [1] │ │
│  │ Number of Children  [0] │  │ Weekday Nights       [3] │ │
│  │ Returning Guest    [No] │  │ Days Until Arrival  [45] │ │
│  │ Previous Cancellations  │  │ Reservation Date  [date] │ │
│  │                     [0] │  │                          │ │
│  │ Previous Stays      [2] │  │                          │ │
│  └─────────────────────────┘  └──────────────────────────┘ │
│                                                             │
│  ┌─── Room & Services ─────┐  ┌─── Booking Info ─────────┐ │
│  │ Room Type    [Deluxe ▼] │  │ Booking Channel          │ │
│  │ Meal Plan  [Plan 1  ▼]  │  │              [Online ▼]  │ │
│  │ Parking         [toggle]│  │ Nightly Rate    [$150]   │ │
│  │ Special Requests    [1] │  │                          │ │
│  └─────────────────────────┘  └──────────────────────────┘ │
│                                                             │
│        [ Train Model ]              [ Submit Booking ]      │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    CANCELLATION RISK                        │
│                                                             │
│              ┌─────────────────────────┐                    │
│              │                         │                    │
│              │          73%            │                    │
│              │     HIGH RISK           │                    │
│              │                         │                    │
│              └─────────────────────────┘                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Acceptance Criteria**:
- [ ] All form fields render with appropriate input types
- [ ] Client-side validation matches constraints
- [ ] Form is responsive (mobile-friendly)
- [ ] Toggle switches work for boolean fields
- [ ] Select dropdowns populated with correct options
- [ ] Booking ID is NOT included in form (backend-generated)

---

### Branch 3: `feature/api-integration`

**Purpose**: Connect the frontend to FastAPI and Airflow backends with real-time training progress.

**JavaScript Functions**:

```javascript
// 1. Submit booking for prediction
async function submitBooking(formData) {
    const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: [formData] })
    });
    return response.json();
}

// 2. Trigger model training via Airflow
async function triggerTraining() {
    const response = await fetch('/api/train', {
        method: 'POST'
    });
    return response.json();  // Returns { dag_run_id: "..." }
}

// 3. Poll training status
async function getTrainingStatus(dagRunId) {
    const response = await fetch(`/api/train/status/${dagRunId}`);
    return response.json();
}

// 4. Poll task instances for granular progress
async function getTaskStatus(dagRunId) {
    const response = await fetch(`/api/train/tasks/${dagRunId}`);
    return response.json();
}
```

**Flask Proxy Routes**:

```python
@app.route('/api/predict', methods=['POST'])
def proxy_predict():
    """Forward prediction request to FastAPI."""
    # Generate Booking_ID on backend (best practice)
    data = request.json
    for record in data.get('data', []):
        if 'Booking_ID' not in record:
            record['Booking_ID'] = f"WEB{uuid.uuid4().hex[:8].upper()}"
        # Ensure booking_status is set
        record['booking status'] = 'Not_Canceled'

    response = requests.post(
        f"{FASTAPI_URL}/predict/",
        json=data,
        timeout=30
    )
    return jsonify(response.json())


@app.route('/api/train', methods=['POST'])
def trigger_training():
    """Trigger Airflow DAG for model training."""
    response = requests.post(
        f"{AIRFLOW_URL}/api/v1/dags/training_pipeline/dagRuns",
        auth=(AIRFLOW_USER, AIRFLOW_PASSWORD),
        json={"conf": {}},
        timeout=10
    )
    result = response.json()
    return jsonify({
        "status": "triggered",
        "dag_run_id": result.get("dag_run_id"),
        "state": result.get("state")
    })


@app.route('/api/train/status/<dag_run_id>')
def get_training_status(dag_run_id):
    """Get DAG run status for progress tracking."""
    response = requests.get(
        f"{AIRFLOW_URL}/api/v1/dags/training_pipeline/dagRuns/{dag_run_id}",
        auth=(AIRFLOW_USER, AIRFLOW_PASSWORD),
        timeout=10
    )
    result = response.json()
    return jsonify({
        "dag_run_id": dag_run_id,
        "state": result.get("state"),  # running, success, failed
        "start_date": result.get("start_date"),
        "end_date": result.get("end_date")
    })


@app.route('/api/train/tasks/<dag_run_id>')
def get_task_status(dag_run_id):
    """Get individual task states for granular progress."""
    response = requests.get(
        f"{AIRFLOW_URL}/api/v1/dags/training_pipeline/dagRuns/{dag_run_id}/taskInstances",
        auth=(AIRFLOW_USER, AIRFLOW_PASSWORD),
        timeout=10
    )
    tasks = response.json().get("task_instances", [])
    return jsonify({
        "dag_run_id": dag_run_id,
        "tasks": [
            {
                "task_id": t.get("task_id"),
                "state": t.get("state"),
                "start_date": t.get("start_date"),
                "end_date": t.get("end_date")
            }
            for t in tasks
        ]
    })
```

**Training Progress UI Component**:

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PROGRESS                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ● import_csv_data ─────────────────────────── ✓ Complete  │
│  ● validate_bronze_data ────────────────────── ✓ Complete  │
│  ● train_model ─────────────────────────────── ◐ Running   │
│  ○ validate_model ──────────────────────────── ○ Pending   │
│  ○ register_model ──────────────────────────── ○ Pending   │
│                                                             │
│  Status: Training in progress...                            │
│  Started: 2025-11-28 14:32:05                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Polling Logic**:

```javascript
async function monitorTraining(dagRunId) {
    const POLL_INTERVAL = 3000;  // 3 seconds

    const poll = async () => {
        const status = await getTrainingStatus(dagRunId);
        const tasks = await getTaskStatus(dagRunId);

        updateProgressUI(status, tasks);

        if (status.state === 'running' || status.state === 'queued') {
            setTimeout(poll, POLL_INTERVAL);
        } else if (status.state === 'success') {
            showSuccess("Model training complete!");
        } else if (status.state === 'failed') {
            showError("Training failed. Check Airflow logs.");
        }
    };

    poll();
}
```

**Result Display Logic**:

```javascript
function displayResult(probability) {
    const percentage = (probability * 100).toFixed(1);
    const riskLevel = probability > 0.7 ? 'HIGH' :
                      probability > 0.4 ? 'MODERATE' : 'LOW';
    const riskClass = probability > 0.7 ? 'risk-high' :
                      probability > 0.4 ? 'risk-moderate' : 'risk-low';

    resultElement.innerHTML = `
        <div class="result-card ${riskClass}">
            <span class="percentage">${percentage}%</span>
            <span class="risk-label">${riskLevel} RISK</span>
        </div>
    `;
}
```

**Error Handling**:

| Scenario | User Message |
|----------|--------------|
| FastAPI unreachable | "Prediction service unavailable. Please try again." |
| Airflow unreachable | "Training service unavailable. Contact administrator." |
| Invalid form data | "Please check your inputs and try again." |
| Model not found | "No model available. Please train a model first." |
| Training failed | "Training failed. Check Airflow logs for details." |

**Acceptance Criteria**:
- [ ] Submit button sends form data to `/api/predict`
- [ ] Backend generates `Booking_ID` (format: `WEB` + 8 hex chars)
- [ ] Prediction result displays with risk classification
- [ ] Train button triggers Airflow DAG
- [ ] Training progress panel shows real-time task status
- [ ] Polling stops on completion/failure
- [ ] Success/failure notifications displayed
- [ ] Error states handled gracefully
- [ ] Loading states shown during API calls

---

## File Structure (Final State)

```
no-vacancy/
├── frontend/
│   ├── __init__.py
│   ├── app.py                 # Flask application with proxy routes
│   ├── config.py              # Configuration (URLs, credentials)
│   ├── templates/
│   │   ├── base.html          # Base template with design system
│   │   └── index.html         # Main page with form + progress panel
│   └── static/
│       └── css/
│           └── style.css      # Noir styling + progress animations
├── requirements/
│   └── frontend.txt           # Flask dependencies (new)
├── Dockerfile.frontend        # Frontend container (new)
└── docker-compose.yml         # Updated with frontend service
```

---

## Dependencies

**requirements/frontend.txt**:
```
Flask==3.0.0
requests==2.32.3
python-dotenv==1.0.1
gunicorn==21.2.0
```

---

## Dockerfile.frontend

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements/frontend.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY frontend/ .

# Expose port
EXPOSE 5050

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5050", "--workers", "2", "app:app"]
```

---

## Airflow API Configuration

The Airflow REST API requires authentication. The existing setup uses Basic Auth:

**Credentials** (from docker-compose.yml airflow-init):
- Username: `homer`
- Password: `waffles`

**Required Airflow Config** (should already be set in Dockerfile.airflow):
```ini
[api]
auth_backends = airflow.api.auth.backend.basic_auth
```

**DAG Details** (from `app/dags/training_pipeline_dag.py`):
- DAG ID: `training_pipeline`
- Schedule: `@weekly`
- Tasks: `import_csv_data` → `validate_bronze_data` → `train_model` → `validate_model` → `register_model`

---

## Testing Strategy

| Test Type | Scope | Tool |
|-----------|-------|------|
| Unit | Form validation, config loading, ID generation | pytest |
| Integration | Flask ↔ FastAPI communication | pytest + requests |
| Integration | Flask ↔ Airflow API | pytest + mocked responses |
| E2E | Full prediction flow | Manual / Playwright |
| E2E | Full training flow with progress | Manual |
| Visual | Noir design rendering | Manual browser check |

---

## Rollout Checklist

### Pre-merge (each branch)
- [ ] Code review completed
- [ ] Local docker-compose testing passes
- [ ] No console errors in browser

### Post-merge (main)
- [ ] Full `docker-compose --profile airflow up` works
- [ ] Prediction flow works end-to-end
- [ ] Training trigger works (Airflow DAG runs)
- [ ] Training progress displays correctly
- [ ] UI renders correctly on Chrome, Firefox, Safari

---

## Estimated Timeline

| Branch | Effort | Dependencies |
|--------|--------|--------------|
| `feature/flask-foundation` | 2-3 hours | None |
| `feature/booking-form` | 2-3 hours | Branch 1 |
| `feature/api-integration` | 4-5 hours | Branch 2 |

**Total**: ~9-11 hours of focused development

---

## Appendix: API Contracts

### POST /predict/ (FastAPI)

**Request**:
```json
{
  "data": [
    {
      "Booking_ID": "WEBA1B2C3D4",
      "number of adults": 2,
      "number of children": 1,
      "number of weekend nights": 2,
      "number of week nights": 3,
      "type of meal": "Meal Plan 1",
      "car parking space": 0,
      "room type": "Room_Type 2",
      "lead time": 45,
      "market segment type": "Online",
      "repeated": 0,
      "P-C": 0,
      "P-not-C": 2,
      "average price": 150.00,
      "special requests": 1,
      "date of reservation": "11/28/2025",
      "booking status": "Not_Canceled"
    }
  ]
}
```

**Response**:
```json
{
  "predictions": [0.73],
  "version": "1.0.0"
}
```

### POST /api/v1/dags/training_pipeline/dagRuns (Airflow)

**Request**:
```json
{
  "conf": {}
}
```

**Response**:
```json
{
  "dag_run_id": "manual__2025-11-28T14:32:05.123456+00:00",
  "state": "queued",
  "start_date": null,
  "end_date": null
}
```

### GET /api/v1/dags/training_pipeline/dagRuns/{dag_run_id} (Airflow)

**Response**:
```json
{
  "dag_run_id": "manual__2025-11-28T14:32:05.123456+00:00",
  "state": "running",
  "start_date": "2025-11-28T14:32:05.123456+00:00",
  "end_date": null
}
```

### GET /api/v1/dags/training_pipeline/dagRuns/{dag_run_id}/taskInstances (Airflow)

**Response**:
```json
{
  "task_instances": [
    {
      "task_id": "import_csv_data",
      "state": "success",
      "start_date": "2025-11-28T14:32:06+00:00",
      "end_date": "2025-11-28T14:32:15+00:00"
    },
    {
      "task_id": "validate_bronze_data",
      "state": "success",
      "start_date": "2025-11-28T14:32:16+00:00",
      "end_date": "2025-11-28T14:32:20+00:00"
    },
    {
      "task_id": "train_model",
      "state": "running",
      "start_date": "2025-11-28T14:32:21+00:00",
      "end_date": null
    },
    {
      "task_id": "validate_model",
      "state": null,
      "start_date": null,
      "end_date": null
    },
    {
      "task_id": "register_model",
      "state": null,
      "start_date": null,
      "end_date": null
    }
  ]
}
```
