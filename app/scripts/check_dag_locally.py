# A script to run locally to confirm successful DAG execution
import sys
from pathlib import Path

app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))
print("app_dir: ", app_dir)

try:
    from dags.training_pipeline_dag import dag

    print(f"✅ DAG '{dag.dag_id}' imports successfully with {len(dag.tasks)} tasks")
except Exception as e:
    print(f"❌ DAG import failed: {e}")
    sys.exit(1)
