"""
One-time setup script for Great Expectations.
Run: python app/validations/setup_ge.py
"""

from pathlib import Path

import great_expectations as gx


def initialize_great_expectations():
    """
    Initialize Great Expectations for NoVacancy project.
    Creates file-based context and configures pandas datasource with data assets.
    """
    project_root = Path(__file__).parent

    # Create context (replaces CLI init)
    context = gx.get_context(mode="file", project_root_dir=str(project_root))

    # Configure pandas datasource; describe souce type, not the project
    try:
        datasource = context.data_sources.add_pandas(name="pandas_datasource")

    except Exception:
        # Datasource already exists
        datasource = context.data_sources.get("pandas_datasource")

    # Add data assets for each validation layer
    asset_names = {
        "bronze_bookings": "Raw CSV data after import",
        "silver_processed": "Transformed features after preprocessing",
        "model_input": "Final features before model inference",
    }

    for asset_name, description in asset_names.items():
        try:
            datasource.add_dataframe_asset(name=asset_name)
            print(f"✅ Created asset: {asset_name} ({description})")
        except Exception:
            print(f"⚠️  Asset '{asset_name}' already exists")

    print("\n✅ Great Expectations initialized!")
    print(f"📁 Config location: {project_root / 'gx'}")

    return context


if __name__ == "__main__":
    initialize_great_expectations()
