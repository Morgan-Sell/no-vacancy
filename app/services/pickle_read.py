import joblib

def main():
    # Load the saved model
    pkl_file_path = "app/models/no_vacancy_pipeline.pkl"  # Adjust if needed
    artifacts = joblib.load(pkl_file_path)

    print("Loaded artifacts keys:", artifacts.keys())  # Should be 'pipeline' and 'processor'

    pipeline = artifacts["pipeline"]
    processor = artifacts["processor"]

    # Print object types
    print("\nType of pipeline:", type(pipeline))
    print("Type of processor:", type(processor))

    # Print pipeline attributes
    print("\nPipeline attributes:", dir(pipeline))

    # Print processor attributes
    print("\nProcessor attributes:", dir(processor))

if __name__ == "__main__":
    main()
