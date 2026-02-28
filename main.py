from src.data_preprocessing import clean_data
from src.model_training import train


def main():
    print("Starting AI Sales Forecast Pipeline 🚀\n")

    print("Step 1: Data Preprocessing...")
    clean_data()

    print("Step 2: Model Training...")
    train()

    print("\nPipeline completed successfully ✅")


if __name__ == "__main__":
    main()
