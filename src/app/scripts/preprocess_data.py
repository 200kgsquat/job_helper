from src.app.core.etl import ETLProcessor

def main():
    # Define input file and output folder
    input_file = "data/fake_job_postings.csv"
    output_folder = "data"

    # Initialize and run the ETL processor
    etl = ETLProcessor(input_file, output_folder)
    etl.transform()

if __name__ == "__main__":
    main()