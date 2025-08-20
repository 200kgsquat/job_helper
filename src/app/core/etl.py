import os
import pandas as pd

class ETLProcessor:
    def __init__(self, input_file, output_folder):
        self.input_file = input_file
        self.output_folder = output_folder
        self.output_file = os.path.join(output_folder, "cleaned_job_postings.csv")

    def load_data(self):
        """Load the dataset."""
        return pd.read_csv(self.input_file)

    def filter_quantiles(self, df, column, lower_quantile=0.01, upper_quantile=0.99):
        """Filter rows based on quantiles of a column."""
        low = df[column].quantile(lower_quantile)
        high = df[column].quantile(upper_quantile)
        return df[(df[column] >= low) & (df[column] <= high)]

    def filter_top_industries(self, df, column, top_n=10):
        """Filter rows to keep only the top N most popular industries."""
        top_industries = df[column].value_counts().nlargest(top_n).index
        return df[df[column].isin(top_industries)]

    def transform(self):
        """Run the ETL process."""
        # Load data
        df = self.load_data()

        # Drop rows with missing values in critical columns
        df = df.dropna(subset=['industry', 'description'])

        # Remove duplicates
        df = df.drop_duplicates()

        # Add word count column
        df['word_count'] = df['description'].apply(lambda x: len(str(x).split()))

        # Filter rows based on word count quantiles
        df = self.filter_quantiles(df, 'word_count')

        # Filter rows to keep only the top 10 most popular industries
        df = self.filter_top_industries(df, 'industry', top_n=10)

        # Select relevant columns
        df = df[['description', 'industry']]

        # Ensure the output folder exists
        os.makedirs(self.output_folder, exist_ok=True)

        # Save cleaned data
        df.to_csv(self.output_file, index=False)
        print(f"Cleaned data saved to {self.output_file}")


