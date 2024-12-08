import pandas as pd
from abc import ABC, abstractmethod
import zipfile
import os

class IDataLoad(ABC):
    @abstractmethod
    def ingest(self, file_path:str)->pd.DataFrame:
        pass
    

class CSVDataReader(IDataLoad):
    def ingest(self, file_path):
        if not file_path.endswith('.csv'):
            raise ValueError("The provided file is not a .csv file")
        
        df = pd.read_csv(file_path)
        return df

class ZipDataReader(IDataLoad):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Extracts a .zip file and returns the content as a pandas DataFrame."""
        # Ensure the file is a .zip
        if not file_path.endswith(".zip"):
            raise ValueError("The provided file is not a .zip file.")

        # Extract the zip file
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall("extracted_data")

        # Find the extracted CSV file (assuming there is one CSV file inside the zip)
        extracted_files = os.listdir("extracted_data")
        csv_files = [f for f in extracted_files if f.endswith(".csv")]

        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in the extracted data.")
        if len(csv_files) > 1:
            raise ValueError("Multiple CSV files found. Please specify which one to use.")

        # Read the CSV into a DataFrame
        csv_file_path = os.path.join("extracted_data", csv_files[0])
        df = pd.read_csv(csv_file_path)

        # Return the DataFrame
        return df 



# In this way multiple ingestors can be added for different file types

class DataLoader:
    @staticmethod
    def get_ingestor(file_extension:str)->pd.DataFrame:
        if file_extension ==".csv":
            return CSVDataReader()
        elif file_extension == ".zip":
            return ZipDataReader()
        else:
            raise ValueError(f"No ingestor found with the given file extension {file_extension}")
    

    

if __name__ =="__main__":
    data_loader = DataLoader()
    data = data_loader.get_ingestor(".csv").ingest("C:\Documents\Code\TVaritAssignment2\DSData_Assignments 1.csv")
    print(data.head())
      
    
