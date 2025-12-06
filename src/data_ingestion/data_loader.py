"""
Load raw data from various source
-Local CSV
-BigQuery
...
"""
from abc import ABC, abstractmethod
import pandas as pd
import logging 
from pathlib import Path
from typing import Union,Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RawData:
    """
    Container for raw data and data attributes
    """
    transactions: pd.DataFrame
    file_path:str
    n_rows:int
    n_customers:int
    n_orders: int
    n_products:int
    date_range:tuple

class BaseIngestionStrategy(ABC):
    """
    Abstract interfact for all ingestion type
    """
    @abstractmethod
    def load(self) -> RawData:
        """Return data in a RawData container."""
        pass

class CSVIngestion(BaseIngestionStrategy):
    """
    Local CSV ingestion strategy and return RawData format
    """
    def __init__(self, file_path: str, date_columns=None):
            self.file_path = Path(file_path)
            #Project Specific - review if reuse
            self.date_columns = date_columns or [
                "order_date", "first_order_date", "last_order_date"
            ]

    def load(self) ->RawData:

        #assert file in path
        if not self.file_path.exists():
             raise FileNotFoundError(f"Data file not found: {self.file_path}")
        logger.info(f"Loading data from {self.file_path}")

        #Load CSV
        df = pd.read_csv(self.file_path,parse_dates=self.date_columns,index_col=0)

        #Update product grain at product_variable level
        df['product'] = df['product_name'] + ' (' + df['product_variant'].fillna('Regular') + ')'

        #Stat
        n_rows = len(df)
        n_customers = df["customer_id"].nunique()
        n_orders = df["order_id"].nunique()
        n_products = df['product'].nunique()
        date_range = (df["order_date"].min(),df["order_date"].max())

        logger.info(f"Loaded {n_rows} rows")
        logger.info(f"Customers: {n_customers:,}")
        logger.info(f"Orders: {n_orders:,}")
        logger.info(f"Products: {n_products:,}")
        logger.info(f"Date range: {date_range[0]} to {date_range[1]}")

        return RawData(
            transactions=df,
            file_path=str(self.file_path),
            n_rows=n_rows,
            n_customers=n_customers,
            n_orders=n_orders,
            n_products=n_products,
            date_range=date_range
        )
    

class BigQueryIngestion(BaseIngestionStrategy):
    """
    Bigquery ingestion strategy and return RawData format
    """
    def __init__(self, project_id: str,query = str,credentials_path: Optional[str] = None):
            self.project_id = project_id
            self.query = query
            self.credentials_path = credentials_path

    def load(self) ->RawData:

        try:
            from google.cloud import bigquery
            from google.oauth2 import service_account
        except ImportError:
             raise ImportError("Install Bigquery depencencies google.cloud, google.oauth2")

        logger.info("Loading data from BigQuery...")
        if self.credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path
            )
            client = bigquery.Client(project=self.project_id, credentials=credentials)
        else:
            client = bigquery.Client(project=self.project_id)

        #Running Query and to Pandas Dataframe
        logger.info("Executing SQL query on BigQuery")
        df = client.query(self.query).to_dataframe()

        if "order_date" in df.columns:
            df["order_date"] = pd.to_datetime(df["order_date"])

        if "product_name" in df.columns:
            df["product"] = df["product_name"] + " (" + df["product_variant"].fillna("Regular") + ")"

        #Stat
        #Move to return to handle various column
        # n_rows = len(df)
        # n_customers = df["customer_id"].nunique()
        # n_orders = df["order_id"].nunique()
        # n_products = df["product"].nunique()
        # date_range = (df["order_date"].min(),df["order_date"].max())

        # logger.info(f"Loaded {n_rows} rows")
        # logger.info(f"Customers: {n_customers:,}")
        # logger.info(f"Orders: {n_orders:,}")
        # logger.info(f"Products: {n_products:,}")
        # logger.info(f"Date range: {date_range[0]} to {date_range[1]}")

        return RawData(
            transactions=df,
            file_path=str(f"bigquery://{self.project_id}"),
            n_rows=len(df),
            n_customers=df['customer_id'].nunique() if 'customer_id' in df else None,
            n_orders=df["order_id"].nunique() if "order_id" in df else None,
            n_products=df["product"].nunique() if "product" in df else None ,
            date_range=(df["order_date"].min() if "order_date" in df else None,
                        df["order_date"].max() if "order_date" in df else None)
            )
    

class IngestionFactory:
     @staticmethod
     def create(source_type:str,**kwargs) -> BaseIngestionStrategy:
          source_type = source_type.lower()

          if source_type == "csv":
               return CSVIngestion(
                    file_path= kwargs.get("file_path"),
                    date_columns=  kwargs.get("date_columns")
               ) 
          
          elif source_type == "bigquery":
            return BigQueryIngestion(
                project_id=kwargs.get("project_id"),
                query=kwargs.get("query"),
                credentials_path=kwargs.get("credentials_path")
            )
          else:
               raise ModuleNotFoundError(f"No module configured for {source_type}")
    
class DataLoader:
    def __init__(self, ingestion_strategy: BaseIngestionStrategy):
        self.ingestion_strategy = ingestion_strategy
        logger.info(f"Using ingestion method: {self.ingestion_strategy.__class__.__name__}")

    def load(self) -> RawData:
        return self.ingestion_strategy.load()
    

if __name__ == "__main__":
     pass
"""
Use example
#Load CSV
ingestion = IngestionFactory.create(
    source_type="csv",
    file_path = 'data/raw/data_raw.csv',
    date_columns=["order_date", "first_order_date", "last_order_date"]
)

loader = DataLoader(ingestion)
raw = loader.load()

#Load via bigquery
qeury = "SQL Query"

ingestion = IngestionFactory.create(
    source_type="bigquery",
    project_id = 'project id',
    query=query
)

loader = DataLoader(ingestion)
raw = loader.load()
"""

          
    

    


        
        

