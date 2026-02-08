"""
Supabase configuration and helper functions for the library occupancy system.
"""
import os
from supabase import create_client, Client
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import json

# Load environment variables
load_dotenv()

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


class SupabaseStorage:
    """Helper class for Supabase storage operations."""

    @staticmethod
    def upload_file_to_storage(file_path: str, bucket_name: str, destination_path: str):
        """
        Upload a file to Supabase Storage.

        Args:
            file_path: Local file path
            bucket_name: Supabase storage bucket name
            destination_path: Path in the storage bucket

        Returns:
            str: Public URL of uploaded file
        """
        with open(file_path, 'rb') as f:
            file_data = f.read()

        # Upload to Supabase Storage
        response = supabase.storage.from_(bucket_name).upload(
            destination_path,
            file_data,
            file_options={"upsert": "true"}
        )

        # Get public URL
        public_url = supabase.storage.from_(bucket_name).get_public_url(destination_path)
        return public_url

    @staticmethod
    def download_file_from_storage(bucket_name: str, file_path: str, destination_path: str):
        """
        Download a file from Supabase Storage.

        Args:
            bucket_name: Supabase storage bucket name
            file_path: Path in the storage bucket
            destination_path: Local destination path

        Returns:
            bool: Success status
        """
        try:
            response = supabase.storage.from_(bucket_name).download(file_path)

            with open(destination_path, 'wb') as f:
                f.write(response)

            return True
        except Exception as e:
            print(f"Error downloading file: {e}")
            return False

    @staticmethod
    def save_wifi_data(df: pd.DataFrame, library_id: str = None):
        """
        Save WiFi data to Supabase.

        Args:
            df: DataFrame with columns: AP MAC, Client MAC, Start_dt, etc.
            library_id: Optional library identifier
        """
        # Convert DataFrame to list of dicts
        records = df.to_dict('records')

        # Add timestamp and library_id if provided
        for record in records:
            if 'created_at' not in record:
                record['created_at'] = datetime.utcnow().isoformat()
            if library_id:
                record['library_id'] = library_id

        # Insert in batches of 1000 to avoid payload limits
        batch_size = 1000
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            response = supabase.table('wifi_data').insert(batch).execute()

        return len(records)

    @staticmethod
    def get_wifi_data(library_id: str = None, start_date: str = None, end_date: str = None):
        """
        Retrieve WiFi data from Supabase.

        Args:
            library_id: Filter by library
            start_date: Start date (ISO format)
            end_date: End date (ISO format)

        Returns:
            pandas DataFrame
        """
        query = supabase.table('wifi_data').select('*')

        if library_id:
            query = query.eq('library_id', library_id)
        if start_date:
            query = query.gte('Start_dt', start_date)
        if end_date:
            query = query.lte('Start_dt', end_date)

        response = query.execute()
        return pd.DataFrame(response.data)

    @staticmethod
    def save_predictions_cache(predictions_dict: dict):
        """
        Save predictions cache to Supabase.

        Args:
            predictions_dict: Dictionary of predictions by library
        """
        record = {
            'predictions': json.dumps(predictions_dict),
            'updated_at': datetime.utcnow().isoformat()
        }

        # Upsert (insert or update) the predictions cache
        response = supabase.table('predictions_cache').upsert(record, on_conflict='id').execute()
        return response

    @staticmethod
    def get_predictions_cache():
        """
        Retrieve latest predictions cache from Supabase.

        Returns:
            dict: Predictions dictionary
        """
        response = supabase.table('predictions_cache').select('*').order('updated_at', desc=True).limit(1).execute()

        if response.data and len(response.data) > 0:
            return json.loads(response.data[0]['predictions'])
        return {}

    @staticmethod
    def upload_model_metadata(model_type: str, library_id: str, metrics: dict):
        """
        Save model training metadata to Supabase.

        Args:
            model_type: Type of model (e.g., 'advanced_cnn_lstm')
            library_id: Library identifier
            metrics: Dictionary of model metrics (R2, RMSE, etc.)
        """
        record = {
            'model_type': model_type,
            'library_id': library_id,
            'metrics': json.dumps(metrics),
            'trained_at': datetime.utcnow().isoformat()
        }

        response = supabase.table('model_metadata').insert(record).execute()
        return response

    @staticmethod
    def get_model_metadata(model_type: str = None, library_id: str = None):
        """
        Retrieve model metadata from Supabase.

        Args:
            model_type: Filter by model type
            library_id: Filter by library

        Returns:
            list: Model metadata records
        """
        query = supabase.table('model_metadata').select('*')

        if model_type:
            query = query.eq('model_type', model_type)
        if library_id:
            query = query.eq('library_id', library_id)

        response = query.execute()
        return response.data
