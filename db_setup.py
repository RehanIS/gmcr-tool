import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# ==========================================
# 🔑 DB CONNECTION (NEON POSTGRESQL)
# ==========================================
# Load environment variables from .env file
load_dotenv()

DATABASE_URL = os.getenv("NEON_DATABASE_URL")

if not DATABASE_URL:
    print("❌ Connection Failed: Missing NEON_DATABASE_URL in .env file")
    exit()

# Initialize SQLAlchemy Engine
try:
    engine = create_engine(DATABASE_URL)
    print("✅ Connected to Neon DB via SQLAlchemy!")
except Exception as e:
    print(f"❌ Connection Failed: {e}")
    exit()

def push_csv_to_table(csv_filename, table_name):
    file_path = os.path.join("data", csv_filename)
    
    if not os.path.exists(file_path):
        print(f"⚠️  Missing file: {file_path}")
        return

    print(f"📂 Processing {csv_filename}...")
    
    try:
        # 1. Read CSV
        df = pd.read_csv(file_path)
        
        # 2. Lowercase Headers (Matches SQL Schema)
        df.columns = df.columns.str.lower()
        
        # 3. Add Metadata
        # 'id' and 'created_at' should be set to auto-generate in your Neon DB schema
        df['record_source'] = 'system'
        
        # 4. Push Data to Neon (Handles chunking automatically!)
        # chunksize=100 replicates your previous batch insert logic safely
        df.to_sql(
            name=table_name, 
            con=engine, 
            if_exists='append', 
            index=False, 
            chunksize=100
        )
        
        print(f"   🚀 Uploaded {len(df)} rows to '{table_name}'")

    except Exception as e:
        print(f"   ❌ Error: {e}")

# ==========================================
# 🚀 EXECUTION
# ==========================================
if __name__ == "__main__":
    print("🔄 Starting Neon DB Migration...")
    
    # 1. AWS
    push_csv_to_table("aws_restore_data.csv", "aws_training_data")
    
    # 2. Azure
    push_csv_to_table("azure_restore_data.csv", "azure_training_data")
    
    # 3. VMware
    push_csv_to_table("vmware_restore_dataset.csv", "vmware_training_data")
    
    print("\n✨ Migration Finished.")