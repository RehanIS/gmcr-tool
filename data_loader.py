import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from datetime import datetime

# ==========================================
# 1. NEON DB CONNECTION (SQLAlchemy)
# ==========================================
load_dotenv()

DATABASE_URL = os.getenv("NEON_DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("❌ Missing NEON_DATABASE_URL key in .env file")

try:
    # This creates a standard PostgreSQL connection
    engine = create_engine(DATABASE_URL)
except Exception as e:
    print(f"Neon DB Init Error: {e}")
    engine = None

# ==========================================
# 2. DATA FETCHING (READ)
# ==========================================
def fetch_training_data(platform):
    """
    Fetches all records (System + User Feedback) for a specific platform.
    """
    if engine is None:
        return pd.DataFrame()

    table_map = {
        'Azure': 'azure_training_data',
        'AWS': 'aws_training_data',
        'VMware': 'vmware_training_data'
    }
    
    table_name = table_map.get(platform)
    if not table_name:
        return pd.DataFrame()

    try:
        # Fetch all rows using pandas read_sql
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
        return df
            
    except Exception as e:
        print(f"Error fetching {platform}: {e}")
        return pd.DataFrame()

# ==========================================
# 3. DATA APPENDING (WRITE)
# ==========================================
def save_feedback(platform, user_inputs, actual_time):
    """
    Appends a user feedback row to the database.
    user_inputs: Dict containing 'size', 'region', etc.
    actual_time: The float value provided by the user.
    """
    if engine is None: return False

    table_map = {
        'Azure': 'azure_training_data',
        'AWS': 'aws_training_data',
        'VMware': 'vmware_training_data'
    }
    table_name = table_map.get(platform)

    # Construct Payload based on Platform Schema
    # Note: We match the LOWERCASING from our migration
    record = {
        "record_source": "user_feedback",
        "restore_time_min": float(actual_time),
        "created_at": datetime.utcnow().isoformat()
    }

    try:
        if platform == 'Azure':
            record.update({
                "vm_size_gb": user_inputs.get('size'),
                "region": user_inputs.get('region'),
                "vm_resource_id": "manual_feedback",
                "disk_tier": user_inputs.get('tier', 'Premium_SSD'),
                "vault_redundancy": user_inputs.get('redundancy', 'LRS'),
                "restore_method": user_inputs.get('method', 'Instant'),
                "network_bandwidth_mbps": 1000 # Default assumption for manual entry
            })
        elif platform == 'AWS':
            record.update({
                "vm_size_gb": user_inputs.get('size'),
                "region": user_inputs.get('region'),
                "instance_id": "manual_feedback",
                "ebs_volume_type": user_inputs.get('vol_type', 'gp3'),
                "provisioned_iops": user_inputs.get('iops', 3000),
                "snapshot_age_days": user_inputs.get('age', 1)
            })
        elif platform == 'VMware':
            record.update({
                "vm_size_gb": user_inputs.get('size'),
                "vm_id": "manual_feedback",
                "transport_mode": user_inputs.get('mode', 'HotAdd'),
                "disk_provisioning": "Thin",
                "target_storage": "SSD",
                "network_gbps": 10,
                "concurrency_level": 1
            })

        # Execute Insert using pandas to_sql
        df_record = pd.DataFrame([record])
        df_record.to_sql(table_name, engine, if_exists='append', index=False)
        return True

    except Exception as e:
        print(f"Feedback Insert Error: {e}")
        log_system_event("ERROR", "DB_Insert", f"Failed to insert feedback: {e}")
        return False

def log_system_event(level, component, message):
    """
    Writes a log entry to system_logs table.
    """
    if engine is None: return

    try:
        payload = {
            "level": level,
            "component": component,
            "message": str(message),
            "timestamp": datetime.utcnow().isoformat()
        }
        df_log = pd.DataFrame([payload])
        df_log.to_sql("system_logs", engine, if_exists='append', index=False)
    except Exception:
        pass # Never break the app if logging fails

# ==========================================
# 4. PRE-PROCESSING (LOWERCASE LOGIC)
# ==========================================
def calculate_physics_t_initial(row, platform):
    """
    Calculates T_initial using PHYSICS (Size / Speed).
    UPDATED: Uses lowercase keys to match Supabase schema.
    """
    S = row.get('vm_size_gb', 0)
    
    if platform == 'Azure':
        speed_mb_s = row.get('network_bandwidth_mbps', 0) / 8.0
        size_mb = S * 1024
        t_minutes = (size_mb / speed_mb_s) / 60.0 if speed_mb_s > 0 else 0
        
    elif platform == 'AWS':
        iops = row.get('provisioned_iops', 0)
        speed_mb_s = (iops * 0.25) # Approx MB/s per IO chunk
        t_minutes = ((S * 1024) / speed_mb_s) / 60.0 if speed_mb_s > 0 else 0
        
    else: # VMware
        speed_mb_s = (row.get('network_gbps', 0) * 1000) / 8.0
        t_minutes = ((S * 1024) / speed_mb_s) / 60.0 if speed_mb_s > 0 else 0

    return t_minutes

def clean_and_prep(df, platform, low_q=0.01, high_q=0.99):
    """
    Engineers features for AI training.
    UPDATED: Matches main.py Region logic EXACTLY (1-based index).
    """
    if df.empty:
        return None, None

    df = df.copy()
    
    # 1. Physics Feature
    df['t_initial_physics'] = df.apply(lambda row: calculate_physics_t_initial(row, platform), axis=1)

    # 2. Encoding & Feature Selection
    features = []
    
    # --- CRITICAL UPDATE: REGION MAPPING ---
    # We must match main.py logic: v4 = list.index(reg) + 1
    # This ensures inputs align perfectly.
    if 'region' in df.columns:
        # Sort uniques to match main.py's sorting
        unique_regions = sorted(df['region'].dropna().astype(str).unique())
        # Map: East US -> 1, West US -> 2
        region_map = {r: i+1 for i, r in enumerate(unique_regions)}
        df['region_code'] = df['region'].map(region_map).fillna(0)
    else:
        df['region_code'] = 0

    if platform == 'Azure':
        # restore_method -> Instant=1, else 0
        df['method_score'] = df['restore_method'].apply(lambda x: 1 if x == 'Instant' else 0)
        
        tier_map = {'Standard_HDD': 1, 'Standard_SSD': 2, 'Premium_SSD': 3, 'Ultra': 4}
        df['tier_score'] = df['disk_tier'].map(tier_map).fillna(2)
        
        features = ['vm_size_gb', 'method_score', 'tier_score', 'region_code', 'network_bandwidth_mbps', 't_initial_physics']
        
    elif platform == 'AWS':
        vol_map = {'st1': 1, 'gp2': 2, 'gp3': 3, 'io2': 4}
        df['vol_score'] = df['ebs_volume_type'].map(vol_map).fillna(2)
            
        features = ['vm_size_gb', 'vol_score', 'provisioned_iops', 'snapshot_age_days', 'region_code', 't_initial_physics']
        
    elif platform == 'VMware':
        mode_map = {'NBD': 1, 'HotAdd': 2, 'SAN': 3}
        df['mode_score'] = df['transport_mode'].map(mode_map).fillna(1)
        
        store_map = {'HDD': 1, 'SSD': 3, 'NVMe': 5}
        df['storage_score'] = df['target_storage'].map(store_map).fillna(3)
        
        features = ['vm_size_gb', 'mode_score', 'concurrency_level', 'network_gbps', 'storage_score', 't_initial_physics']

    # 3. Target
    target = 'restore_time_min'
    if target not in df.columns:
        return None, None

    # 4. Outlier Removal (Robust)
    df.dropna(subset=features + [target], inplace=True)
    
    q_low_val = df[target].quantile(low_q)
    q_high_val = df[target].quantile(high_q)
    
    df_clean = df[(df[target] >= q_low_val) & (df[target] <= q_high_val)].copy()

    return df_clean[features].values, df_clean[target].values


# ==========================================
# 5. HISTORICAL TREND DATA (NEW)
# ==========================================
def fetch_historical_trend(platform):
    """
    Fetches restore time data with timestamps for trend analysis.
    Returns a DataFrame sorted by created_at.
    """
    if engine is None:
        return pd.DataFrame()

    table_map = {
        'Azure': 'azure_training_data',
        'AWS': 'aws_training_data',
        'VMware': 'vmware_training_data'
    }
    table_name = table_map.get(platform)
    if not table_name:
        return pd.DataFrame()

    try:
        # Standard SQL query directly into Pandas
        query = f"""
            SELECT restore_time_min, created_at, vm_size_gb, record_source 
            FROM {table_name} 
            ORDER BY created_at ASC
        """
        df = pd.read_sql(query, engine)
        
        if not df.empty:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            df.dropna(subset=['created_at', 'restore_time_min'], inplace=True)
            return df
            
        return pd.DataFrame()
    except Exception as e:
        print(f"Historical trend fetch error: {e}")
        return pd.DataFrame()


# ==========================================
# 6. ALL-PLATFORM SUMMARY (NEW)
# ==========================================
def fetch_all_platforms_summary():
    """
    Fetches summary statistics for all platforms.
    Used by the Multi-Cloud Comparison tab.
    """
    summary = {}
    for plat in ['Azure', 'AWS', 'VMware']:
        df = fetch_training_data(plat)
        if not df.empty and 'restore_time_min' in df.columns:
            col = df['restore_time_min']
            summary[plat] = {
                'count': len(df),
                'mean': round(col.mean(), 1),
                'median': round(col.median(), 1),
                'min': round(col.min(), 1),
                'max': round(col.max(), 1),
                'std': round(col.std(), 1)
            }
        else:
            summary[plat] = {'count': 0, 'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0}
    return summary
    