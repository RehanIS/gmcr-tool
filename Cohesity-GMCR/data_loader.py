import pandas as pd
import numpy as np
import os

def load_files():
    # Dynamic path handling: Finds the 'data' folder relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    
    files = {
        'azure': os.path.join(data_dir, 'azure_restore_data.csv'),
        'aws': os.path.join(data_dir, 'aws_restore_data.csv'),
        'vmware': os.path.join(data_dir, 'vmware_restore_dataset.csv')
    }

    try:
        df_az = pd.read_csv(files['azure'])
        df_aw = pd.read_csv(files['aws'])
        df_vm = pd.read_csv(files['vmware'])
        return df_az, df_aw, df_vm
    except FileNotFoundError as e:
        print(f"❌ File Error: {e}")
        return None, None, None

def calculate_physics_t_initial(row, platform):
    """
    Calculates the Theoretical Time (T_initial) based on physics.
    This creates the 6th feature.
    """
    # 1. Get Size
    S = row['VM_Size_GB']
    
    # 2. Get Bandwidth/Speed
    # We use simple physics: Time = Size / Speed
    if platform == 'Azure':
        # Mbps to MB/s = /8
        speed_mb_s = row['Network_Bandwidth_Mbps'] / 8.0
        # Size GB to MB = *1024
        size_mb = S * 1024
        if speed_mb_s > 0:
            t_minutes = (size_mb / speed_mb_s) / 60.0
        else:
            t_minutes = 0
        
    elif platform == 'AWS':
        # AWS often limited by IOPS. Approx 256KB per IO.
        # This is a rough estimation for the "Physics" baseline
        iops = row['Provisioned_IOPS']
        speed_mb_s = (iops * 0.25) # Approx MB/s
        if speed_mb_s > 0:
            t_minutes = ((S * 1024) / speed_mb_s) / 60.0
        else:
            t_minutes = 0
        
    else: # VMware
        # Gbps to MB/s
        speed_mb_s = (row['Network_Gbps'] * 1000) / 8.0
        if speed_mb_s > 0:
            t_minutes = ((S * 1024) / speed_mb_s) / 60.0
        else:
            t_minutes = 0

    return t_minutes

def clean_and_prep(df, platform, low_q=0.01, high_q=0.99):
    """
    Clean data, engineer features, and remove outliers.
    Now accepts low_q and high_q from the sidebar slider.
    """
    df = df.copy()
    df.dropna(inplace=True)

    # --- 1. GENERATE THE 6TH FEATURE (PHYSICS FORMULA) ---
    df['T_initial_Physics'] = df.apply(lambda row: calculate_physics_t_initial(row, platform), axis=1)

    # --- 2. ENCODING (Text -> Numbers) ---
    features = []
    
    if platform == 'Azure':
        df['Method_Score'] = df['Restore_Method'].apply(lambda x: 1 if x == 'Instant' else 0)
        tier_map = {'Standard_HDD': 1, 'Standard_SSD': 2, 'Premium_SSD': 3, 'Ultra': 4}
        df['Tier_Score'] = df['Disk_Tier'].map(tier_map).fillna(2)
        
        # Safe Region Encoding
        if 'Region' in df.columns:
            df['Region_Code'] = df['Region'].astype('category').cat.codes
        else:
            df['Region_Code'] = 0
        
        # DEFINING THE 6 FEATURES
        features = ['VM_Size_GB', 'Method_Score', 'Tier_Score', 'Region_Code', 'Network_Bandwidth_Mbps', 'T_initial_Physics']
        
    elif platform == 'AWS':
        vol_map = {'st1': 1, 'gp2': 2, 'gp3': 3, 'io2': 4}
        # Check if column is 'EBS_Volume_Type' or 'Volume_Type' (Adjust based on your CSV)
        vol_col = 'EBS_Volume_Type' if 'EBS_Volume_Type' in df.columns else 'Volume_Type'
        df['Vol_Score'] = df[vol_col].map(vol_map).fillna(2)
        
        if 'Region' in df.columns:
            df['Region_Code'] = df['Region'].astype('category').cat.codes
        else:
            df['Region_Code'] = 0
        
        features = ['VM_Size_GB', 'Vol_Score', 'Provisioned_IOPS', 'Snapshot_Age_Days', 'Region_Code', 'T_initial_Physics']
        
    elif platform == 'VMware':
        mode_map = {'NBD': 1, 'HotAdd': 2, 'SAN': 3}
        df['Mode_Score'] = df['Transport_Mode'].map(mode_map).fillna(1)
        store_map = {'HDD': 1, 'SSD': 3, 'NVMe': 5}
        df['Storage_Score'] = df['Target_Storage'].map(store_map).fillna(3)
        
        features = ['VM_Size_GB', 'Mode_Score', 'Concurrency_Level', 'Network_Gbps', 'Storage_Score', 'T_initial_Physics']

    # Identify Target Column (Handle variations in naming)
    if 'Restore_Time_Min' in df.columns:
        target = 'Restore_Time_Min'
    elif 'Restore_Time_Minutes' in df.columns:
        target = 'Restore_Time_Minutes'
    else:
        # Fallback if neither exists
        raise ValueError(f"Target column not found in {platform}. Check CSV headers.")

    # --- 3. CLEAN OUTLIERS (Dynamic Filter) ---
    # Use the low_q and high_q passed from main.py
    q_low_val = df[target].quantile(low_q)
    q_high_val = df[target].quantile(high_q)
    
    df_clean = df[(df[target] >= q_low_val) & (df[target] <= q_high_val)].copy()

    # Return X (Features) and y (Target)
    return df_clean[features].values, df_clean[target].values