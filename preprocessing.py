import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def run():
    print("--- Phase 1: Cleaning & Scaling ---")
    df = pd.read_csv('WSN-DS.csv')
    df.columns = df.columns.str.strip()
    
    le = LabelEncoder()
    df['Attack type'] = le.fit_transform(df['Attack type'])
    
    
    for col in df.columns:
        df[col] = df[col].fillna(df[col].mean())

    scaler = MinMaxScaler()
    X = df.drop(columns=['Attack type'])
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    pd.concat([X_scaled, df['Attack type']], axis=1).to_csv('cleaned_wsn.csv', index=False)
    print("Phase 1 Complete.")