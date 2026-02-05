import pandas as pd
from sklearn.model_selection import train_test_split

def run():
    print("--- Phase 3: Data Splitting ---")
    df = pd.read_csv('optimized_wsn.csv')
    
    # random_state=42 ensures the 80/20 split is identical every run
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_csv('train_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)
    print("Phase 3 Complete.")