import pandas as pd
from sklearn.model_selection import train_test_split

def run():
    print("--- Phase 3: Data Splitting ---")
    df = pd.read_csv('optimized_wsn.csv')
    

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

  
    train_df.to_csv('train_data.csv', index=False)

    test_df.to_csv('test_with_labels.csv', index=False)

  
    test_no_labels = test_df.drop(columns=['Attack type'])
    test_no_labels.to_csv('test_no_labels.csv', index=False)

    print(f"Files Generated:")
    print(f"- train_data.csv: {train_df.shape}")
    print(f"- test_with_labels.csv: {test_df.shape} (17 columns)")
    print(f"- test_no_labels.csv: {test_no_labels.shape} (16 columns)")
    print("Phase 3 Complete.")

if __name__ == "__main__":
    run()