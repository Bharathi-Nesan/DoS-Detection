import pandas as pd
from sklearn.model_selection import train_test_split

def run():
    print("--- Phase 3: Data Splitting ---")
    df = pd.read_csv('optimized_wsn.csv')
    
    # 1. Split into Training (80%) and Testing (20%)
    # random_state=42 ensures the split is identical every run
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 2. Save the Training Data (Used in Phase 4)
    train_df.to_csv('train_data.csv', index=False)

    # 3. Save the Testing Data WITH the 17th column (For validation/accuracy checking)
    test_df.to_csv('test_with_labels.csv', index=False)

    # 4. Save the Testing Data WITHOUT the 17th column (To simulate real-world input)
    # This file will have only 16 columns
    test_no_labels = test_df.drop(columns=['Attack type'])
    test_no_labels.to_csv('test_no_labels.csv', index=False)

    print(f"Files Generated:")
    print(f"- train_data.csv: {train_df.shape}")
    print(f"- test_with_labels.csv: {test_df.shape} (17 columns)")
    print(f"- test_no_labels.csv: {test_no_labels.shape} (16 columns)")
    print("Phase 3 Complete.")

if __name__ == "__main__":
    run()