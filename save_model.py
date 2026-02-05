import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier

def export_model():
    print("--- GENERATING MODEL PKL FILE ---")
    
    # 1. Load your training data
    try:
        train = pd.read_csv('train_data.csv')
    except FileNotFoundError:
        print("Error: train_data.csv not found. Please run Phase 3 first!")
        return

    X_train = train.drop(columns=['Attack type'])
    y_train = train['Attack type']

    # 2. Define the exact same model used in your project
    # Using max_depth=10 and random_state=42 to match your results
    model = DecisionTreeClassifier(max_depth=10, random_state=42)

    # 3. Train it
    print("Training model for export...")
    model.fit(X_train, y_train)

    # 4. SAVE the model as a .pkl file
    joblib.dump(model, 'wsn_dos_model.pkl')
    
    print("SUCCESS: 'wsn_dos_model.pkl' has been created in your folder.")
    print("You can now run 'streamlit run app.py'.")

if __name__ == "__main__":
    export_model()