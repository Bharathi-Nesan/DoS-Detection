import pandas as pd
import time
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score

def run():
    print("="*85)
    print(f"{'PHASE 4: 10-FOLD CV & LATENCY ANALYSIS':^85}")
    print("="*85)
    
    # 1. Load Data
    try:
        train = pd.read_csv('train_data.csv')
        # Using the blind and truth files generated in Phase 3
        test_blind = pd.read_csv('test_no_labels.csv') # 16 columns (Real world input)
        test_truth = pd.read_csv('test_with_labels.csv') # 17 columns (Validation key)
    except FileNotFoundError:
        print("Error: Required CSV files not found! Please run Phase 3.")
        return

    # 2. Prepare Features and Targets
    X_train, y_train = train.drop(columns=['Attack type']), train['Attack type']
    X_test = test_blind 
    y_test = test_truth['Attack type']

    # 3. Updated Model Definitions to Match Base Paper Benchmarks
    models = {
        # Removing max_depth allows the tree to reach the 99.5% paper benchmark
        "Decision Tree": DecisionTreeClassifier(random_state=42), 
        
        # 100 estimators is the standard for the 99.7% RF paper result
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        
        # k=5 is the optimal balance for KNN accuracy in WSN-DS
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5)
    }

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Table Header
    header = f"{'Model':<22} | {'CV Mean Acc':<12} | {'Test Acc':<10} | {'Latency/Pkt':<12}"
    print(header)
    print("-" * len(header))

    for name, model in models.items():
        # A. 10-Fold Cross-Validation on Training Data
        cv_scores = cross_val_score(model, X_train, y_train, cv=kf, n_jobs=-1)
        mean_cv = cv_scores.mean() * 100
        
        # B. Final Training
        model.fit(X_train, y_train)
        
        # C. Latency Measurement (Simulating Real-World Inference)
        start_time = time.time()
        y_pred = model.predict(X_test)
        end_time = time.time() - start_time
        
        # D. Evaluation against Ground Truth
        latency_per_packet = end_time / len(X_test)
        test_acc = accuracy_score(y_test, y_pred) * 100

        # Print Row
        print(f"{name:<22} | {mean_cv:>10.2f}% | {test_acc:>8.2f}% | {latency_per_packet:.8f}s")
        
        # --- UPDATED SAVING SECTOR ---
        # Save each model for the multi-model dashboard comparison
        if name == "Decision Tree":
            joblib.dump(model, 'wsn_dt.pkl')
            # Keeping backward compatibility for the main model file
            joblib.dump(model, 'wsn_dos_model.pkl')
        elif name == "Random Forest":
            joblib.dump(model, 'wsn_rf.pkl')
        elif name == "KNN (k=5)":
            joblib.dump(model, 'wsn_knn.pkl')

    print("=" * len(header))
    print("[SUCCESS]: All three models (DT, RF, KNN) have been saved with paper-aligned settings.")

if __name__ == "__main__":
    run()