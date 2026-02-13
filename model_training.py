import pandas as pd
import time
import joblib
import os  # To calculate physical file size on disk
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score

def run():
    print("="*95)
    print(f"{'PHASE 4: 10-FOLD CV, LATENCY & STORAGE ANALYSIS':^95}")
    print("="*95)
    
    # 1. Load Preprocessed Data from Phase 3
    try:
        train = pd.read_csv('train_data.csv')
        test_blind = pd.read_csv('test_no_labels.csv') # 16 features (No target)
        test_truth = pd.read_csv('test_with_labels.csv') # 17 columns (Ground truth)
    except FileNotFoundError:
        print("Error: Required CSV files not found! Please run Phase 3.")
        return

    # 2. Prepare Features (X) and Target Labels (y)
    X_train, y_train = train.drop(columns=['Attack type']), train['Attack type']
    X_test = test_blind 
    y_test = test_truth['Attack type']

    # 3. Define Models (Pruned DT Focus + Benchmarks)
    models = {
        # Pruning to depth 10 ensures high accuracy with a small memory footprint
        "Pruned Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5)
    }

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Header for Comparison Table
    header = f"{'Model':<22} | {'CV Mean Acc':<12} | {'Test Acc':<10} | {'Latency/Pkt':<12}"
    print(header)
    print("-" * len(header))

    dt_size_kb = 0 

    for name, model in models.items():
        # A. 10-Fold Cross-Validation (Stability Check)
        cv_scores = cross_val_score(model, X_train, y_train, cv=kf, n_jobs=-1)
        mean_cv = cv_scores.mean() * 100
        
        # B. Final Training on Training Dataset
        model.fit(X_train, y_train)
        
        # C. Latency Measurement (Simulated Inference)
        start_time = time.time()
        y_pred = model.predict(X_test)
        execution_time = time.time() - start_time
        
        # D. Metrics Calculation
        latency_per_packet = execution_time / len(X_test)
        test_acc = accuracy_score(y_test, y_pred) * 100

        print(f"{name:<22} | {mean_cv:>10.2f}% | {test_acc:>8.2f}% | {latency_per_packet:.8f}s")
        
        # E. Model Saving Sector
        if name == "Pruned Decision Tree":
            joblib.dump(model, 'wsn_dt.pkl')
            joblib.dump(model, 'wsn_dos_model.pkl') # Backward compatibility
            # Measure physical file size in KB
            dt_size_kb = os.path.getsize('wsn_dt.pkl') / 1024
        elif name == "Random Forest":
            joblib.dump(model, 'wsn_rf.pkl')
        elif name == "KNN (k=5)":
            joblib.dump(model, 'wsn_knn.pkl')

    print("-" * len(header))
    # Proof of Lightweight Optimization
    print(f"📦 [STORAGE METRIC]: Pruned Decision Tree Model Size: {dt_size_kb:.2f} KB")
    print("=" * len(header))
    print("[SUCCESS]: Models evaluated and saved. Ready for real-time terminal testing.")

if __name__ == "__main__":
    run()