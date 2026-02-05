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
        # Use BOTH new files created in Phase 3
        test_blind = pd.read_csv('test_no_labels.csv') # 16 columns (Real world)
        test_truth = pd.read_csv('test_with_labels.csv') # 17 columns (Ground Truth)
    except FileNotFoundError:
        print("Error: Required CSV files not found! Please run Phase 3.")
        return

    # 2. Prepare Features and Targets
    X_train, y_train = train.drop(columns=['Attack type']), train['Attack type']
    
    # X_test comes from the blind file (no labels)
    X_test = test_blind 
    # y_test comes from the truth file (for validation)
    y_test = test_truth['Attack type']

    # 3. Define Models
    models = {
        "Pruned Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5)
    }

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Table Header
    header = f"{'Model':<22} | {'CV Mean Acc':<12} | {'Test Acc':<10} | {'Latency/Pkt':<12}"
    print(header)
    print("-" * len(header))

    for name, model in models.items():
        # A. 10-Fold Cross-Validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=kf, n_jobs=-1)
        mean_cv = cv_scores.mean() * 100
        
        # B. Final Training
        model.fit(X_train, y_train)
        
        # C. Latency Measurement (Using the blind 16-column data)
        start_time = time.time()
        y_pred = model.predict(X_test)
        end_time = time.time() - start_time
        
        # D. Evaluation (Comparing prediction to the truth file)
        latency_per_packet = end_time / len(X_test)
        test_acc = accuracy_score(y_test, y_pred) * 100

        # Print Row
        print(f"{name:<22} | {mean_cv:>10.2f}% | {test_acc:>8.2f}% | {latency_per_packet:.8f}s")
        
        # Save the best model (Decision Tree for your dashboard)
        if name == "Pruned Decision Tree":
            joblib.dump(model, 'wsn_dos_model.pkl')

    print("=" * len(header))
    print("[SUCCESS]: Models evaluated and 'wsn_dos_model.pkl' is saved.")

if __name__ == "__main__":
    run()