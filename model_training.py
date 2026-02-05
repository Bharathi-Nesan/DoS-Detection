import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score

def run():
    print("="*85)
    print(f"{'PHASE 4: 10-FOLD CV & LATENCY ANALYSIS':^85}")
    print("="*85)
    
    # Load Split Data
    try:
        train = pd.read_csv('train_data.csv')
        test = pd.read_csv('test_data.csv')
    except FileNotFoundError:
        print("Error: train_data.csv not found! Please run Phase 3.")
        return

    X_train, y_train = train.drop(columns=['Attack type']), train['Attack type']
    X_test, y_test = test.drop(columns=['Attack type']), test['Attack type']

    # Define Models
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
        # 1. 10-Fold Cross-Validation on Training Data
        cv_scores = cross_val_score(model, X_train, y_train, cv=kf, n_jobs=-1)
        mean_cv = cv_scores.mean() * 100
        
        # 2. Final Training
        model.fit(X_train, y_train)
        
        # 3. Latency Measurement (Testing Phase)
        start_time = time.time()
        y_pred = model.predict(X_test)
        end_time = time.time() - start_time
        
        # Calculate per-packet latency
        latency_per_packet = end_time / len(X_test)
        test_acc = accuracy_score(y_test, y_pred) * 100

        # Print Row
        print(f"{name:<22} | {mean_cv:>10.2f}% | {test_acc:>8.2f}% | {latency_per_packet:.8f}s")

    print("=" * len(header))
    print("[SUCCESS]: All models evaluated with 10-Fold CV and Latency metrics.")

if __name__ == "__main__":
    run()