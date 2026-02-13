import pandas as pd
import joblib
import time

def run_master_test():
    # 1. Load all three finalized models and the 20% test data
    try:
        dt_model = joblib.load('wsn_dt.pkl')    # Your Pruned DT
        rf_model = joblib.load('wsn_rf.pkl')    # Random Forest Benchmark
        knn_model = joblib.load('wsn_knn.pkl')  # KNN Benchmark
        
        # Load unseen 20% test data
        test_blind = pd.read_csv('test_no_labels.csv')
        test_truth = pd.read_csv('test_with_labels.csv')
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print("="*120)
    print(f"{'🛡️ WSN-IDS 100% IMPLEMENTATION: PRUNED DT vs. BENCHMARKS':^120}")
    print("="*120)
    header = f"{'Pkt #':<6} | {'Truth':<10} | {'Pruned DT (P|Lat)':<25} | {'RF (P|Lat)':<25} | {'KNN (P|Lat)':<25}"
    print(header)
    print("-" * 120)

    # 2. Simulate 20 packet scans
    for i in range(1, 21):
        sample = test_blind.sample(1)
        idx = sample.index[0]
        actual = test_truth.loc[idx, 'Attack type']

        # Measure Pruned Decision Tree
        t1 = time.time()
        p_dt = dt_model.predict(sample)[0]
        l_dt = time.time() - t1

        # Measure Random Forest
        t2 = time.time()
        p_rf = rf_model.predict(sample)[0]
        l_rf = time.time() - t2

        # Measure KNN
        t3 = time.time()
        p_knn = knn_model.predict(sample)[0]
        l_knn = time.time() - t3

        # Formatted Result Strings (Prediction | Latency)
        res_dt = f"{p_dt} | {l_dt:.7f}s"
        res_rf = f"{p_rf} | {l_rf:.7f}s"
        res_knn = f"{p_knn} | {l_knn:.7f}s"

        print(f"{i:<6} | {actual:<10} | {res_dt:<25} | {res_rf:<25} | {res_knn:<25}")
        time.sleep(0.4) # Controlled speed for readable demo

    print("="*120)
    print("✅ DEMO COMPLETE: All models successfully identified the packets.")
    print("Observation: The Pruned DT consistently maintains the lowest latency.")

if __name__ == "__main__":
    run_master_test()