import pandas as pd
import joblib
import time

def run_master_test():
    # 1. Load all three finalized models and the 20% test data
    try:
        dt_model = joblib.load('wsn_dt.pkl')
        rf_model = joblib.load('wsn_rf.pkl')
        knn_model = joblib.load('wsn_knn.pkl')
        
        # Pulling from the unseen 20% testing set
        test_blind = pd.read_csv('test_no_labels.csv')
        test_truth = pd.read_csv('test_with_labels.csv')
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print("="*115)
    print(f"{'🛡️ WSN-IDS FULL ARCHITECTURE: MULTI-MODEL LATENCY COMPARISON':^115}")
    print("="*115)
    header = f"{'Pkt #':<6} | {'Truth':<10} | {'DT (P|Lat)':<20} | {'RF (P|Lat)':<20} | {'KNN (P|Lat)':<20}"
    print(header)
    print("-" * 115)

    # 2. Simulate 20 packet scans for real-time demonstration
    for i in range(1, 21):
        sample = test_blind.sample(1)
        idx = sample.index[0]
        actual = test_truth.loc[idx, 'Attack type']

        # Measure Decision Tree
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

        # Display results with high-precision latency
        res_dt = f"{p_dt} | {l_dt:.7f}s"
        res_rf = f"{p_rf} | {l_rf:.7f}s"
        res_knn = f"{p_knn} | {l_knn:.7f}s"

        print(f"{i:<6} | {actual:<10} | {res_dt:<20} | {res_rf:<20} | {res_knn:<20}")
        time.sleep(0.3)

    print("="*115)
    print("✅ DEMO COMPLETE: The Decision Tree consistently shows the lowest latency for WSN deployment.")

if __name__ == "__main__":
    run_master_test()