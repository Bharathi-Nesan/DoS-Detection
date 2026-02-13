import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def run():
    print("--- Phase 2: Gini Feature Selection ---")
    df = pd.read_csv('cleaned_wsn.csv')
    X = df.drop(columns=['Attack type'])
    y = df['Attack type']

    
    selector = RandomForestClassifier(n_estimators=100, random_state=42)
    selector.fit(X, y)

    importances = pd.Series(selector.feature_importances_, index=X.columns)
    top_16 = importances.sort_values(ascending=False).head(16).index.tolist()

    df[top_16 + ['Attack type']].to_csv('optimized_wsn.csv', index=False)
    print(f"Phase 2 Complete. Features locked: {top_16}")