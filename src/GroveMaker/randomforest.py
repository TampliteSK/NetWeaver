# randomforest.py

import os
import time
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import QuantileTransformer
from joblib import dump, load
import matplotlib.pyplot as plt 

df = None

def load_data():
    try:
        df = pl.scan_csv("fen_analysis.csv")  # LazyFrame
        # print("Schema:", df.collect_schema())
        fen_eval_df = df.select(["FEN", "Analysis"]).collect()
        # print(f"Row count: {len(fen_eval_df)}")
        return fen_eval_df
    except Exception as e:
        print(f"Caught exception {e} while loading data")
        return None
    
def print_onehot(onehot):
    idx_to_piece = {
        0: 'White pawn (P)',
        1: 'White knight (N)',
        2: 'White bishop (B)',
        3: 'White rook (R)',
        4: 'White queen (Q)',
        5: 'White king (K)',
        6: 'Black pawn (p)',
        7: 'Black knight (n)',
        8: 'Black bishop (b)',
        9: 'Black rook (r)',
        10: 'Black queen (q)',
        11: 'Black king (k)'
    }
    
    for piece_idx in range(12):
        print(f"\n{idx_to_piece[piece_idx]}")
        start = piece_idx * 64
        end = start + 64
        piece_bits = onehot[start:end]
        
        for rank in range(0, 8):
            row_start = rank * 8
            row = piece_bits[row_start : row_start + 8]
            print("    " + " ".join(str(int(x)) for x in row))

# Convert fen to 768-bit input for random forest training
def fen_to_onehot(fen: str):
    onehot = np.zeros(768, dtype=np.uint8)
    piece_dict = {'P':0, 'N':1, 'B':2, 'R':3, 'Q':4, 'K':5,
                  'p':6, 'n':7, 'b':8, 'r':9, 'q':10, 'k':11}
    
    parts = fen.split()
    rows = parts[0].split('/')
    
    square = 0
    for row in rows:
        for c in row:
            if c.isdigit():
                square += int(c)
            else:
                idx = piece_dict[c] * 64 + square
                onehot[idx] = 1.0
                square += 1
    return onehot

def main():
    global df

    if not os.path.exists("fen_analysis.csv"):
        print("ERROR: fen_analysis.csv not found in:", os.getcwd())
        return
    
    checkpoint = time.time()
    df = load_data()
    print(f"Data loaded in {time.time() - checkpoint:.2f}s")
    # print(df)

    checkpoint = time.time()
    X = np.array([fen_to_onehot(fen) for fen in df["FEN"]])  # Input (in 768 onehot encoding)
    y_raw = np.clip(df["Analysis"].to_numpy(), -1000, 1000)
    qt = QuantileTransformer(
        n_quantiles=1000,
        output_distribution='normal',  # Forces Gaussian shape
        random_state=42
    )
    qt.fit(y_raw.reshape(-1, 1))
    y = qt.transform(y_raw.reshape(-1, 1)).flatten()
    
    """
    print(f"Evaluation stats - Min: {y.min():.3f}, Max: {y.max():.3f}, Mean: {y.mean():.3f}")
    plt.hist(y, bins=50)
    plt.title("Normalized Evaluation Distribution")
    plt.show()
    return
    """
    print(f"Data converted in {time.time() - checkpoint:.2f}s")

    # Split data train/test 80/20
    checkpoint = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    print(f"Data split in {time.time() - checkpoint:.2f}s")

    # Train model
    params = {
        'max_depth': 8,                  
        'learning_rate': 0.05,            
        'objective': 'reg:pseudohubererror',             
        'subsample': 0.8,                # Randomly sample 80% of data
        'colsample_bytree': 0.8,         # Randomly sample 80% of features
        'reg_alpha': 0.5,                # L1 regularization
        'reg_lambda': 1.5,               # L2 regularization
        'gamma': 0.1,                    # Minimum loss reduction
        'min_child_weight': 3,
        'tree_method': 'hist',         
        'grow_policy': 'lossguide',
        'max_bin': 64,                   
        'eval_metric': ['rmse', 'mae'],
        'seed': 42
    }

    checkpoint = time.time()
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=300,  # Trees/estimators count
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
        evals_result=evals_result,
        verbose_eval=25
    )
    print(f"Forest trained in {time.time() - checkpoint:.2f}s")

    # Evaluate
    checkpoint = time.time()
    train_pred = model.predict(dtrain)
    test_pred = model.predict(dtest)
    print(f"\nTraining R²: {r2_score(y_train, train_pred):.3f}")
    print(f"Test R²: {r2_score(y_test, test_pred):.3f}")
    print(f"Tests completed in {time.time() - checkpoint:.2f}s")

    checkpoint = time.time()
    model_name = "XGBforest_100T_D6_20250807.json"
    model.save_model(model_name)  # XGBoost's native format
    print(f"Model saved as {model_name} in {time.time() - checkpoint:.2f}s")

main()