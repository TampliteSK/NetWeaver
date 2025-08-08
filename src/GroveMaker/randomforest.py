# randomforest.py

import os
import time
from datetime import date

import chess
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import QuantileTransformer
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

def fen_to_material_array(fen: str):
    # Material values as provided
    material = {'P':1, 'N':3, 'B':3, 'R':5, 'Q':9, 'K':0,
                'p':-1, 'n':-3, 'b':-3, 'r':-5, 'q':-9, 'k':0}
    
    # Initialize 64-element array for board squares (a1=0, a2=8, ..., h8=63)
    encoding = np.zeros(64 + 6, dtype=np.float32)  # +6 for additional features
    
    # Parse FEN
    board = chess.Board(fen)
    
    # Fill board squares with material values
    for square, piece in board.piece_map().items():
        encoding[square] = material[piece.symbol()]
    
    # Side to move
    encoding[64] = 1 if board.turn == chess.WHITE else -1
    
    return encoding


def normalise_centipawns(arr):
    clipped = np.clip(arr, -1500, 1500)
    arr = (arr + 1500) / 3000
    return arr

def main():
    global df

    if not os.path.exists("fen_analysis.csv"):
        print("ERROR: fen_analysis.csv not found in:", os.getcwd())
        return
    
    checkpoint = time.time()
    df = load_data()
    print(f"Data loaded with {len(df)} entries in {time.time() - checkpoint:.2f}s")
    # print(df)

    checkpoint = time.time()
    X = np.array([fen_to_material_array(fen) for fen in df["FEN"]])  # Input (in 768 onehot encoding)
    y_raw = df["Analysis"].to_numpy()
    y = normalise_centipawns(y_raw)
    # y = y_raw
    
    """
    print(f"Evaluation stats - Min: {y.min():.3f}, Max: {y.max():.3f}, Mean: {y.mean():.3f}, STDEV: {y.std():.3f}")
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
    TREES = 300
    DEPTH = 20
    LR = 0.2
    params = {
        'max_depth': DEPTH,                  
        'learning_rate': LR,            
        'objective': 'reg:squarederror',             
        'subsample': 0.6,                # Randomly sample 80% of data
        'colsample_bytree': 0.6,         # Randomly sample 80% of features
        'reg_alpha': 3,                # L1 regularization
        'reg_lambda': 10,               # L2 regularization
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
        num_boost_round=TREES, # Trees/estimators count
        evals=[(dtrain, 'train'), (dtest, 'test')],
        # early_stopping_rounds=int(TREES * 0.3),  # Stop if no improvement after x rounds
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

    xgb.plot_importance(model, importance_type='gain', max_num_features=20)
    plt.tight_layout()
    plt.show()

    checkpoint = time.time()
    fdate = date.today().strftime("%Y%m%d")
    model_name = f"XGBforest_{TREES}T_D{DEPTH}_LR{LR}_{fdate}.json"
    model.save_model(model_name)  # XGBoost's native format
    print(f"Model saved as {model_name} in {time.time() - checkpoint:.2f}s")

main()