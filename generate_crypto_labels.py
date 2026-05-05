"""
generate_crypto_labels.py
Generate labels for crypto training data with enhanced features:
- Lagged features + garch_vol + future_return
- Time features (hour_of_day, day_of_week)
- Volume ratio (current vs 20-period avg)
- Regime detection (ADX-based trending/ranging)
- Spread-aware labeling (deducts spread from returns)
- Time-decay sample weights for training
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from textblob import TextBlob

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

os.makedirs(config['log_path'],        exist_ok=True)
os.makedirs(config['data_output_path'], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config['log_path'], 'generate_crypto_labels.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

BASE_FEATURES = [
    'atr', 'ema_diff', 'rsi', 'macd_diff', 'vwap', 'price_vwap_diff',
    'bb_position', 'roc', 'stochastic', 'cci', 'williams', 'momentum',
    'realized_vol', 'chaikin_vol', 'adx', 'rvi', 'obv', 'volume_delta',
    'ad_line', 'vol_osc', 'supertrend', 'hma', 'ichimoku_tenkan', 'sar',
    'dpo', 'spread', 'sentiment'
]

FEATURE_DEFAULTS = {
    'atr': 0.0001, 'ema_diff': 0.0, 'rsi': 50.0, 'macd_diff': 0.0,
    'price_vwap_diff': 0.0, 'bb_position': 0.5, 'roc': 0.0, 'stochastic': 50.0,
    'cci': 0.0, 'williams': -50.0, 'momentum': 0.0,
    'realized_vol': 0.0, 'chaikin_vol': 0.0, 'adx': 25.0, 'rvi': 0.0,
    'obv': 0.0, 'volume_delta': 0.0, 'ad_line': 0.0, 'vol_osc': 0.0,
    'supertrend': 0.0, 'dpo': 0.0, 'spread': 2.0, 'sentiment': 0.0,
    'vwap': None, 'hma': None, 'ichimoku_tenkan': None, 'sar': None,
}

SENTIMENT_MAP = {
    "BTCUSD": "Bitcoin showing strong institutional buying",
    "ETHUSD": "Ethereum neutral consolidation",
    "XRPUSD": "XRP regulatory clarity improving",
}


def get_sentiment(symbol):
    text = SENTIMENT_MAP.get(str(symbol).strip(), "Neutral")
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0


def compute_derived_features(df):
    """Compute lagged features, garch_vol, future_return, time features, volume ratio, regime."""
    group_cols = ['symbol', 'timeframe'] if 'timeframe' in df.columns else ['symbol']
    logging.info(f"Computing derived features grouped by {group_cols}...")

    df = df.sort_values(group_cols + (['time'] if 'time' in df.columns else [])).reset_index(drop=True)

    # -- GARCH volatility --
    df['log_return'] = df.groupby(group_cols)['price'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df['garch_vol'] = df.groupby(group_cols)['log_return'].transform(
        lambda x: x.rolling(window=20, min_periods=5).std()
    )
    df['garch_vol'] = df['garch_vol'].fillna(0.0)

    # -- Future return --
    df['future_return'] = df.groupby(group_cols)['price'].transform(
        lambda x: x.pct_change(periods=1).shift(-1)
    )
    df['future_return'] = df['future_return'].fillna(0.0)

    # -- Lagged features --
    lag_sources = ['price', 'rsi', 'macd_diff', 'atr']
    for col in lag_sources:
        for lag in [1, 2, 3]:
            col_name = f'{col}_lag{lag}'
            df[col_name] = df.groupby(group_cols)[col].shift(lag)
            df[col_name] = df[col_name].fillna(df[col] if col != 'price' else df['price'])

    # -- Time features (hour_of_day, day_of_week) --
    if 'time' in df.columns:
        df['time_parsed'] = pd.to_datetime(df['time'], errors='coerce')
        df['hour_of_day'] = df['time_parsed'].dt.hour.fillna(12).astype(float)
        df['day_of_week'] = df['time_parsed'].dt.dayofweek.fillna(2).astype(float)
        df.drop(columns=['time_parsed'], inplace=True, errors='ignore')
    else:
        df['hour_of_day'] = 12.0
        df['day_of_week'] = 2.0

    # -- Volume ratio (current volume_delta vs rolling avg) --
    # Using abs(volume_delta) as proxy for volume activity
    df['vol_abs'] = df['volume_delta'].abs()
    df['volume_ratio'] = df.groupby(group_cols)['vol_abs'].transform(
        lambda x: x / x.rolling(window=20, min_periods=5).mean()
    )
    df['volume_ratio'] = df['volume_ratio'].fillna(1.0).clip(0, 10)
    df.drop(columns=['vol_abs'], inplace=True, errors='ignore')

    # -- Regime detection (ADX-based) --
    # 0 = ranging (ADX < 20), 1 = weak trend (20-30), 2 = strong trend (>30)
    df['regime'] = np.select(
        [df['adx'] < 20, df['adx'] < 30],
        [0.0, 1.0],
        default=2.0
    )

    df.drop(columns=['log_return'], inplace=True, errors='ignore')

    new_cols = ['garch_vol', 'future_return',
                'hour_of_day', 'day_of_week', 'volume_ratio', 'regime'] + \
               [f'{c}_lag{l}' for c in lag_sources for l in [1, 2, 3]]
    logging.info(f"Added {len(new_cols)} derived features: {new_cols}")
    return df


def generate_labels(df, threshold=0.002, look_ahead=5):
    """Generate labels with spread-aware thresholds."""
    df = df.copy()
    group_cols = ['symbol', 'timeframe'] if 'timeframe' in df.columns else ['symbol']
    df['future_price'] = df.groupby(group_cols)['price'].shift(-look_ahead)
    df['price_change'] = (df['future_price'] - df['price']) / df['price']

    # Spread-aware: deduct spread cost from return before labeling
    # spread is in points, convert to pct of price
    df['spread_pct'] = df['spread'] / df['price']
    df['net_return'] = df['price_change'] - df['spread_pct']

    df['label'] = np.select(
        [df['net_return'] > threshold, df['net_return'] < -threshold],
        [1, -1],
        default=0
    )
    df = df.dropna(subset=['future_price', 'price_change'])
    df.drop(columns=['spread_pct', 'net_return'], inplace=True, errors='ignore')
    logging.info(f"Label distribution (spread-aware): {df['label'].value_counts().to_dict()}")
    return df


def compute_sample_weights(df):
    """Compute time-decay weights: recent data weighted higher.
    Last 2 years = 1.0, 2-5 years = 0.5, 5+ years = 0.25
    """
    if 'time' not in df.columns:
        df['sample_weight'] = 1.0
        return df

    df['time_dt'] = pd.to_datetime(df['time'], errors='coerce')
    latest = df['time_dt'].max()
    years_ago = (latest - df['time_dt']).dt.days / 365.25

    df['sample_weight'] = np.select(
        [years_ago <= 2, years_ago <= 5],
        [1.0, 0.5],
        default=0.25
    )
    df.drop(columns=['time_dt'], inplace=True, errors='ignore')
    logging.info(f"Sample weights: 1.0={int((df['sample_weight']==1.0).sum())}, "
                 f"0.5={int((df['sample_weight']==0.5).sum())}, "
                 f"0.25={int((df['sample_weight']==0.25).sum())}")
    return df


def main():
    input_path    = os.path.join(config['data_path'],        'FXJEFE_Crypto_Features.csv')
    fixed_path    = os.path.join(config['data_output_path'], 'FXJEFE_Crypto_Features_fixed.csv')
    labeled_path  = os.path.join(config['data_output_path'], 'FXJEFE_Crypto_Features_with_labels.csv')
    training_path = os.path.join(config['data_output_path'], 'crypto_training_data.csv')

    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path, encoding='utf-8-sig', low_memory=False)
    logging.info(f"Read {len(df)} rows.  Columns: {list(df.columns)}")

    all_cols = ['time', 'symbol', 'price'] + BASE_FEATURES + ['signal']
    for col in all_cols:
        if col not in df.columns:
            df[col] = '' if col in ('time', 'symbol', 'signal') else 0.0
            logging.info(f"Added missing column: {col}")

    df['price'] = pd.to_numeric(df['price'], errors='coerce').ffill()

    for col in BASE_FEATURES:
        default = FEATURE_DEFAULTS.get(col)
        if default is None:
            default = df['price']
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)

    df['sentiment'] = df['symbol'].apply(get_sentiment)

    # Compute all derived features
    df = compute_derived_features(df)

    # Compute time-decay sample weights
    df = compute_sample_weights(df)

    nan_after = df[['price'] + BASE_FEATURES].isna().sum()
    if nan_after.any():
        logging.warning(f"Remaining NaNs:\n{nan_after[nan_after > 0]}")

    df.to_csv(fixed_path, encoding='utf-8', index=False)
    logging.info(f"Saved cleaned CSV  -> {fixed_path}")

    # Spread-aware labeling
    df = generate_labels(df)
    df.to_csv(labeled_path, encoding='utf-8', index=False)
    logging.info(f"Saved labeled CSV  -> {labeled_path}")

    # Training CSV: features + label + sample_weight
    train_cols = config['features'] + ['label', 'sample_weight']
    missing = [c for c in train_cols if c not in df.columns]
    if missing:
        logging.error(f"Missing training columns: {missing}")
        raise ValueError(f"Missing columns: {missing}")
    df[train_cols].dropna().to_csv(training_path, encoding='utf-8', index=False)
    logging.info(f"Saved training CSV -> {training_path}  ({len(df)} rows)")

if __name__ == '__main__':
    main()
