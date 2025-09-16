import math
import pandas as pd

def fmt_price(v):
    if v is None or (isinstance(v,(int,float)) and (not math.isfinite(v) or v==0)):
        return "—"
    try:
        return f"{float(v):.6f}".rstrip("0").rstrip(".")
    except Exception:
        return str(v)

def to_ohlcv_df(raw):
    """
    raw: [ [ts, open, high, low, close, vol], ... ]
    """
    df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df
