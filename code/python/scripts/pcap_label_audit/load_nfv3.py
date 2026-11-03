"""Load an NF-v3 day CSV with timestamps, original and corrected labels, and duplicate counts."""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nfv3_relabel import relabel_chunk

KEY = ['FLOW_START_MILLISECONDS', 'IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT', 'PROTOCOL']

def load(path):
    df = pd.read_csv(path)
    df['ts'] = pd.to_datetime(df.FLOW_START_MILLISECONDS, unit='ms')
    df = relabel_chunk(df)
    df['ncopy'] = df.groupby(KEY).Label.transform('size')
    return df
