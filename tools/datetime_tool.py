"""
DateTime tool — gives the model awareness of the current date and time.
"""

import time
from datetime import datetime, timezone


def now() -> dict:
    """Return current date, time, timezone, and unix timestamp."""
    dt = datetime.now()
    utc = datetime.now(timezone.utc)
    return {
        "local_time": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "utc_time": utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "timezone": time.tzname[0],
        "day_of_week": dt.strftime("%A"),
        "unix_timestamp": int(time.time()),
    }
