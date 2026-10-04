# utils/helpers.py

def safe_float(val, default=0.0):
    if val is None or val == "" or val == "-":
        return default
    try:
        return float(str(val).replace(",", ".").replace("€", "").strip())
    except (ValueError, TypeError):
        return default