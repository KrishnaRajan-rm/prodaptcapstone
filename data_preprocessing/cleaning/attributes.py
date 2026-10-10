"""Phase 2: Business attributes and hours parser."""

import ast
import json
import re
from typing import Dict, Any, Optional, Tuple


def safe_parse_literal(val: Any) -> Any:
    """Safely parses stringified Python literal dicts or returns default."""
    if not val or not isinstance(val, str):
        return {}
    val_clean = val.strip()
    if not val_clean or val_clean == "None":
        return {}
    try:
        return ast.literal_eval(val_clean)
    except Exception:
        # Fallback regex for common key: value patterns
        return {}


def parse_business_attributes(raw_attrs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extracts typed attribute fields from Yelp's raw business attributes dictionary."""
    if not raw_attrs:
        return {
            "price_range": None,
            "noise_level": None,
            "alcohol": None,
            "wifi": None,
            "outdoor_seating": None,
            "good_for_kids": None,
            "has_tv": None,
            "reservations": None,
            "delivery": None,
            "takeout": None,
            "parking_garage": False,
            "parking_street": False,
            "parking_lot": False,
            "parking_valet": False
        }

    def clean_str_attr(val: Any) -> Optional[str]:
        if not val or str(val).strip() in ("None", ""):
            return None
        s = str(val).strip().strip("'\"")
        if s.startswith("u'") or s.startswith('u"'):
            s = s[2:].rstrip("'\"")
        s = s.strip("'\"").lower()
        return s if s and s != "none" else None

    # Price range
    price = raw_attrs.get("RestaurantsPriceRange2")
    price_val = None
    if price is not None:
        try:
            p_clean = clean_str_attr(price)
            if p_clean and p_clean.isdigit():
                p_int = int(p_clean)
                if 1 <= p_int <= 4:
                    price_val = p_int
        except (ValueError, TypeError):
            pass

    # Noise level, alcohol, wifi
    noise_val = clean_str_attr(raw_attrs.get("NoiseLevel"))
    alcohol_val = clean_str_attr(raw_attrs.get("Alcohol"))
    wifi_val = clean_str_attr(raw_attrs.get("WiFi"))

    # Booleans helper
    def parse_bool(key: str) -> Optional[bool]:
        v = raw_attrs.get(key)
        if v is None:
            return None
        v_str = str(v).strip("'\"").lower()
        if v_str in ("true", "1"):
            return True
        if v_str in ("false", "0"):
            return False
        return None

    # Parking dictionary flattening
    parking_raw = raw_attrs.get("BusinessParking")
    parking_dict = {}
    if isinstance(parking_raw, str):
        parking_dict = safe_parse_literal(parking_raw)
    elif isinstance(parking_raw, dict):
        parking_dict = parking_raw

    return {
        "price_range": price_val,
        "noise_level": noise_val,
        "alcohol": alcohol_val,
        "wifi": wifi_val,
        "outdoor_seating": parse_bool("OutdoorSeating"),
        "good_for_kids": parse_bool("GoodForKids"),
        "has_tv": parse_bool("HasTV"),
        "reservations": parse_bool("RestaurantsReservations"),
        "delivery": parse_bool("RestaurantsDelivery"),
        "takeout": parse_bool("RestaurantsTakeOut"),
        "parking_garage": bool(parking_dict.get("garage", False)),
        "parking_street": bool(parking_dict.get("street", False)),
        "parking_lot": bool(parking_dict.get("lot", False)),
        "parking_valet": bool(parking_dict.get("valet", False))
    }


def parse_business_hours(raw_hours: Optional[Dict[str, str]]) -> Dict[str, Optional[str]]:
    """Extracts open/close times for each of the 7 days. '0:0-0:0' indicates closed."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_abbrevs = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    result = {}

    if not raw_hours or not isinstance(raw_hours, dict):
        for abbrev in day_abbrevs:
            result[f"{abbrev}_open"] = None
            result[f"{abbrev}_close"] = None
        return result

    for day, abbrev in zip(days, day_abbrevs):
        val = raw_hours.get(day)
        if val and isinstance(val, str) and val.strip() not in ("0:0-0:0", "None", ""):
            parts = val.split("-")
            if len(parts) == 2:
                result[f"{abbrev}_open"] = parts[0].strip()
                result[f"{abbrev}_close"] = parts[1].strip()
                continue
        result[f"{abbrev}_open"] = None
        result[f"{abbrev}_close"] = None

    return result


def extract_primary_category(categories_str: Optional[str]) -> str:
    """Extracts a primary cuisine/restaurant category from comma-separated list."""
    if not categories_str:
        return "Restaurant"
    cats = [c.strip() for c in categories_str.split(",") if c.strip()]
    if not cats:
        return "Restaurant"
    # Skip generic terms if a more specific cuisine exists
    generic = {"Restaurants", "Food", "Nightlife", "Bars"}
    for c in cats:
        if c not in generic:
            return c
    return cats[0]
