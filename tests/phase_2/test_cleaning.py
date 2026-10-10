"""Unit tests for Phase 2: Data Cleaning."""

import pytest
from data_preprocessing.cleaning.attributes import (
    parse_business_attributes,
    parse_business_hours,
    extract_primary_category
)
from data_preprocessing.cleaning.credibility import calculate_elite_years


def test_parse_business_attributes():
    raw = {
        "RestaurantsPriceRange2": "2",
        "NoiseLevel": "u'average'",
        "Alcohol": "'beer_and_wine'",
        "OutdoorSeating": "True",
        "BusinessParking": "{u'valet': False, u'garage': False, u'street': True, u'lot': False}"
    }
    parsed = parse_business_attributes(raw)
    assert parsed["price_range"] == 2
    assert parsed["noise_level"] == "average"
    assert parsed["alcohol"] == "beer_and_wine"
    assert parsed["outdoor_seating"] is True
    assert parsed["parking_street"] is True
    assert parsed["parking_valet"] is False


def test_parse_business_hours():
    raw = {
        "Monday": "11:0-22:0",
        "Tuesday": "0:0-0:0",
        "Wednesday": "11:0-23:0"
    }
    hours = parse_business_hours(raw)
    assert hours["mon_open"] == "11:0"
    assert hours["mon_close"] == "22:0"
    assert hours["tue_open"] is None
    assert hours["tue_close"] is None
    assert hours["wed_open"] == "11:0"
    assert hours["wed_close"] == "23:0"


def test_extract_primary_category():
    cats = "Restaurants, Italian, Pizza, Bars"
    primary = extract_primary_category(cats)
    assert primary in ("Italian", "Pizza")


def test_calculate_elite_years():
    assert calculate_elite_years("2012,2013,2014") == 3
    assert calculate_elite_years("20,20") == 1
    assert calculate_elite_years(None) == 0
    assert calculate_elite_years("") == 0
