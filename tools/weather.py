"""
Weather tool — current weather via Open-Meteo (free, no API key).

Uses geocoding to resolve city names to coordinates, then fetches
current conditions from Open-Meteo's free API.
"""

import requests

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT = 8


def get_weather(location: str) -> dict:
    """
    Get current weather for a location (city name or coordinates).

    Returns:
        {
            "location": str,
            "temperature_c": float,
            "temperature_f": float,
            "conditions": str,
            "humidity": int,
            "wind_speed_kmh": float,
            "error": str | None,
        }
    """
    # Geocode the location name
    try:
        geo = requests.get(
            _GEOCODE_URL,
            params={"name": location, "count": 1, "language": "en"},
            timeout=_TIMEOUT,
        ).json()
    except Exception as e:
        return {"location": location, "error": f"Geocoding failed: {e}"}

    results = geo.get("results")
    if not results:
        return {"location": location, "error": f"Location '{location}' not found"}

    place = results[0]
    lat, lon = place["latitude"], place["longitude"]
    resolved = f"{place.get('name', location)}, {place.get('country', '')}"

    # Fetch current weather
    try:
        weather = requests.get(
            _WEATHER_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                "temperature_unit": "celsius",
                "wind_speed_unit": "kmh",
            },
            timeout=_TIMEOUT,
        ).json()
    except Exception as e:
        return {"location": resolved, "error": f"Weather fetch failed: {e}"}

    current = weather.get("current", {})
    temp_c = current.get("temperature_2m")
    if temp_c is None:
        return {"location": resolved, "error": "No weather data available"}

    return {
        "location": resolved,
        "temperature_c": temp_c,
        "temperature_f": round(temp_c * 9 / 5 + 32, 1),
        "conditions": _weather_code_to_text(current.get("weather_code", -1)),
        "humidity": current.get("relative_humidity_2m"),
        "wind_speed_kmh": current.get("wind_speed_10m"),
        "error": None,
    }


def _weather_code_to_text(code: int) -> str:
    """Convert WMO weather code to human-readable text."""
    codes = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Foggy", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        95: "Thunderstorm", 96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    return codes.get(code, f"Unknown (code {code})")
