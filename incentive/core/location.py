"""Best-effort public datacenter location detection."""

from __future__ import annotations

import json
from typing import Any
from urllib.request import Request, urlopen


_PUBLIC_LOCATION_CACHE: dict[str, Any] | None = None


def _normalise_public_location(payload: dict[str, Any]) -> dict[str, Any] | None:
    lat = payload.get("latitude", payload.get("lat"))
    lon = payload.get("longitude", payload.get("lon"))
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return None
    if not (-90 <= lat_f <= 90 and -180 <= lon_f <= 180):
        return None

    connection = payload.get("connection") if isinstance(payload.get("connection"), dict) else {}
    provider = payload.get("org") or payload.get("isp") or connection.get("org")
    region = payload.get("region") or payload.get("region_name")
    country = payload.get("country_name") or payload.get("country")
    label = ", ".join(str(item) for item in (region, country) if item) or str(country or "unknown-area")
    location: dict[str, Any] = {
        "lat": round(lat_f, 1),
        "lon": round(lon_f, 1),
        "label": label,
        "precision": "region",
        "source": "public_ip",
    }
    for key, value in {
        "region": region,
        "country": country,
        "provider": provider,
    }.items():
        if value:
            location[key] = str(value)
    return location


def detect_public_location(timeout_sec: float = 1.5) -> dict[str, Any] | None:
    """Return approximate datacenter area from the host's public egress IP."""

    global _PUBLIC_LOCATION_CACHE
    if _PUBLIC_LOCATION_CACHE is not None:
        return dict(_PUBLIC_LOCATION_CACHE) if _PUBLIC_LOCATION_CACHE else None

    for url in ("https://ipapi.co/json/", "https://ipwho.is/"):
        try:
            request = Request(url, headers={"User-Agent": "quasar-incentive/1.0"})
            with urlopen(request, timeout=timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if isinstance(payload, dict) and payload.get("success", True) is not False:
                location = _normalise_public_location(payload)
                if location:
                    _PUBLIC_LOCATION_CACHE = location
                    return dict(location)
        except Exception:
            continue
    _PUBLIC_LOCATION_CACHE = {}
    return None
