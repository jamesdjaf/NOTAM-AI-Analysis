import math
import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Optional

import pandas as pd
import streamlit as st
from shapely.geometry import LineString, Point, Polygon


UTC_DATE = datetime.utcnow().date()


@dataclass
class NotamRestriction:
    raw_text: str
    restriction_type: str
    airway: Optional[str]
    segment_start: Optional[str]
    segment_end: Optional[str]
    fl_min: Optional[int]
    fl_max: Optional[int]
    fl_text: str
    active_start: Optional[datetime]
    active_end: Optional[datetime]
    polygon: Optional[Polygon]
    polygon_points: list[tuple[float, float]]
    reference: str = "NOTAM"
    radius_center: Optional[tuple[float, float]] = None
    radius_nm: Optional[float] = None
    fir: str = ""
    mentioned_waypoints: Optional[list[str]] = None
    mentioned_airways: Optional[list[str]] = None
    schedule_text: str = ""
    fields: Optional[dict] = None
    start_utc_text: str = ""
    end_utc_text: str = ""
    parsing_status: str = "Parsed Successfully"
    review_reason: str = ""
    q_code: str = ""
    traffic: str = ""
    purpose: str = ""
    scope: str = ""
    geometry_type: str = "TEXT_ONLY"
    q_reference: str = ""
    q_reference_center: Optional[tuple[float, float]] = None
    q_radius_nm: Optional[float] = None


@dataclass
class RouteSuggestionResult:
    available: bool
    message: str
    candidates: list[dict]


class RoutingDatabaseProvider:
    name = "RoutingDatabaseProvider"

    def is_loaded(self) -> bool:
        return False

    def suggest_ats_routes(self, affected_result: dict, route_df: pd.DataFrame) -> RouteSuggestionResult:
        return RouteSuggestionResult(
            available=False,
            message=(
                "Route alternatives unavailable — no validated airway/AIP/AIRAC database loaded. "
                "Use LIDO to calculate and validate any reroute."
            ),
            candidates=[],
        )


class NullRoutingDatabase(RoutingDatabaseProvider):
    name = "No routing database"


class RouteEngine:
    def __init__(self, provider: Optional[RoutingDatabaseProvider] = None):
        self.provider = provider or NullRoutingDatabase()

    def suggest_alternatives(self, analyses: list[dict], route_df: pd.DataFrame) -> RouteSuggestionResult:
        if not self.provider.is_loaded():
            return RouteSuggestionResult(
                available=False,
                message=(
                    "Route alternatives unavailable — no validated airway/AIP/AIRAC database loaded. "
                    "Use LIDO to calculate and validate any reroute."
                ),
                candidates=[],
            )
        candidates = []
        for analysis in analyses:
            if analysis["result"].get("impacted"):
                candidates.extend(self.provider.suggest_ats_routes(analysis["result"], route_df).candidates)
        return RouteSuggestionResult(available=bool(candidates), message="", candidates=candidates)


def parse_hhmm(value: str, base_date=UTC_DATE) -> Optional[datetime]:
    match = re.search(r"\b([01]\d|2[0-3])([0-5]\d)\b", value.strip())
    if not match:
        return None
    return datetime.combine(base_date, time(int(match.group(1)), int(match.group(2))))


def normalize_time_window(start_hhmm: str, end_hhmm: str, base_date=UTC_DATE) -> tuple[datetime, datetime]:
    start = parse_hhmm(start_hhmm, base_date)
    end = parse_hhmm(end_hhmm, base_date)
    if end <= start:
        end += timedelta(days=1)
    return start, end


def parse_fl_range(text: str) -> tuple[Optional[int], Optional[int], str]:
    upper = text.upper()
    if re.search(r"\bSFC\s*[-/]\s*UNL\b", upper):
        return 0, 999, "SFC-UNL"

    range_match = re.search(r"\bFL\s*(\d{2,3})\s*[-/]\s*FL?\s*(\d{2,3})\b", upper)
    if range_match:
        low, high = int(range_match.group(1)), int(range_match.group(2))
        return min(low, high), max(low, high), f"FL{min(low, high):03d}-FL{max(low, high):03d}"

    above_match = re.search(r"\bFL\s*(\d{2,3})\s*(?:AND\s+ABOVE|\+)(?!\w)", upper)
    if above_match:
        fl = int(above_match.group(1))
        return fl, 999, f"FL{fl:03d} AND ABOVE"

    below_match = re.search(r"\bFL\s*(\d{2,3})\s+AND\s+BELOW\b", upper)
    if below_match:
        fl = int(below_match.group(1))
        return 0, fl, f"FL{fl:03d} AND BELOW"

    single_match = re.search(r"\bFL\s*(\d{2,3})\b", upper)
    if single_match:
        fl = int(single_match.group(1))
        return fl, fl, f"FL{fl:03d}"

    return None, None, "Not specified"


def parse_coord_pair(lat_token: str, lon_token: str) -> Optional[tuple[float, float]]:
    def convert(token: str, is_lat: bool) -> Optional[float]:
        clean = re.sub(r"[^0-9.NSEW]", "", token.upper())
        direction_match = re.search(r"[NSEW]", clean)
        if not direction_match:
            return None
        direction = direction_match.group(0)
        digits = re.sub(r"[NSEW]", "", clean)
        deg_len = 2 if is_lat else 3
        if len(digits) < deg_len + 2:
            return None
        degrees = int(digits[:deg_len])
        rest = digits[deg_len:]
        if "." in rest:
            minutes = float(rest)
            seconds = 0
        elif len(rest) >= 4:
            minutes = int(rest[:2])
            seconds = int(rest[2:4])
        else:
            minutes = int(rest[:2])
            seconds = 0
        decimal = degrees + minutes / 60 + seconds / 3600
        if direction in {"S", "W"}:
            decimal *= -1
        return decimal

    lat = convert(lat_token, True)
    lon = convert(lon_token, False)
    if lat is None or lon is None:
        return None
    return lon, lat


def extract_coordinate_pairs(text: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    seen = set()
    patterns = [
        r"\b([NS]\s*\d{4}(?:\.\d+)?|[NS]\s*\d{6})\s*[-,/ ]*\s*([EW]\s*\d{5}(?:\.\d+)?|[EW]\s*\d{7})\b",
        r"\b(\d{4}(?:\.\d+)?\s*[NS]|\d{6}\s*[NS])\s*[-,/ ]*\s*(\d{5}(?:\.\d+)?\s*[EW]|\d{7}\s*[EW])\b",
    ]
    for pattern in patterns:
        for first, second in re.findall(pattern, text, flags=re.IGNORECASE):
            point = parse_coord_pair(first, second)
            if point and point not in seen:
                seen.add(point)
                points.append(point)
    return points


def parse_fl_restriction(text: str) -> tuple[Optional[int], Optional[int], str]:
    return parse_fl_range(text)


def parse_notam_fields(notam_text: str) -> dict:
    fields = {letter: "" for letter in "QABCDEFG"}
    for match in re.finditer(r"(?ims)^\s*([A-G])\)\s*(.*?)(?=^\s*[A-G]\)\s*|\Z)", notam_text.strip()):
        fields[match.group(1).upper()] = match.group(2).strip()
    q_match = re.search(r"(?im)^\s*Q\)\s*(.+)$", notam_text)
    if q_match:
        fields["Q"] = q_match.group(1).strip()
    return fields


def parse_q_reference(value: str) -> tuple[Optional[tuple[float, float]], Optional[float], str]:
    match = re.search(r"\b(\d{2})(\d{2})([NS])(\d{3})(\d{2})([EW])(\d{3})\b", value or "")
    if not match:
        return None, None, ""
    lat = int(match.group(1)) + int(match.group(2)) / 60
    lon = int(match.group(4)) + int(match.group(5)) / 60
    if match.group(3) == "S":
        lat *= -1
    if match.group(6) == "W":
        lon *= -1
    radius = float(int(match.group(7)))
    return (lon, lat), radius, match.group(0)


def parse_q_line(q_line: str) -> dict:
    if not q_line:
        return {}
    parts = [part.strip() for part in q_line.split("/")]
    result = {"raw": q_line}
    if len(parts) >= 1:
        result["fir"] = parts[0]
    if len(parts) >= 2:
        result["q_code"] = parts[1]
    if len(parts) >= 3:
        result["traffic"] = parts[2]
    if len(parts) >= 4:
        result["purpose"] = parts[3]
    if len(parts) >= 5:
        result["scope"] = parts[4]
    if len(parts) >= 6 and parts[5].isdigit():
        result["lower_fl"] = int(parts[5])
    if len(parts) >= 7 and parts[6].isdigit():
        result["upper_fl"] = int(parts[6])
    if len(parts) >= 8:
        center, radius, raw_ref = parse_q_reference(parts[7])
        result["reference_center"] = center
        result["radius_nm"] = radius
        result["reference_raw"] = raw_ref
    return result


def parse_notam_datetime(value: str) -> Optional[datetime]:
    match = re.search(r"\b(\d{2})(\d{2})(\d{2})([01]\d|2[0-3])([0-5]\d)\b", value or "")
    if not match:
        return None
    year = 2000 + int(match.group(1))
    return datetime(year, int(match.group(2)), int(match.group(3)), int(match.group(4)), int(match.group(5)))


def parse_notam_time(fields: dict, text: str) -> tuple[Optional[datetime], Optional[datetime], str, str, str, list[str]]:
    reasons = []
    start_raw = re.search(r"\b\d{10}\b", fields.get("B", ""))
    end_raw = re.search(r"\b\d{10}\b", fields.get("C", ""))
    if start_raw and end_raw:
        start = parse_notam_datetime(start_raw.group(0))
        end = parse_notam_datetime(end_raw.group(0))
        if start and end:
            return start, end, start_raw.group(0), end_raw.group(0), fields.get("D", ""), reasons

    window = re.search(r"\b([01]\d|2[0-3])([0-5]\d)\s*[-/]\s*([01]\d|2[0-3])([0-5]\d)\b", text)
    if window:
        start, end = normalize_time_window(f"{window.group(1)}{window.group(2)}", f"{window.group(3)}{window.group(4)}")
        return start, end, f"{window.group(1)}{window.group(2)}", f"{window.group(3)}{window.group(4)}", fields.get("D", window.group(0)), reasons

    reasons.append("Time not parsed")
    return None, None, "Unknown", "Unknown", fields.get("D", ""), reasons


def fl_token_value(value: str) -> Optional[int]:
    upper = (value or "").upper().strip()
    if upper in {"SFC", "GND"}:
        return 0
    if upper == "UNL":
        return 999
    match = re.search(r"FL\s*(\d{2,3})\b", upper)
    if match:
        return int(match.group(1))
    if upper.isdigit():
        return int(upper)
    return None


def parse_notam_fl(fields: dict, text: str) -> tuple[Optional[int], Optional[int], str, list[str]]:
    reasons = []
    f_value = fl_token_value(fields.get("F", ""))
    g_value = fl_token_value(fields.get("G", ""))
    if f_value is not None or g_value is not None:
        low = f_value if f_value is not None else 0
        high = g_value if g_value is not None else 999
        return low, high, f"{fields.get('F', 'Unknown')}-{fields.get('G', 'Unknown')}", reasons

    upper = text.upper()
    for pattern in [
        r"\b(SFC|GND)\s*[-/]\s*(UNL)\b",
        r"\b(FL\s*\d{2,3})\s*(?:-|/|TO)\s*(FL\s*\d{2,3})\b",
    ]:
        match = re.search(pattern, upper)
        if match:
            low = fl_token_value(match.group(1))
            high = fl_token_value(match.group(2))
            return low, high, f"{match.group(1)}-{match.group(2)}", reasons

    below = re.search(r"\bBELOW\s+FL\s*(\d{2,3})\b", upper)
    if below:
        return 0, int(below.group(1)), f"BELOW FL{int(below.group(1)):03d}", reasons
    above = re.search(r"\bABOVE\s+FL\s*(\d{2,3})\b", upper)
    if above:
        return int(above.group(1)), 999, f"ABOVE FL{int(above.group(1)):03d}", reasons

    reasons.append("FL ambiguous")
    return None, None, "Unknown", reasons


def parse_notam_radius(text: str) -> tuple[Optional[tuple[float, float]], Optional[float], list[str]]:
    reasons = []
    pattern = r"\b(?:WITHIN|WI|RADIUS)\s*(\d+(?:\.\d+)?)\s*NM(?:\s+RADIUS)?(?:\s+OF|\s+CENTERED\s+ON)?\s+(.{0,40})"
    match = re.search(pattern, text.upper(), flags=re.DOTALL)
    if not match:
        return None, None, reasons
    points = extract_coordinate_pairs(match.group(2))
    if not points:
        reasons.append("Circle center not parsed")
        return None, float(match.group(1)), reasons
    return points[0], float(match.group(1)), reasons


NOTAM_WORD_BLOCKLIST = {
    "A", "AN", "AREA", "TEMPORARY", "DANGER", "RESTRICTED", "PROHIBITED", "ESTABLISHED",
    "BOUNDED", "BACK", "START", "VERTICAL", "LIMITS", "AIRCRAFT", "ARE", "FORBIDDEN",
    "FLY", "INTO", "THE", "WITHIN", "RADIUS", "CENTERED", "CENTER", "SFC", "UNL", "GND",
    "FROM", "UNTIL", "DAILY", "ACTIVE", "CLSD", "CLOSED", "NOT", "AVBL", "AVAILABLE",
    "TEMP", "WARNING", "OF", "BY", "AND", "TO", "BTN", "BETWEEN", "VIA", "OVER", "AT",
}


def parse_notam_airway_segment(e_text: str) -> tuple[list[str], list[str], Optional[str], Optional[str], Optional[str]]:
    upper = e_text.upper()
    airways = sorted(set(token for token in re.findall(r"\b[A-Z]{1,3}\d{1,4}[A-Z]?\b", upper) if is_airway_token(token)))
    segment_start = segment_end = airway = None
    patterns = [
        r"\b([A-Z]{1,3}\d{1,4}[A-Z]?)?\s*(?:BTN|BETWEEN)\s+([A-Z][A-Z0-9]{2,5})\s+(?:AND|TO)\s+([A-Z][A-Z0-9]{2,5})\b",
        r"\b(?:FROM)\s+([A-Z][A-Z0-9]{2,5})\s+(?:TO)\s+([A-Z][A-Z0-9]{2,5})\b",
        r"\b([A-Z]{1,3}\d{1,4}[A-Z]?)?\s*([A-Z][A-Z0-9]{2,5})\s*[-/]\s*([A-Z][A-Z0-9]{2,5})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if not match:
            continue
        groups = [group for group in match.groups()]
        if len(groups) == 3:
            airway = groups[0] if groups[0] and is_airway_token(groups[0]) else None
            segment_start, segment_end = groups[1], groups[2]
        else:
            segment_start, segment_end = groups[0], groups[1]
        if segment_start in NOTAM_WORD_BLOCKLIST or segment_end in NOTAM_WORD_BLOCKLIST:
            segment_start = segment_end = airway = None
            continue
        break

    waypoints = set()
    if segment_start and segment_end:
        waypoints.update([segment_start, segment_end])
    proximity = re.findall(r"\b(?:BTN|BETWEEN|FROM|TO|VIA|OVER|AT|OF|AND)\s+([A-Z][A-Z0-9]{2,5})\b", upper)
    waypoints.update(token for token in proximity if token not in NOTAM_WORD_BLOCKLIST and is_waypoint_token(token))
    return airways, sorted(waypoints), segment_start, segment_end, airway


def parse_restriction_type(e_text: str, polygon_points: list, radius_center, segment_start) -> str:
    upper = e_text.upper()
    if "DANGER AREA" in upper:
        return "Danger Area"
    if "PROHIBITED" in upper:
        return "Prohibited Area"
    if "RESTRICTED" in upper:
        return "Restricted Area"
    if radius_center:
        return "Radius"
    if polygon_points or any(term in upper for term in ["BOUNDED BY", "POLYGON", "BACK TO START", "AREA WITHIN"]):
        return "Area"
    if segment_start:
        return "Airway/Segment"
    return "Unknown"


def geometry_type_from_operational_parse(polygon_points: list, radius_center, segment_start, airways: list, waypoints: list, fir: str, fl_min, fl_max, e_text: str) -> str:
    if polygon_points:
        return "POLYGON"
    if radius_center:
        return "CIRCLE"
    if segment_start:
        return "SEGMENT"
    if airways:
        return "AIRWAY"
    if waypoints:
        return "WAYPOINT"
    if fir and re.search(r"\bFIR\b", e_text.upper()):
        return "FIR"
    if fl_min is not None or fl_max is not None:
        return "VERTICAL_ONLY"
    return "TEXT_ONLY"


def parse_notam(notam_text: str) -> NotamRestriction:
    fields = parse_notam_fields(notam_text)
    q_data = parse_q_line(fields.get("Q", ""))
    e_text = fields.get("E", "") or notam_text
    searchable = "\n".join(value for value in fields.values() if value) or notam_text
    upper = searchable.upper()
    review_reasons = []

    ref_match = re.search(r"\b([A-Z]\d{3,5}/\d{2}|[A-Z]\d{3,5})\b", notam_text.upper())
    reference = ref_match.group(1) if ref_match else "NOTAM"
    fir = q_data.get("fir") or (fields.get("A", "").strip().split()[0] if fields.get("A", "").strip() else "")
    if not fir:
        fir_match = re.search(r"\b([A-Z]{4})\s+FIR\b|\bFIR\s+([A-Z]{4})\b", upper)
        fir = next((group for group in fir_match.groups() if group), "") if fir_match else ""

    active_start, active_end, start_text, end_text, schedule_text, time_reasons = parse_notam_time(fields, searchable)
    review_reasons.extend(time_reasons)
    if "lower_fl" in q_data or "upper_fl" in q_data:
        fl_min = q_data.get("lower_fl")
        fl_max = q_data.get("upper_fl")
        fl_text = f"{fl_min if fl_min is not None else 'Unknown'}-{fl_max if fl_max is not None else 'Unknown'}"
    else:
        fl_min, fl_max, fl_text, fl_reasons = parse_notam_fl(fields, searchable)
        review_reasons.extend(fl_reasons)

    coord_points = extract_coordinate_pairs(e_text)
    radius_center, radius_nm, radius_reasons = parse_notam_radius(e_text)
    review_reasons.extend(radius_reasons)
    polygon_terms = bool(
        not radius_center
        and (
            any(term in e_text.upper() for term in ["BOUNDED BY", "POLYGON", "BACK TO START"])
            or ("AREA WITHIN" in e_text.upper() and len(coord_points) >= 3)
        )
    )
    polygon = Polygon(coord_points) if len(coord_points) >= 3 and not radius_center and polygon_terms else None
    if polygon_terms and len(coord_points) < 3:
        review_reasons.append("Coordinates incomplete")

    mentioned_airways, mentioned_waypoints, segment_start, segment_end, airway = parse_notam_airway_segment(e_text)
    restriction_type = parse_restriction_type(e_text, coord_points if polygon_terms else [], radius_center, segment_start)
    if restriction_type == "Unknown":
        review_reasons.append("Restriction type unknown")
    geometry_type = geometry_type_from_operational_parse(
        coord_points if polygon_terms else [],
        radius_center,
        segment_start,
        mentioned_airways,
        mentioned_waypoints,
        fir,
        fl_min,
        fl_max,
        e_text,
    )

    mandatory_missing = []
    if not fir:
        mandatory_missing.append("FIR missing")
    if not active_start or not active_end:
        mandatory_missing.append("Validity time missing")
    if not e_text.strip():
        mandatory_missing.append("E text missing")
    review_reasons.extend(mandatory_missing)

    parsing_status = "Parsed Successfully" if not review_reasons else "Review Required"
    return NotamRestriction(
        raw_text=notam_text,
        restriction_type=restriction_type,
        airway=airway,
        segment_start=segment_start,
        segment_end=segment_end,
        fl_min=fl_min,
        fl_max=fl_max,
        fl_text=fl_text,
        active_start=active_start,
        active_end=active_end,
        polygon=polygon,
        polygon_points=coord_points if polygon_terms else [],
        reference=reference,
        radius_center=radius_center,
        radius_nm=radius_nm,
        fir=fir,
        mentioned_waypoints=mentioned_waypoints,
        mentioned_airways=mentioned_airways,
        schedule_text=schedule_text,
        fields=fields,
        start_utc_text=start_text,
        end_utc_text=end_text,
        parsing_status=parsing_status,
        review_reason="; ".join(dict.fromkeys(review_reasons)),
        q_code=q_data.get("q_code", ""),
        traffic=q_data.get("traffic", ""),
        purpose=q_data.get("purpose", ""),
        scope=q_data.get("scope", ""),
        geometry_type=geometry_type,
        q_reference=q_data.get("reference_raw", ""),
        q_reference_center=q_data.get("reference_center"),
        q_radius_nm=q_data.get("radius_nm"),
    )


def parse_notams(notam_bulletin: str) -> list[NotamRestriction]:
    text = notam_bulletin.strip()
    if not text:
        return []
    starts = list(re.finditer(r"(?im)^\s*(?=(?:[A-Z]\d{3,5}/\d{2}\b|NOTAM[NC]?\b))", text))
    if not starts:
        starts = list(re.finditer(r"(?im)^\s*(?=(?:Q\)\s+[A-Z]{4}/|A\)\s+[A-Z]{4}\b))", text))
    if len(starts) > 1:
        chunks = []
        if starts[0].start() > 0 and re.search(r"(?im)^\s*A\)\s+", text[:starts[0].start()]):
            chunks.append(text[:starts[0].start()].strip())
        chunks.extend(text[starts[i].start():starts[i + 1].start()].strip() for i in range(len(starts) - 1))
        chunks.append(text[starts[-1].start():].strip())
    elif len(starts) == 1 and starts[0].start() > 0 and re.search(r"(?im)^\s*A\)\s+", text[:starts[0].start()]):
        chunks = [text[:starts[0].start()].strip(), text[starts[0].start():].strip()]
    else:
        chunks = re.split(r"\n\s*\n(?=\s*(?:[A-Z]\d{3,5}/\d{2}\b|NOTAM[NC]?\b|A\)\s+[A-Z]{4}\b))", text, flags=re.IGNORECASE)
    chunks = [chunk for chunk in chunks if chunk.strip()]
    return [parse_notam(chunk) for chunk in chunks]


def parse_lido_header(ofp_text: str) -> dict:
    first_line = next((line.strip() for line in ofp_text.splitlines() if line.strip()), "")
    upper = first_line.upper()
    flight_match = re.search(r"\b(?:[A-Z]{3})?(\d{2,4})[A-Z]?/[A-Z]{2}(\d{2,4})[A-Z]?\b", upper)
    date_match = re.search(r"\b(\d{2}[A-Z]{3})\b", upper)
    dep_dest_match = re.search(r"\b([A-Z]{4})/([A-Z]{4})\b", upper)
    aircraft_match = re.search(r"\b(A3\d{2}|A35K|A388|B7\d{2}|B77W|B78X|B789|B788|B744)\b", upper)
    registration_match = re.search(r"\b([A-Z0-9]{2}[A-Z]{3})\b\s*$", upper)
    return {
        "flight_number": f"QR{flight_match.group(2)}" if flight_match else "",
        "date": date_match.group(1) if date_match else "",
        "dep": dep_dest_match.group(1) if dep_dest_match else "",
        "dest": dep_dest_match.group(2) if dep_dest_match else "",
        "aircraft": aircraft_match.group(1) if aircraft_match else "",
        "registration": registration_match.group(1) if registration_match else "",
        "header_raw": first_line,
    }


def extract_route_from_ofp(ofp_text: str) -> str:
    lines = [line.rstrip() for line in ofp_text.splitlines()]
    for idx, line in enumerate(lines):
        if not re.match(r"^\s*ATC\s+CLEARANCE\b", line.upper()):
            continue
        block = []
        for candidate in lines[idx + 1:]:
            stripped = candidate.strip()
            upper_candidate = stripped.upper()
            if not stripped:
                if block:
                    break
                continue
            if re.match(r"^(FL\s*\d{2,3}|FUEL|TAXI|TRIP|CONT|ALTN|FINL|MIN|WEIGHT|ATIS|NOTAMS?)\b", upper_candidate):
                break
            block.append(stripped)
        return clean_route_text(" ".join(block))
    return ""


def is_runway_token(token: str) -> bool:
    return bool(re.fullmatch(r"\d{2}[LCR]?", token))


def is_route_connector(token: str) -> bool:
    return token == "DCT" or is_airway_token(token)


def is_valid_route_waypoint(token: str, dep: str = "", dest: str = "") -> bool:
    blocked = {"TAXI", "TRIP", "CONT", "ALTN", "FINL", "MIN", "FUEL", "WEIGHT", "ATIS", "NOTAMS", "NOTAM"}
    if not is_waypoint_token(token):
        return False
    if token in {dep, dest, "DCT"} or token in blocked:
        return False
    if is_runway_token(token):
        return False
    return True


def route_tokens(route_text: str) -> list[str]:
    return clean_route_text(route_text).split()


def parse_route_sequence(route_text: str, dep: str = "", dest: str = "") -> pd.DataFrame:
    tokens = route_tokens(route_text)
    waypoint_order = []
    airway_in: dict[str, str] = {}
    airway_out: dict[str, str] = {}

    def add_waypoint(token: str):
        if token not in waypoint_order:
            waypoint_order.append(token)

    for idx, token in enumerate(tokens):
        if not is_route_connector(token):
            continue
        prev_wp = next((candidate for candidate in reversed(tokens[:idx]) if is_valid_route_waypoint(candidate, dep, dest)), "")
        next_wp = next((candidate for candidate in tokens[idx + 1:] if is_valid_route_waypoint(candidate, dep, dest)), "")
        if prev_wp:
            add_waypoint(prev_wp)
            airway_out[prev_wp] = token
        if next_wp:
            add_waypoint(next_wp)
            airway_in[next_wp] = token

    rows = []
    for idx, waypoint in enumerate(waypoint_order, start=1):
        rows.append({
            "seq": idx,
            "waypoint": waypoint,
            "airway_from_prev": airway_in.get(waypoint, ""),
            "airway_to_next": airway_out.get(waypoint, ""),
        })
    return pd.DataFrame(rows)


def parse_route_segments(route_text: str, dep: str = "", dest: str = "") -> list[dict]:
    sequence = parse_route_sequence(route_text, dep, dest)
    if sequence.empty:
        return []
    segments = []
    rows = list(sequence.itertuples())
    for current, nxt in zip(rows, rows[1:]):
        airway = current.airway_to_next or nxt.airway_from_prev
        if airway:
            segments.append({"airway": airway, "start": current.waypoint, "end": nxt.waypoint})
    return segments


def lido_coord_to_decimal(direction: str, degrees: str, minutes: str) -> float:
    value = int(degrees) + float(minutes) / 60
    return -value if direction.upper() in {"S", "W"} else value


def parse_coordinates(coordinates_text: str) -> pd.DataFrame:
    rows = []
    in_section = False
    for raw_line in coordinates_text.splitlines():
        line = raw_line.strip()
        upper = line.upper()
        if re.match(r"^LAT\s*/\s*LONG\b", upper):
            in_section = True
            continue
        if not in_section:
            continue
        if re.match(r"^PAGE\b", upper):
            continue
        if re.match(r"^(AWY|WPT\s*/\s*FREQ|ATC\s+CLEARANCE|FUEL|TAXI|TRIP|CONT|ALTN|FINL|WX|WEATHER|NOTAM)\b", upper):
            break
        match = re.search(
            r"\b([NS])(\d{2})(\d{2}(?:\.\d+)?)\s*/\s*([EW])(\d{3})(\d{2}(?:\.\d+)?)\s+([A-Z][A-Z0-9]{1,5})\b",
            upper,
        )
        if not match:
            continue
        lat = lido_coord_to_decimal(match.group(1), match.group(2), match.group(3))
        lon = lido_coord_to_decimal(match.group(4), match.group(5), match.group(6))
        waypoint = match.group(7)
        if any(row["waypoint"] == waypoint for row in rows):
            continue
        rows.append({
            "waypoint": waypoint,
            "lat_raw": f"{match.group(1)}{match.group(2)}{match.group(3)}",
            "lon_raw": f"{match.group(4)}{match.group(5)}{match.group(6)}",
            "lat": lat,
            "lon": lon,
            "raw": raw_line,
        })
    return pd.DataFrame(rows)


def build_route_sequence_table(route_text: str, dep: str = "", dest: str = "") -> pd.DataFrame:
    route_sequence = parse_route_sequence(route_text, dep, dest)
    if route_sequence.empty:
        return pd.DataFrame(columns=["Seq", "Waypoint", "Airway_In", "Airway_Out"])
    return route_sequence.rename(columns={
        "seq": "Seq",
        "waypoint": "Waypoint",
        "airway_from_prev": "Airway_In",
        "airway_to_next": "Airway_Out",
    })[["Seq", "Waypoint", "Airway_In", "Airway_Out"]]


def build_coordinate_table(coord_df: pd.DataFrame) -> pd.DataFrame:
    if coord_df.empty:
        return pd.DataFrame(columns=["Waypoint", "Latitude", "Longitude", "Raw_LatLon"])
    table = coord_df.copy()
    table["Raw_LatLon"] = table.get("lat_raw", "").astype(str) + "/" + table.get("lon_raw", "").astype(str)
    return table.rename(columns={
        "waypoint": "Waypoint",
        "lat": "Latitude",
        "lon": "Longitude",
    })[["Waypoint", "Latitude", "Longitude", "Raw_LatLon"]].drop_duplicates("Waypoint", keep="first")


def build_eto_table(navlog_df: pd.DataFrame) -> pd.DataFrame:
    if navlog_df.empty:
        return pd.DataFrame(columns=["Waypoint", "ETO_UTC", "eta", "offset"])
    cols = ["waypoint", "eto_utc", "eta", "offset"]
    present = [col for col in cols if col in navlog_df]
    table = navlog_df[present].copy()
    for col in cols:
        if col not in table:
            table[col] = None
    return table.rename(columns={"waypoint": "Waypoint", "eto_utc": "ETO_UTC"})[
        ["Waypoint", "ETO_UTC", "eta", "offset"]
    ].drop_duplicates("Waypoint", keep="first")


def build_fir_table(navlog_df: pd.DataFrame) -> pd.DataFrame:
    if navlog_df.empty or "fir" not in navlog_df:
        return pd.DataFrame(columns=["Waypoint", "FIR"])
    table = navlog_df[["waypoint", "fir"]].copy()
    return table.rename(columns={"waypoint": "Waypoint", "fir": "FIR"})[
        ["Waypoint", "FIR"]
    ].drop_duplicates("Waypoint", keep="first")


def build_fl_table(route_sequence_table: pd.DataFrame, ofp_text: str) -> pd.DataFrame:
    if route_sequence_table.empty:
        return pd.DataFrame(columns=["Waypoint", "Planned_FL"])
    default_fl, changes = parse_planned_fl_profile(ofp_text)
    rows = []
    current_fl = default_fl
    for row in route_sequence_table.itertuples(index=False):
        waypoint = row.Waypoint
        if waypoint in changes:
            current_fl = changes[waypoint]
        rows.append({"Waypoint": waypoint, "Planned_FL": current_fl})
    return pd.DataFrame(rows)


def merge_ofp_tables(
    route_sequence_table: pd.DataFrame,
    coordinate_table: pd.DataFrame,
    eto_table: pd.DataFrame,
    fir_table: pd.DataFrame,
    fl_table: pd.DataFrame,
) -> pd.DataFrame:
    final = route_sequence_table.copy()
    for table in [coordinate_table, eto_table, fir_table, fl_table]:
        final = final.merge(table, on="Waypoint", how="left")
    final = final.rename(columns={
        "Seq": "seq",
        "Waypoint": "waypoint",
        "Airway_In": "airway_from_prev",
        "Airway_Out": "airway_to_next",
        "Latitude": "lat",
        "Longitude": "lon",
        "FIR": "fir",
        "ETO_UTC": "eto_utc",
        "Planned_FL": "planned_fl",
        "Raw_LatLon": "raw_coord",
    })
    final["fl"] = final["planned_fl"]
    for col in ["eta", "offset", "eto_utc", "fir", "planned_fl", "lat", "lon", "raw_coord"]:
        if col not in final:
            final[col] = None
    return final


def ofp_parser_quality(route_table: pd.DataFrame, coordinate_table: pd.DataFrame, eto_table: pd.DataFrame, fir_table: pd.DataFrame, fl_table: pd.DataFrame) -> dict:
    route_waypoints = set(route_table["Waypoint"]) if not route_table.empty else set()
    coordinate_waypoints = set(coordinate_table["Waypoint"]) if not coordinate_table.empty else set()
    eto_waypoints = set(eto_table.loc[eto_table["ETO_UTC"].fillna("").astype(str).ne(""), "Waypoint"]) if not eto_table.empty else set()
    fir_waypoints = set(fir_table.loc[fir_table["FIR"].fillna("").astype(str).ne(""), "Waypoint"]) if not fir_table.empty else set()
    fl_waypoints = set(fl_table.loc[fl_table["Planned_FL"].notna(), "Waypoint"]) if not fl_table.empty else set()
    missing_coordinates = sorted(route_waypoints - coordinate_waypoints)
    missing_eto = sorted(route_waypoints - eto_waypoints)
    missing_fir = sorted(route_waypoints - fir_waypoints)
    missing_fl = sorted(route_waypoints - fl_waypoints)
    coordinates_not_used = sorted(coordinate_waypoints - route_waypoints)
    return {
        "summary": {
            "Route waypoints extracted": len(route_waypoints),
            "Coordinates matched": f"{len(route_waypoints & coordinate_waypoints)} / {len(route_waypoints)}",
            "ETO matched": f"{len(route_waypoints & eto_waypoints)} / {len(route_waypoints)}",
            "FIR assigned": f"{len(route_waypoints & fir_waypoints)} / {len(route_waypoints)}",
            "FL assigned": f"{len(route_waypoints & fl_waypoints)} / {len(route_waypoints)}",
        },
        "missing_coordinates": pd.DataFrame({"Waypoint": missing_coordinates}),
        "missing_eto": pd.DataFrame({"Waypoint": missing_eto}),
        "missing_fir": pd.DataFrame({"Waypoint": missing_fir}),
        "missing_fl": pd.DataFrame({"Waypoint": missing_fl}),
        "coordinates_not_used": pd.DataFrame({"Waypoint": coordinates_not_used}),
    }


def parse_planned_fl_profile(ofp_text: str) -> tuple[Optional[int], dict[str, int]]:
    match = re.search(r"\bFL\s*(\d{2,3})\b(?P<rest>(?:\s+[A-Z][A-Z0-9]{1,5}/FL\s*\d{2,3})+)", ofp_text.upper())
    if not match:
        return None, {}
    default_fl = int(match.group(1))
    changes = {
        waypoint: int(fl)
        for waypoint, fl in re.findall(r"\b([A-Z][A-Z0-9]{1,5})/FL\s*(\d{2,3})\b", match.group("rest"))
    }
    return default_fl, changes


def parse_navlog(navlog_text: str, etd: datetime, route_waypoints: Optional[list[str]] = None, coordinate_waypoints: Optional[list[str]] = None) -> pd.DataFrame:
    allowed = set(route_waypoints or []) | set(coordinate_waypoints or [])
    if not allowed:
        return pd.DataFrame(columns=["waypoint", "eta", "offset", "eto_utc", "fir", "raw"])

    rows = []
    current_fir = ""
    in_navlog = False
    seen_header = False
    seen_waypoints = set()
    fir_pattern = re.compile(r"\b([A-Z][A-Z ]+?)\s+(?:FIR|UIR)\s+([A-Z]{4})\b")
    for raw_line in navlog_text.splitlines():
        line = raw_line.strip()
        upper = line.upper()
        if not line:
            continue
        if re.search(r"\bAWY\s+ITT\s+FL\s+WIND\s+ISAD\s+STM\s+ETA\s+AFOB\b", upper):
            in_navlog = True
            seen_header = True
            continue
        if seen_header and re.search(r"\bWPT\s*/\s*FREQ\b", upper):
            in_navlog = True
            continue
        if not in_navlog:
            continue
        if re.match(r"^PAGE\b", upper):
            continue
        if re.match(r"^(FUEL|TAXI|TRIP|CONT|ALTN|FINL|LAT\s*/\s*LONG|ATC\s+CLEARANCE|WX|WEATHER|NOTAM)\b", upper):
            break

        fir_match = fir_pattern.search(upper)
        if fir_match:
            current_fir = fir_match.group(2)
            continue

        tokens = upper.split()
        if not tokens:
            continue
        waypoint = tokens[0]
        if waypoint not in allowed or waypoint in seen_waypoints:
            continue
        eta_candidates = re.findall(r"\b(\d{4})\b", upper)
        if not eta_candidates:
            continue
        eto_token = eta_candidates[-1]
        eta = etd + timedelta(hours=int(eto_token[:2]), minutes=int(eto_token[2:]))
        rows.append({
            "waypoint": waypoint,
            "eta": eta,
            "offset": eta - etd,
            "eto_utc": eto_token,
            "fir": current_fir,
            "raw": raw_line,
        })
        seen_waypoints.add(waypoint)
    return pd.DataFrame(rows)


def apply_route_flight_level_profile(route_df: pd.DataFrame, ofp_text: str) -> pd.DataFrame:
    if route_df.empty:
        return route_df
    default_fl, changes = parse_planned_fl_profile(ofp_text)
    profiled = route_df.copy()
    if default_fl is None:
        profiled["fl"] = None
        profiled["planned_fl"] = None
        return profiled

    current_fl = default_fl
    fl_values = []
    for row in profiled.itertuples():
        if row.waypoint in changes:
            current_fl = changes[row.waypoint]
        fl_values.append(current_fl)
    profiled["fl"] = fl_values
    profiled["planned_fl"] = fl_values
    return profiled


def navlog_signature(navlog_text: str) -> str:
    return re.sub(r"\s+", " ", navlog_text.strip().upper())


def ensure_baseline_etd(navlog_text: str, current_etd: datetime, extracted_etd_text: Optional[str] = None) -> datetime:
    signature = navlog_signature(navlog_text)
    if st.session_state.get("navlog_signature") != signature:
        st.session_state["navlog_signature"] = signature
        st.session_state["baseline_etd"] = extracted_etd_text or current_etd.strftime("%H%M")
    baseline = parse_hhmm(st.session_state.get("baseline_etd", current_etd.strftime("%H%M")), current_etd.date())
    return baseline or current_etd


def shift_navlog_to_etd(navlog_df: pd.DataFrame, scenario_etd: datetime) -> pd.DataFrame:
    shifted = navlog_df.copy()
    if shifted.empty or "offset" not in shifted:
        return shifted
    shifted["eta"] = shifted["offset"].apply(lambda offset: scenario_etd + offset)
    return shifted


def parse_coordinates(coordinates_text: str) -> pd.DataFrame:
    rows = []
    in_section = False
    for raw_line in coordinates_text.splitlines():
        line = raw_line.strip()
        upper = line.upper()
        if re.match(r"^LAT\s*/\s*LONG\b", upper):
            in_section = True
            continue
        if not in_section:
            continue
        if re.match(r"^PAGE\b", upper):
            continue
        if re.match(r"^(AWY|WPT\s*/\s*FREQ|ATC\s+CLEARANCE|FUEL|TAXI|TRIP|CONT|ALTN|FINL|WX|WEATHER|NOTAM)\b", upper):
            break
        match = re.search(
            r"\b([NS])(\d{2})(\d{2}(?:\.\d+)?)\s*/\s*([EW])(\d{3})(\d{2}(?:\.\d+)?)\s+([A-Z][A-Z0-9]{1,5})\b",
            upper,
        )
        if not match:
            continue
        waypoint = match.group(7)
        if any(row["waypoint"] == waypoint for row in rows):
            continue
        rows.append({
            "waypoint": waypoint,
            "lat_raw": f"{match.group(1)}{match.group(2)}{match.group(3)}",
            "lon_raw": f"{match.group(4)}{match.group(5)}{match.group(6)}",
            "lat": lido_coord_to_decimal(match.group(1), match.group(2), match.group(3)),
            "lon": lido_coord_to_decimal(match.group(4), match.group(5), match.group(6)),
            "raw": raw_line,
        })
    return pd.DataFrame(rows)


def extract_etd_from_ofp(ofp_text: str) -> Optional[str]:
    patterns = [
        r"\b(?:ETD|STD|DEP(?:ARTURE)?\s*TIME)\s*[:=]?\s*\d{1,2}/([01]\d|2[0-3])([0-5]\d)\b",
        r"\b(?:ETD|STD|DEP(?:ARTURE)?\s*TIME)\s*[:=]?\s*([01]\d|2[0-3])([0-5]\d)\b",
        r"\b(?:ETD|STD)\s*[:=]?\s*([01]\d|2[0-3]):([0-5]\d)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, ofp_text.upper())
        if match:
            return f"{match.group(1)}{match.group(2)}"
    return None


def extract_ofp_sections(ofp_text: str) -> dict:
    markers = {
        "atc_clearance": r"\b(?:ATC\s+CLEARANCE|ATC\s+ROUTE|ROUTE)\b",
        "navlog": r"\b(?:AWY|WPT\s*/\s*FREQ|NAVLOG)\b",
        "coordinates": r"\b(?:WAYPOINT COORDINATES|COORDINATES|LAT/LON|LAT\s+LON)\b",
        "fuel_summary": r"\b(?:FUEL|TRIP|MIN FUEL REQ|RAMP FUEL)\b",
        "weights": r"\b(?:MZFW|MTOW|MLWT|EZFW|ETOW|ELWT)\b",
        "alternate": r"\b(?:DEST ALTN|ALTN|ALTERNATE)\b",
        "weather": r"\b(?:WX|FORECAST WX|CEILING/VIS)\b",
        "dispatcher_notes": r"\b(?:REMARK|NOTAM|SELF-BRIEFING|SIGNATURE)\b",
    }
    lines = ofp_text.splitlines()
    sections = {name: [] for name in markers}
    current = None
    for line in lines:
        upper = line.upper()
        matched = next((name for name, pattern in markers.items() if re.search(pattern, upper)), None)
        if matched:
            current = matched
        if current:
            sections[current].append(line)
    return {name: "\n".join(value).strip() for name, value in sections.items() if value}


def extract_text_from_pdf(uploaded_pdf) -> str:
    if uploaded_pdf is None:
        return ""
    try:
        import pypdf

        uploaded_pdf.seek(0)
        reader = pypdf.PdfReader(uploaded_pdf)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip()
    except Exception:
        return ""


def extract_route_components(route_text: str) -> dict:
    tokens = clean_route_text(route_text).split()
    airways = [token for token in tokens if is_airway_token(token)]
    waypoints = [token for token in tokens if is_valid_route_waypoint(token)]
    return {"airways": airways, "waypoints": waypoints}


def extract_ofp_metadata(ofp_text: str, route_text: str, navlog_df: pd.DataFrame) -> dict:
    header = parse_lido_header(ofp_text)
    route_sequence = parse_route_sequence(route_text, header.get("dep", ""), header.get("dest", ""))
    route_parts = {
        "airways": [token for token in route_tokens(route_text) if is_airway_token(token)],
        "waypoints": route_sequence["waypoint"].tolist() if not route_sequence.empty else [],
    }
    firs = sorted(set(navlog_df["fir"].dropna().astype(str)) - {""}) if not navlog_df.empty and "fir" in navlog_df else []
    planned_fls = sorted({int(fl) for fl in navlog_df["fl"].dropna().tolist()}) if not navlog_df.empty and "fl" in navlog_df else []
    return {
        "flight_number": header.get("flight_number", ""),
        "date": header.get("date", ""),
        "dep": header.get("dep", ""),
        "dest": header.get("dest", ""),
        "aircraft": header.get("aircraft", ""),
        "registration": header.get("registration", ""),
        "airways": route_parts["airways"],
        "waypoints": route_parts["waypoints"],
        "fir_boundaries": firs,
        "planned_flight_levels": planned_fls,
        "eto_utc": {
            row.waypoint: (
                row.eto_utc
                if hasattr(row, "eto_utc") and pd.notna(row.eto_utc) and str(row.eto_utc)
                else (row.eta.strftime("%H:%M UTC") if hasattr(row, "eta") and pd.notna(row.eta) else "")
            )
            for row in navlog_df.itertuples()
        } if not navlog_df.empty else {},
    }


def is_airway_token(token: str) -> bool:
    if token.startswith(("QR", "QTR", "FL")):
        return False
    return bool(re.fullmatch(r"[A-Z]{1,3}\d{1,4}[A-Z]?", token))


def is_waypoint_token(token: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z0-9]{1,5}", token)) and not is_airway_token(token)


def clean_route_text(route_text: str) -> str:
    route = route_text.upper()
    route = re.sub(r"\b(?:ROUTE|ATC|CLEARANCE|RTE|FPL|FLT|FLIGHT|PLAN)\b\s*[:=]?", " ", route)
    route = re.sub(r"[^A-Z0-9\s]", " ", route)
    return re.sub(r"\s+", " ", route).strip()


def extract_route_from_ofp(ofp_text: str) -> str:
    lines = [line.rstrip() for line in ofp_text.splitlines()]
    for idx, line in enumerate(lines):
        if not re.match(r"^\s*ATC\s+CLEARANCE\b", line.upper()):
            continue
        block = []
        for candidate in lines[idx + 1:]:
            stripped = candidate.strip()
            upper_candidate = stripped.upper()
            if not stripped:
                if block:
                    break
                continue
            if re.match(r"^(FL\s*\d{2,3}|FUEL|TAXI|TRIP|CONT|ALTN|FINL|MIN|WEIGHT|ATIS|NOTAMS?)\b", upper_candidate):
                break
            block.append(stripped)
        return clean_route_text(" ".join(block))
    return ""


def parse_route_segments(route_text: str, dep: str = "", dest: str = "") -> list[dict]:
    sequence = parse_route_sequence(route_text, dep, dest)
    if sequence.empty:
        return []
    segments = []
    rows = list(sequence.itertuples())
    for current, nxt in zip(rows, rows[1:]):
        airway = current.airway_to_next or nxt.airway_from_prev
        if airway:
            segments.append({"airway": airway, "start": current.waypoint, "end": nxt.waypoint})
    return segments


def parse_route(ofp_text: str, manual_route: str = "") -> str:
    return extract_route_from_ofp(ofp_text) or manual_route.strip()


def parse_ofp(ofp_text: str, etd: datetime, manual_route: str = "") -> dict:
    header = parse_lido_header(ofp_text)
    route_text = parse_route(ofp_text, manual_route)
    route_sequence_table = build_route_sequence_table(route_text, header.get("dep", ""), header.get("dest", ""))
    route_waypoints = route_sequence_table["Waypoint"].tolist() if not route_sequence_table.empty else []
    route_segments = parse_route_segments(route_text, header.get("dep", ""), header.get("dest", ""))
    coord_df = parse_coordinates(ofp_text)
    coordinate_waypoints = coord_df["waypoint"].tolist() if not coord_df.empty else []
    navlog_df = parse_navlog(ofp_text, etd, route_waypoints, coordinate_waypoints)
    coordinate_table = build_coordinate_table(coord_df)
    eto_table = build_eto_table(navlog_df)
    fir_table = build_fir_table(navlog_df)
    fl_table = build_fl_table(route_sequence_table, ofp_text)

    if route_sequence_table.empty:
        internal_route_df = pd.DataFrame(columns=[
            "seq", "waypoint", "airway_from_prev", "airway_to_next", "eta", "offset",
            "eto_utc", "fir", "fl", "planned_fl", "lat", "lon", "raw_coord",
        ])
    else:
        internal_route_df = merge_ofp_tables(route_sequence_table, coordinate_table, eto_table, fir_table, fl_table)

    coord_route_df = internal_route_df.dropna(subset=["lat", "lon"]) if not internal_route_df.empty and {"lat", "lon"}.issubset(internal_route_df.columns) else pd.DataFrame()
    route = LineString(list(zip(coord_route_df["lon"], coord_route_df["lat"]))) if len(coord_route_df) >= 2 else None
    parser_tables = {
        "route_sequence": route_sequence_table,
        "coordinates": coordinate_table,
        "eto": eto_table[["Waypoint", "ETO_UTC"]] if not eto_table.empty else pd.DataFrame(columns=["Waypoint", "ETO_UTC"]),
        "fir": fir_table,
        "fl": fl_table,
    }
    return {
        "route_text": route_text,
        "route_segments": route_segments,
        "navlog": navlog_df,
        "coordinates": coord_df,
        "route_geometry": route,
        "route_df": internal_route_df,
        "etd": extract_etd_from_ofp(ofp_text),
        "sections": extract_ofp_sections(ofp_text),
        "metadata": extract_ofp_metadata(ofp_text, route_text, internal_route_df),
        "parser_tables": parser_tables,
        "parser_quality": ofp_parser_quality(route_sequence_table, coordinate_table, eto_table, fir_table, fl_table),
    }


def route_contains_segment(route_segments: list[dict], airway: Optional[str], start_wp: str, end_wp: str) -> bool:
    for segment in route_segments:
        same_waypoints = {segment["start"], segment["end"]} == {start_wp, end_wp}
        same_airway = not airway or segment["airway"] == airway
        if same_waypoints and same_airway:
            return True
    return False


def build_route(navlog_df: pd.DataFrame, coord_df: pd.DataFrame) -> tuple[Optional[LineString], pd.DataFrame]:
    if navlog_df.empty or coord_df.empty:
        return None, pd.DataFrame()
    merged = navlog_df.merge(coord_df, on="waypoint", how="inner", suffixes=("", "_coord"))
    merged = merged.drop_duplicates(subset=["waypoint", "eta"], keep="first")
    if len(merged) < 2:
        return None, merged
    route = LineString(list(zip(merged["lon"], merged["lat"])))
    return route, merged


def build_internal_route_dataframe(route_df: pd.DataFrame, route_segments: list[dict]) -> pd.DataFrame:
    if route_df.empty:
        return route_df
    enriched = route_df.copy()
    if {"airway_to_next", "airway_from_prev", "eto_utc", "planned_fl", "fir"}.issubset(enriched.columns):
        return enriched
    airway_by_start = {segment["start"]: segment["airway"] for segment in route_segments}
    airway_by_end = {segment["end"]: segment["airway"] for segment in route_segments}
    enriched["airway_to_next"] = enriched["waypoint"].map(airway_by_start).fillna("")
    enriched["airway_from_prev"] = enriched["waypoint"].map(airway_by_end).fillna("")
    enriched["eto_utc"] = enriched["eta"].apply(lambda value: value.strftime("%H:%M UTC") if pd.notna(value) else "")
    enriched["planned_fl"] = enriched["fl"] if "fl" in enriched else None
    enriched["fir"] = enriched["fir"] if "fir" in enriched else ""
    return enriched


def fl_matches(fl: Optional[int], fl_min: Optional[int], fl_max: Optional[int]) -> bool:
    if fl is None:
        return False
    if fl_min is None and fl_max is None:
        return True
    return (fl_min or 0) <= fl <= (fl_max or 999)


def restriction_closes_all_levels(notam: NotamRestriction) -> bool:
    return notam.fl_min == 0 and (notam.fl_max or 0) >= 999


def suggest_flight_levels(
    current_fl: Optional[int],
    fl_min: Optional[int],
    fl_max: Optional[int],
    ceiling: int = 600,
) -> list[int]:
    if current_fl is None or fl_min is None or fl_max is None:
        return []
    if fl_min <= 0 and fl_max >= 999:
        return []

    candidates = []
    below = fl_min - 10
    above = fl_max + 10
    if below >= 0:
        candidates.append(below)
    if above <= ceiling:
        candidates.append(above)
    return sorted(candidates, key=lambda fl: (abs(fl - current_fl), 0 if fl > current_fl else 1))


def flight_level_resolution_options(result: dict, notam: NotamRestriction) -> list[dict]:
    current_fl = result.get("flight_level")
    if not result.get("impacted") or current_fl is None:
        return []
    if not fl_matches(current_fl, notam.fl_min, notam.fl_max):
        return []
    if notam.fl_min is None or notam.fl_max is None or restriction_closes_all_levels(notam):
        return []

    options = []
    above = notam.fl_max + 10
    below = notam.fl_min - 10
    if notam.fl_max < 999 and above <= 600:
        options.append({
            "type": "fl",
            "label": f"FL{above:03d}",
            "subtitle": "Above restriction",
            "detail": "Avoids restricted airspace vertically.",
        })
    if notam.fl_min > 0 and below >= 0:
        options.append({
            "type": "fl",
            "label": f"FL{below:03d}",
            "subtitle": "Below restriction",
            "detail": "Avoids restricted airspace vertically.",
        })
    return options


def evaluate_operational_resolution(result: dict, notam: NotamRestriction) -> dict:
    if not result.get("impacted"):
        return {
            "resolution": "No Action Required",
            "reason": "At least one required impact condition does not match.",
            "expected_result": "No change required based on the current prototype assessment.",
        }

    current_fl = result.get("flight_level")
    suggestions = suggest_flight_levels(current_fl, notam.fl_min, notam.fl_max)
    if suggestions:
        suggested = suggestions[0]
        direction = "below" if suggested < (notam.fl_min or 0) else "above"
        return {
            "resolution": "FL Change Recommended",
            "suggested_fl": suggested,
            "alternate_fls": suggestions[1:],
            "reason": (
                f"Current FL{current_fl:03d} is inside {notam.fl_text}. "
                f"FL{suggested:03d} is immediately {direction} the restricted band."
            ),
            "expected_result": "Impact removed. No ETD change required.",
        }

    if restriction_closes_all_levels(notam) or notam.fl_min is None:
        return {
            "resolution": "ETD Change Recommended",
            "reason": "The restriction leaves no usable flight level in this prototype assessment.",
            "expected_result": "Use timing to avoid the active period, or plan a reroute if the timing change is not operationally acceptable.",
        }

    return {
        "resolution": "Reroute Required",
        "reason": "No practical flight level or timing solution could be established from the pasted data.",
        "expected_result": "Plan a route that avoids the affected segment or polygon.",
    }


def generate_available_resolutions(result: dict, notam: NotamRestriction, etd: datetime, safety_margin: int) -> list[dict]:
    if not result.get("impacted"):
        return [{
            "type": "none",
            "label": "No Action Required",
            "detail": "Current route, FL, or timing does not conflict with the NOTAM.",
        }]

    options = flight_level_resolution_options(result, notam)
    shift = calculate_etd_shift(result, notam, etd, safety_margin)
    if shift:
        if shift.get("minimum_advance", timedelta(0)) > timedelta(0):
            options.append({
                "type": "etd_advance",
                "label": "Advance ETD",
                "subtitle": fmt_duration(shift.get("minimum_advance")),
                "new_etd": etd - shift["minimum_advance"],
                "detail": "Aircraft passes before NOTAM activation.",
            })
        if shift.get("minimum_delay", timedelta(0)) > timedelta(0):
            options.append({
                "type": "etd_delay",
                "label": "Delay ETD",
                "subtitle": fmt_duration(shift.get("minimum_delay")),
                "new_etd": etd + shift["minimum_delay"],
                "detail": "Aircraft passes after NOTAM expiration.",
            })

    if not options:
        options.append({
            "type": "reroute",
            "label": "Reroute Required",
            "detail": "No vertical or timing resolution was calculated from the available data.",
        })
    return options


def analyze_notam(
    notam: NotamRestriction,
    navlog_df: pd.DataFrame,
    route_df: pd.DataFrame,
    route_geometry: Optional[LineString],
    route_segments: list[dict],
    etd: datetime,
    safety_margin: int,
) -> dict:
    if notam.restriction_type == "Unknown":
        result = {
            "impacted": False,
            "restriction_type": "Unknown",
            "affected": "Review NOTAM",
            "geo_match": False,
            "level_match": False,
            "time_match": False,
            "reason": "NOTAM restriction type could not be parsed confidently.",
            "review_required": True,
        }
    elif notam.restriction_type == "Polygon":
        result = check_polygon_impact(notam, route_geometry, route_df)
    elif notam.restriction_type == "Radius":
        result = check_radius_impact(notam, route_geometry, route_df)
    else:
        result = check_segment_impact(notam, navlog_df, route_df, route_segments)

    if notam.restriction_type in {"Polygon", "Radius"} and route_geometry is None:
        result.update({
            "review_required": True,
            "affected": "Review NOTAM",
            "reason": "Waypoint coordinates are missing, so polygon/radius impact cannot be validated.",
        })
    if not notam.active_start or not notam.active_end:
        result.update({
            "review_required": True,
            "affected": "Review NOTAM",
            "time_match": None,
            "reason": "NOTAM active time window could not be parsed.",
        })

    resolution = evaluate_operational_resolution(result, notam)
    timing_relevant = bool(result.get("entry_time") and result.get("exit_time") and result.get("geo_match") and result.get("level_match"))
    shift = calculate_etd_options(result, notam, etd, safety_margin) if timing_relevant else {}
    options = generate_available_resolutions(result, notam, etd, safety_margin)
    return {
        "notam": notam,
        "result": result,
        "resolution": resolution,
        "shift": shift,
        "options": options,
    }


def standardize_route_dataframe(route_df: pd.DataFrame) -> pd.DataFrame:
    if route_df.empty:
        return pd.DataFrame(columns=["Seq", "Waypoint", "Airway_In", "Airway_Out", "Latitude", "Longitude", "FIR", "ETO_UTC", "Planned_FL"])
    data = route_df.copy().reset_index(drop=True)
    seq_values = data["seq"] if "seq" in data else data.index + 1
    eto_values = data["eto_utc"] if "eto_utc" in data else (
        data["eta"].apply(lambda value: fmt_dt(value)) if "eta" in data else pd.Series([""] * len(data))
    )
    return pd.DataFrame({
        "Seq": seq_values,
        "Waypoint": data["waypoint"],
        "Airway_In": data.get("airway_from_prev", pd.Series([""] * len(data))).fillna(""),
        "Airway_Out": data.get("airway_to_next", pd.Series([""] * len(data))).fillna(""),
        "Latitude": data.get("lat", pd.Series([None] * len(data))),
        "Longitude": data.get("lon", pd.Series([None] * len(data))),
        "FIR": data.get("fir", pd.Series([""] * len(data))).fillna(""),
        "ETO_UTC": eto_values,
        "Planned_FL": data.get("planned_fl", data.get("fl", pd.Series([None] * len(data)))),
    })


def notam_records(notams: list[NotamRestriction]) -> pd.DataFrame:
    return pd.DataFrame([{
        "NOTAM_ID": notam.reference,
        "Q_Code": notam.q_code or "Unknown",
        "FIR": notam.fir or "Unknown",
        "Start_UTC": notam.start_utc_text or "Unknown",
        "End_UTC": notam.end_utc_text or "Unknown",
        "Schedule_Text": notam.schedule_text or "Unknown",
        "Restriction_Type": notam.restriction_type,
        "Geometry_Type": notam.geometry_type,
        "Lower_FL": notam.fl_min if notam.fl_min is not None else "Unknown",
        "Upper_FL": notam.fl_max if notam.fl_max is not None else "Unknown",
        "Mentioned_Airways": ", ".join(notam.mentioned_airways or []) or "Unknown",
        "Mentioned_Waypoints": ", ".join(notam.mentioned_waypoints or []) or "Unknown",
        "Segment_From": notam.segment_start or "Unknown",
        "Segment_To": notam.segment_end or "Unknown",
        "Polygon_Coordinates": f"{len(notam.polygon_points)} coordinate pairs" if notam.polygon_points else "Unknown",
        "Circle_Center": notam.radius_center or "Unknown",
        "Radius_NM": notam.radius_nm if notam.radius_nm is not None else "Unknown",
        "Raw_Text": notam.raw_text,
        "Parsing_Status": notam.parsing_status,
        "Review_Reason": notam.review_reason or "",
    } for notam in notams])


def recommendation_from_analysis(analysis: dict) -> tuple[str, str, str]:
    result = analysis["result"]
    if result.get("review_required"):
        return "Review NOTAM", "Manual review", "Restriction could not be parsed confidently."
    if not result.get("impacted"):
        if result.get("restriction_type") in {"Polygon", "Radius"} and not result.get("entry_time") and result.get("geo_match") is None:
            return "Review NOTAM", "Manual review", "Area analysis requires coordinates."
        return "No Impact", "No action", result.get("reason", "No conflict detected.")
    labels = [option["label"] + (f" {option.get('subtitle')}" if option.get("subtitle") else "") for option in analysis["options"]]
    return "Impacted", ", ".join(labels) if labels else "Review NOTAM", result.get("reason", "")


def results_dataframe(analyses: list[dict]) -> pd.DataFrame:
    rows = []
    for analysis in analyses:
        notam = analysis["notam"]
        result = analysis["result"]
        status, recommendation, comment = recommendation_from_analysis(analysis)
        rows.append({
            "NOTAM_ID": notam.reference,
            "Impact status": status,
            "Impact type": result.get("restriction_type", notam.restriction_type),
            "Matched element": result.get("affected", ""),
            "Impacted waypoint/segment": result.get("affected", ""),
            "ETO": fmt_dt(result.get("entry_time")),
            "Planned FL": fmt_fl(result.get("flight_level")),
            "Restriction": notam.fl_text,
            "Recommendation": recommendation,
            "Required action minutes": action_minutes(analysis),
            "Comment": comment,
        })
    return pd.DataFrame(rows)


def action_minutes(analysis: dict) -> str:
    shift = analysis.get("shift") or {}
    parts = []
    if shift.get("minimum_advance", timedelta(0)) > timedelta(0):
        parts.append(f"Advance {fmt_duration(shift['minimum_advance'])}")
    if shift.get("minimum_delay", timedelta(0)) > timedelta(0):
        parts.append(f"Delay {fmt_duration(shift['minimum_delay'])}")
    return "; ".join(parts)


def parsing_status(ofp_text: str, source_label: str, route_df: pd.DataFrame, coord_df: pd.DataFrame, route_segments: list[dict], notams: list[NotamRestriction]) -> dict:
    return {
        "OFP source": source_label,
        "OFP text extracted": "Yes" if ofp_text else "No",
        "Route waypoints": len(route_df) if not route_df.empty else 0,
        "Coordinates": f"{len(coord_df)} found" if not coord_df.empty else "Missing",
        "Route parsed": "Yes" if route_segments else "Route parsing incomplete",
        "FIRs detected": len({notam.fir for notam in notams if notam.fir}),
        "NOTAMs detected": len(notams),
        "Polygons detected": sum(1 for notam in notams if notam.restriction_type == "Polygon"),
        "Radius areas detected": sum(1 for notam in notams if notam.restriction_type == "Radius"),
        "FL restrictions detected": sum(1 for notam in notams if notam.fl_min is not None or notam.fl_max is not None),
        "Time windows detected": sum(1 for notam in notams if notam.active_start and notam.active_end),
    }


def unknown_if_empty(value) -> str:
    if value is None:
        return "Unknown"
    if isinstance(value, float) and pd.isna(value):
        return "Unknown"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "Unknown"
    if isinstance(value, tuple):
        return str(value) if value else "Unknown"
    text = str(value).strip()
    return text if text else "Unknown"


def yes_no_unknown(value) -> str:
    if value is None:
        return "UNKNOWN"
    return "YES" if bool(value) else "NO"


def add_diagnostic(rows: list[dict], engine: str, severity: str, expected: str, found: str, why: str):
    rows.append({
        "Engine": engine,
        "Severity": severity,
        "Expected": expected,
        "Found": found,
        "Why parsing failed": why,
    })


def engine1_ofp_output(parsed_ofp: dict, source_label: str, ofp_text: str, route_text: str, etd_text: str) -> tuple[pd.DataFrame, pd.DataFrame, bool, dict]:
    route_df = parsed_ofp.get("route_df", pd.DataFrame())
    metadata = parsed_ofp.get("metadata", {})
    coord_df = parsed_ofp.get("coordinates", pd.DataFrame())
    route_segments = parsed_ofp.get("route_segments", [])
    quality = parsed_ofp.get("parser_quality", {})
    quality_summary = quality.get("summary", {})
    route_table = standardize_route_dataframe(route_df)
    route_waypoints = set(route_df["waypoint"].dropna().astype(str)) if not route_df.empty and "waypoint" in route_df else set()
    coord_waypoints = set(coord_df["waypoint"].dropna().astype(str)) if not coord_df.empty and "waypoint" in coord_df else set()
    eto_waypoints = set(route_df.loc[route_df["eto_utc"].fillna("").astype(str) != "", "waypoint"].astype(str)) if not route_df.empty and "eto_utc" in route_df else set()
    rows = []
    for row in route_table.itertuples(index=False):
        rows.append({
            "Flight": unknown_if_empty(metadata.get("flight_number")),
            "Departure": unknown_if_empty(metadata.get("dep")),
            "Destination": unknown_if_empty(metadata.get("dest")),
            "Aircraft": unknown_if_empty(metadata.get("aircraft")),
            "Waypoint Sequence": row.Seq,
            "Waypoint": unknown_if_empty(row.Waypoint),
            "Airway_In": unknown_if_empty(row.Airway_In),
            "Airway_Out": unknown_if_empty(row.Airway_Out),
            "Latitude": unknown_if_empty(row.Latitude),
            "Longitude": unknown_if_empty(row.Longitude),
            "FIR": unknown_if_empty(row.FIR),
            "ETO UTC": unknown_if_empty(row.ETO_UTC),
            "Planned FL": unknown_if_empty(row.Planned_FL),
        })
    ofp_table = pd.DataFrame(rows, columns=[
        "Flight", "Departure", "Destination", "Aircraft", "Waypoint Sequence", "Waypoint",
        "Airway_In", "Airway_Out", "Latitude", "Longitude", "FIR", "ETO UTC", "Planned FL",
    ])

    diagnostics = []
    if not ofp_text.strip():
        add_diagnostic(diagnostics, "Engine 1 - OFP Parser", "CRITICAL", "Uploaded PDF text or manual OFP text", "No OFP text found", "The OFP parser has no source text to inspect.")
    if not etd_text:
        add_diagnostic(diagnostics, "Engine 1 - OFP Parser", "CRITICAL", "ETD in OFP or ETD override", "Unknown", "Waypoint ETO offsets cannot be shifted without an ETD.")
    if not route_text:
        add_diagnostic(diagnostics, "Engine 1 - OFP Parser", "CRITICAL", "OFP route string or manual route input", "Unknown", "Airway and segment matching require a route string.")
    if ofp_table.empty:
        add_diagnostic(diagnostics, "Engine 1 - OFP Parser", "CRITICAL", "Waypoint, ETO UTC, and planned FL rows", "No route dataframe rows", "The navlog parser could not identify usable waypoint timing rows.")
    if coord_df.empty:
        add_diagnostic(diagnostics, "Engine 1 - OFP Parser", "WARNING", "Waypoint coordinates", "Missing", "Geospatial polygon/radius analysis will be skipped.")
    if not metadata.get("fir_boundaries"):
        add_diagnostic(diagnostics, "Engine 1 - OFP Parser", "WARNING", "FIR identifiers", "Unknown", "FIR matching may be UNKNOWN unless FIR data is present in the OFP or NOTAM.")
    if not route_segments:
        add_diagnostic(diagnostics, "Engine 1 - OFP Parser", "WARNING", "Airway sequence such as W112 TUSLI DUMAN", "No airway segments parsed", "Airway-specific matching may be UNKNOWN or rely on waypoint text only.")

    unmatched_route_waypoints = sorted(route_waypoints - coord_waypoints)
    coordinates_without_route = sorted(coord_waypoints - route_waypoints)
    eto_without_route = sorted(eto_waypoints - route_waypoints)
    eto_missing_route_waypoints = sorted(route_waypoints - eto_waypoints)
    fir_assigned = int(route_df["fir"].fillna("").astype(str).ne("").sum()) if not route_df.empty and "fir" in route_df else 0
    planned_fl_assigned = int(route_df["planned_fl"].notna().sum()) if not route_df.empty and "planned_fl" in route_df else 0

    status = {
        "Source": "✓ PDF extracted successfully" if source_label == "PDF text extracted" else "✓ Manual OFP text used",
        "Route waypoints extracted": quality_summary.get("Route waypoints extracted", len(route_waypoints)),
        "Coordinates matched": quality_summary.get("Coordinates matched", len(route_waypoints & coord_waypoints)),
        "ETO matched": quality_summary.get("ETO matched", len(route_waypoints & eto_waypoints)),
        "FIR assigned": quality_summary.get("FIR assigned", fir_assigned),
        "Planned FL assigned": quality_summary.get("FL assigned", planned_fl_assigned),
        "Unmatched route waypoints": ", ".join(unmatched_route_waypoints) if unmatched_route_waypoints else "None",
        "Coordinates without route match": ", ".join(coordinates_without_route) if coordinates_without_route else "None",
        "ETO without route match": ", ".join(eto_without_route) if eto_without_route else "None",
        "Route waypoints missing ETO": ", ".join(eto_missing_route_waypoints) if eto_missing_route_waypoints else "None",
        "Coordinates found": len(coord_df),
        "FIRs found": len(metadata.get("fir_boundaries", [])),
    }
    passed = not any(row["Severity"] == "CRITICAL" for row in diagnostics)
    return ofp_table, pd.DataFrame(diagnostics), passed, status


def engine2_notam_output(notams: list[NotamRestriction], notam_text: str) -> tuple[pd.DataFrame, pd.DataFrame, bool, pd.DataFrame]:
    rows = []
    debug_rows = []
    diagnostics = []
    if not notam_text.strip():
        add_diagnostic(diagnostics, "Engine 2 - NOTAM Parser", "CRITICAL", "Complete NOTAM bulletin text", "No NOTAM text found", "The NOTAM parser has no source bulletin to inspect.")
    if not notams:
        add_diagnostic(diagnostics, "Engine 2 - NOTAM Parser", "CRITICAL", "At least one NOTAM record", "No NOTAM records detected", "The bulletin could not be split into parseable NOTAM records.")

    for notam in notams:
        fields = notam.fields or {}
        rows.append({
            "NOTAM ID": unknown_if_empty(notam.reference),
            "Q Code": unknown_if_empty(notam.q_code),
            "FIR": unknown_if_empty(notam.fir),
            "Restriction Type": unknown_if_empty(notam.restriction_type),
            "Geometry Type": unknown_if_empty(notam.geometry_type),
            "Airways": unknown_if_empty(notam.mentioned_airways),
            "Waypoints": unknown_if_empty(notam.mentioned_waypoints),
            "Segments": f"{notam.segment_start}-{notam.segment_end}" if notam.segment_start and notam.segment_end else "Unknown",
            "Polygon vertices": f"{len(notam.polygon_points)} points" if notam.polygon_points else "Unknown",
            "Circle radius": f"{notam.radius_nm:g} NM" if notam.radius_nm else "Unknown",
            "Lower FL": unknown_if_empty(notam.fl_min),
            "Upper FL": unknown_if_empty(notam.fl_max),
            "Start UTC": notam.start_utc_text or "Unknown",
            "End UTC": notam.end_utc_text or "Unknown",
            "Schedule": unknown_if_empty(notam.schedule_text),
            "Raw Text": notam.raw_text,
            "Parsing Status": notam.parsing_status,
            "Review Reason": notam.review_reason,
        })
        debug_rows.append({
            "Raw NOTAM": notam.raw_text,
            "Field Q": fields.get("Q", "Unknown") or "Unknown",
            "Field A": fields.get("A", "Unknown") or "Unknown",
            "Field B": fields.get("B", "Unknown") or "Unknown",
            "Field C": fields.get("C", "Unknown") or "Unknown",
            "Field D": fields.get("D", "Unknown") or "Unknown",
            "Field E": fields.get("E", "Unknown") or "Unknown",
            "Field F": fields.get("F", "Unknown") or "Unknown",
            "Field G": fields.get("G", "Unknown") or "Unknown",
            "Parsed FIR": notam.fir or "Unknown",
            "Q_Code": notam.q_code or "Unknown",
            "Traffic": notam.traffic or "Unknown",
            "Purpose": notam.purpose or "Unknown",
            "Scope": notam.scope or "Unknown",
            "Q Reference": notam.q_reference or "Unknown",
            "Q Radius NM": notam.q_radius_nm if notam.q_radius_nm is not None else "Unknown",
            "Parsed time": f"{notam.start_utc_text or 'Unknown'} - {notam.end_utc_text or 'Unknown'}",
            "Parsed FL": f"{notam.fl_min if notam.fl_min is not None else 'Unknown'} - {notam.fl_max if notam.fl_max is not None else 'Unknown'}",
            "Restriction Type": notam.restriction_type,
            "Geometry Type": notam.geometry_type,
            "Parsed polygon": f"{len(notam.polygon_points)} coordinate pairs" if notam.polygon_points else "Unknown",
            "Parsed radius": f"{notam.radius_nm:g} NM {notam.radius_center}" if notam.radius_nm and notam.radius_center else "Unknown",
            "Parsed airway/segment": (
                f"{','.join(notam.mentioned_airways or [])} {notam.segment_start}-{notam.segment_end}".strip()
                if notam.segment_start and notam.segment_end
                else (", ".join(notam.mentioned_airways or []) or "Unknown")
            ),
            "Review reason": notam.review_reason or "",
        })
        for reason in [part.strip() for part in (notam.review_reason or "").split(";") if part.strip()]:
            add_diagnostic(diagnostics, "Engine 2 - NOTAM Parser", "WARNING", "Structured NOTAM field", "Unknown", f"{notam.reference}: {reason}")

    table = pd.DataFrame(rows, columns=[
        "NOTAM ID", "Q Code", "FIR", "Start UTC", "End UTC", "Schedule", "Lower FL", "Upper FL",
        "Restriction Type", "Geometry Type", "Polygon vertices", "Circle radius", "Airways", "Waypoints", "Segments",
        "Parsing Status", "Review Reason", "Raw Text",
    ])
    debug_table = pd.DataFrame(debug_rows)
    passed = not any(row["Severity"] == "CRITICAL" for row in diagnostics) and all(
        notam.parsing_status == "Parsed Successfully" for notam in notams
    )
    return table, pd.DataFrame(diagnostics), passed, debug_table


def notam_analysis_result(notam: NotamRestriction, analyses: list[dict]) -> dict:
    for analysis in analyses:
        if analysis["notam"] is notam:
            return analysis["result"]
    return {}


def engine3_matching_output(notams: list[NotamRestriction], analyses: list[dict], route_df: pd.DataFrame, route_segments: list[dict]) -> pd.DataFrame:
    route_waypoints = set(route_df["waypoint"].dropna().astype(str)) if not route_df.empty and "waypoint" in route_df else set()
    route_airways = {segment["airway"] for segment in route_segments if segment.get("airway")}
    route_firs = set(route_df["fir"].dropna().astype(str)) if not route_df.empty and "fir" in route_df else set()
    rows = []
    for notam in notams:
        result = notam_analysis_result(notam, analyses)
        if notam.segment_start and notam.segment_end:
            waypoint_match = notam.segment_start in route_waypoints and notam.segment_end in route_waypoints
        elif notam.mentioned_waypoints:
            waypoint_match = any(wp in route_waypoints for wp in notam.mentioned_waypoints)
        else:
            waypoint_match = None
        airway_match = None if not notam.mentioned_airways else any(awy in route_airways for awy in notam.mentioned_airways)
        fir_match = None if not notam.fir or not route_firs else notam.fir in route_firs
        polygon_match = result.get("geo_match") if notam.restriction_type == "Polygon" else None
        radius_match = result.get("geo_match") if notam.restriction_type == "Radius" else None
        fl_match = result.get("level_match") if result.get("flight_level") is not None else None
        time_match = result.get("time_match") if result.get("entry_time") and result.get("exit_time") else None
        rows.append({
            "NOTAM ID": notam.reference,
            "Waypoint Match": yes_no_unknown(waypoint_match),
            "Airway Match": yes_no_unknown(airway_match),
            "FIR Match": yes_no_unknown(fir_match),
            "Polygon Match": yes_no_unknown(polygon_match),
            "Radius Match": yes_no_unknown(radius_match),
            "FL Match": yes_no_unknown(fl_match),
            "Time Match": yes_no_unknown(time_match),
        })
    return pd.DataFrame(rows)


def approximate_route_length_nm(route: Optional[LineString]) -> str:
    if not route:
        return "Unknown"
    return f"{route.length * 60:.1f} NM approx"


def restriction_distance_nm(notam: NotamRestriction, route: Optional[LineString]) -> str:
    if not route:
        return "Unknown"
    if notam.polygon:
        return f"{route.distance(notam.polygon) * 60:.1f} NM approx"
    if notam.radius_center and notam.radius_nm:
        area = Point(notam.radius_center).buffer(notam.radius_nm / 60)
        return f"{route.distance(area) * 60:.1f} NM approx"
    return "Unknown"


def engine4_geospatial_output(notams: list[NotamRestriction], analyses: list[dict], route: Optional[LineString], route_df: pd.DataFrame) -> pd.DataFrame:
    if route is None or route_df.empty:
        return pd.DataFrame([{
            "Status": "Geospatial analysis skipped.",
            "Route length": "Unknown",
            "Number of route segments": 0,
            "Polygon intersections": "UNKNOWN",
            "Circle intersections": "UNKNOWN",
            "Impacted waypoint": "Unknown",
            "Impacted segment": "Unknown",
            "Distance from restriction": "Unknown",
            "Entry point": "Unknown",
            "Exit point": "Unknown",
        }])

    rows = []
    for notam in notams:
        result = notam_analysis_result(notam, analyses)
        rows.append({
            "NOTAM ID": notam.reference,
            "Route length": approximate_route_length_nm(route),
            "Number of route segments": max(len(route_df) - 1, 0),
            "Polygon intersections": yes_no_unknown(result.get("geo_match") if notam.restriction_type == "Polygon" else None),
            "Circle intersections": yes_no_unknown(result.get("geo_match") if notam.restriction_type == "Radius" else None),
            "Impacted waypoint": unknown_if_empty(result.get("affected")),
            "Impacted segment": unknown_if_empty(result.get("affected") if result.get("restriction_type") == "Airway/Segment" else ""),
            "Distance from restriction": restriction_distance_nm(notam, route),
            "Entry point": fmt_dt(result.get("entry_time")) if result.get("entry_time") else "Unknown",
            "Exit point": fmt_dt(result.get("exit_time")) if result.get("exit_time") else "Unknown",
        })
    return pd.DataFrame(rows)


def engine5_time_output(analyses: list[dict], etd: datetime, safety_margin: int) -> pd.DataFrame:
    rows = []
    for analysis in analyses:
        notam = analysis["notam"]
        result = analysis["result"]
        shift = calculate_etd_options(result, notam, etd, safety_margin) if result.get("entry_time") and result.get("exit_time") else {}
        time_difference = "Unknown"
        if result.get("entry_time") and result.get("exit_time") and notam.active_start and notam.active_end:
            if result.get("time_match"):
                time_difference = "Overlap"
            elif result["exit_time"] < notam.active_start:
                time_difference = fmt_duration(notam.active_start - result["exit_time"])
            elif result["entry_time"] > notam.active_end:
                time_difference = fmt_duration(result["entry_time"] - notam.active_end)
        rows.append({
            "NOTAM ID": notam.reference,
            "ETO": f"{fmt_dt(result.get('entry_time'))} - {fmt_dt(result.get('exit_time'))}",
            "NOTAM active period": f"{fmt_dt(notam.active_start)} - {fmt_dt(notam.active_end)}",
            "Safety margin": f"{safety_margin} min",
            "Impacted": yes_no_unknown(result.get("time_match") if result.get("entry_time") and result.get("exit_time") else None),
            "Advance required": fmt_duration(shift.get("minimum_advance")) if shift else "Unknown",
            "Delay required": fmt_duration(shift.get("minimum_delay")) if shift else "Unknown",
            "Time difference": time_difference,
        })
    return pd.DataFrame(rows)


def fl_position(current_fl: Optional[int], fl_min: Optional[int], fl_max: Optional[int]) -> str:
    if current_fl is None or fl_min is None or fl_max is None:
        return "Unknown"
    if current_fl < fl_min:
        return "Below restriction"
    if current_fl > fl_max:
        return "Above restriction"
    return "Inside restriction"


def engine6_fl_output(analyses: list[dict]) -> pd.DataFrame:
    rows = []
    for analysis in analyses:
        notam = analysis["notam"]
        result = analysis["result"]
        suggestions = suggest_flight_levels(result.get("flight_level"), notam.fl_min, notam.fl_max)
        rows.append({
            "NOTAM ID": notam.reference,
            "Planned FL": fmt_fl(result.get("flight_level")) if result.get("flight_level") is not None else "Unknown",
            "Restriction FL": notam.fl_text if notam.fl_text != "Not specified" else "Unknown",
            "Position": fl_position(result.get("flight_level"), notam.fl_min, notam.fl_max),
            "Suggested FL if available": ", ".join(f"FL{fl:03d}" for fl in suggestions) if suggestions else "Unknown",
        })
    return pd.DataFrame(rows)


def engine7_decision_output(analyses: list[dict], available_resolutions: list[dict]) -> pd.DataFrame:
    rows = []
    for analysis in analyses:
        notam = analysis["notam"]
        result = analysis["result"]
        shift = analysis.get("shift") or {}
        fl_options = [option["label"] for option in analysis["options"] if option.get("type") == "fl"]
        status, _, comment = recommendation_from_analysis(analysis)
        decision_status = "Manual Review" if result.get("review_required") else status
        if result.get("impacted") and result.get("restriction_type") == "Polygon":
            decision_status = "Polygon Conflict"
        elif result.get("impacted") and result.get("level_match"):
            decision_status = "Altitude Conflict"
        elif result.get("impacted") and result.get("time_match"):
            decision_status = "Timing Conflict"
        elif result.get("impacted"):
            decision_status = "Waypoint Conflict"
        rows.append({
            "NOTAM ID": notam.reference,
            "Decision": decision_status,
            "Suggested Delay": fmt_duration(shift.get("minimum_delay")) if shift else "Unknown",
            "Suggested Advance": fmt_duration(shift.get("minimum_advance")) if shift else "Unknown",
            "Suggested FL": ", ".join(fl_options) if fl_options else "Unknown",
            "Operational Comment": comment,
        })
    if not rows and available_resolutions:
        rows.append({
            "NOTAM ID": "All",
            "Decision": "No Impact",
            "Suggested Delay": "Unknown",
            "Suggested Advance": "Unknown",
            "Suggested FL": "Unknown",
            "Operational Comment": available_resolutions[0].get("detail", "No action required."),
        })
    return pd.DataFrame(rows)


def render_engine_debug(outputs: dict):
    with st.expander("Debug - Engine 1 OFP Parser", expanded=True):
        for label, value in outputs["engine1_status"].items():
            st.write(f"{label}: {value}")
        st.dataframe(outputs["engine1_ofp"], use_container_width=True, height=260)
        parser_tables = outputs.get("engine1_parser_tables", {})
        if parser_tables:
            st.write("OFP parser internal tables")
            table_tabs = st.tabs(["Route Sequence", "Coordinates", "ETO", "FIR", "FL"])
            table_tabs[0].dataframe(parser_tables.get("route_sequence", pd.DataFrame()), use_container_width=True)
            table_tabs[1].dataframe(parser_tables.get("coordinates", pd.DataFrame()), use_container_width=True)
            table_tabs[2].dataframe(parser_tables.get("eto", pd.DataFrame()), use_container_width=True)
            table_tabs[3].dataframe(parser_tables.get("fir", pd.DataFrame()), use_container_width=True)
            table_tabs[4].dataframe(parser_tables.get("fl", pd.DataFrame()), use_container_width=True)
        quality = outputs.get("engine1_quality", {})
        if quality:
            st.write("OFP parser quality")
            quality_summary = quality.get("summary", {})
            if quality_summary:
                cols = st.columns(5)
                for idx, (label, value) in enumerate(quality_summary.items()):
                    cols[idx % 5].metric(label, value)
            missing_tabs = st.tabs(["Missing coordinates", "Missing ETO", "Missing FIR", "Missing FL", "Coordinates not used"])
            missing_tabs[0].dataframe(quality.get("missing_coordinates", pd.DataFrame()), use_container_width=True)
            missing_tabs[1].dataframe(quality.get("missing_eto", pd.DataFrame()), use_container_width=True)
            missing_tabs[2].dataframe(quality.get("missing_fir", pd.DataFrame()), use_container_width=True)
            missing_tabs[3].dataframe(quality.get("missing_fl", pd.DataFrame()), use_container_width=True)
            missing_tabs[4].dataframe(quality.get("coordinates_not_used", pd.DataFrame()), use_container_width=True)
        if not outputs["engine1_diagnostics"].empty:
            st.dataframe(outputs["engine1_diagnostics"], use_container_width=True)

    with st.expander("Debug - Engine 2 NOTAM Parser", expanded=True):
        st.dataframe(outputs["engine2_notams"], use_container_width=True, height=240)
        if "engine2_debug" in outputs:
            st.write("ICAO PARSER OUTPUT")
            st.dataframe(outputs["engine2_debug"], use_container_width=True, height=260)
        if not outputs["engine2_diagnostics"].empty:
            st.dataframe(outputs["engine2_diagnostics"], use_container_width=True)

    with st.expander("Debug - Engine 3 Matching Engine", expanded=False):
        st.dataframe(outputs["engine3_matching"], use_container_width=True)

    with st.expander("Debug - Engine 4 Geospatial Engine", expanded=False):
        st.dataframe(outputs["engine4_geo"], use_container_width=True)

    with st.expander("Debug - Engine 5 Time Engine", expanded=False):
        st.dataframe(outputs["engine5_time"], use_container_width=True)

    with st.expander("Debug - Engine 6 FL Engine", expanded=False):
        st.dataframe(outputs["engine6_fl"], use_container_width=True)

    with st.expander("Debug - Engine 7 Decision Engine", expanded=False):
        st.dataframe(outputs["engine7_decision"], use_container_width=True)


def aggregate_available_resolutions(analyses: list[dict], etd: datetime) -> list[dict]:
    impacted = [analysis for analysis in analyses if analysis["result"].get("impacted")]
    if not impacted:
        return [{
            "type": "none",
            "label": "No Action Required",
            "detail": "No NOTAM impact was detected for the current ETD scenario.",
        }]

    seen = set()
    options = []
    max_advance = timedelta(0)
    max_delay = timedelta(0)
    for analysis in impacted:
        for option in analysis["options"]:
            if option["type"] == "fl":
                key = (option["type"], option["label"])
                if key not in seen:
                    seen.add(key)
                    options.append(option)
        shift = analysis.get("shift") or {}
        max_advance = max(max_advance, shift.get("minimum_advance", timedelta(0)))
        max_delay = max(max_delay, shift.get("minimum_delay", timedelta(0)))

    if max_advance > timedelta(0):
        options.append({
            "type": "etd_advance",
            "label": "Advance ETD",
            "subtitle": fmt_duration(max_advance),
            "new_etd": etd - max_advance,
            "detail": "Aircraft passes before NOTAM activation window with margin.",
        })
    if max_delay > timedelta(0):
        options.append({
            "type": "etd_delay",
            "label": "Delay ETD",
            "subtitle": fmt_duration(max_delay),
            "new_etd": etd + max_delay,
            "detail": "Aircraft passes after NOTAM expiration window with margin.",
        })
    if not options:
        options.append({
            "type": "reroute",
            "label": "Reroute Required",
            "detail": "No vertical or timing resolution was calculated for the affected NOTAMs.",
        })
    return options


def overlaps(start: datetime, end: datetime, active_start: datetime, active_end: datetime) -> bool:
    return start <= active_end and end >= active_start


def segment_slice(route_df: pd.DataFrame, start_wp: str, end_wp: str) -> Optional[pd.DataFrame]:
    waypoints = list(route_df["waypoint"])
    if start_wp not in waypoints or end_wp not in waypoints:
        return None
    start_idx, end_idx = waypoints.index(start_wp), waypoints.index(end_wp)
    low, high = sorted([start_idx, end_idx])
    return route_df.iloc[low:high + 1].copy()


def check_segment_impact(
    notam: NotamRestriction,
    navlog_df: pd.DataFrame,
    route_df: pd.DataFrame,
    route_segments: Optional[list[dict]] = None,
) -> dict:
    result = {"impacted": False, "reason": "The restricted segment was not found in the route."}
    if not notam.segment_start or not notam.segment_end:
        result["reason"] = "No segment restriction was detected in the NOTAM."
        return result

    affected = segment_slice(route_df if not route_df.empty else navlog_df, notam.segment_start, notam.segment_end)
    if affected is None or affected.empty:
        return result

    route_segments = route_segments or []
    route_airway_match = True
    if notam.airway:
        route_airway_match = route_contains_segment(
            route_segments,
            notam.airway,
            notam.segment_start,
            notam.segment_end,
        )

    entry_time = affected["eta"].iloc[0]
    exit_time = affected["eta"].iloc[-1]
    flight_level = int(round(affected["fl"].dropna().mean())) if affected["fl"].notna().any() else None
    geo_match = route_airway_match
    level_match = fl_matches(flight_level, notam.fl_min, notam.fl_max)
    time_match = bool(notam.active_start and notam.active_end and overlaps(entry_time, exit_time, notam.active_start, notam.active_end))

    return {
        "impacted": geo_match and level_match and time_match,
        "restriction_type": "Airway/Segment",
        "affected": f"{notam.segment_start}-{notam.segment_end}",
        "entry_time": entry_time,
        "exit_time": exit_time,
        "flight_level": flight_level,
        "geo_match": geo_match,
        "level_match": level_match,
        "time_match": time_match,
        "reason": (
            "Route crosses the restricted airway segment."
            if geo_match
            else "Waypoint pair is present, but the required airway segment was not found in the route string."
        ),
    }


def distance_to_route_point(point: Point, route_df: pd.DataFrame) -> int:
    distances = [
        math.hypot(point.x - row.lon, point.y - row.lat)
        for row in route_df.itertuples()
    ]
    return int(min(range(len(distances)), key=distances.__getitem__))


def check_polygon_impact(notam: NotamRestriction, route: Optional[LineString], route_df: pd.DataFrame) -> dict:
    if not notam.polygon:
        return {"impacted": False, "reason": "No polygon coordinates were detected in the NOTAM."}
    if not route or route_df.empty:
        return {"impacted": False, "reason": "Route geometry could not be built from waypoint coordinates."}

    geo_match = route.intersects(notam.polygon)
    if not geo_match:
        return {
            "impacted": False,
            "restriction_type": "Polygon",
            "affected": f"{len(notam.polygon_points)}-point polygon",
            "geo_match": False,
            "reason": "Route does not intersect the restricted polygon.",
        }

    intersection = route.intersection(notam.polygon)
    points = []
    if intersection.geom_type == "LineString":
        points = [Point(intersection.coords[0]), Point(intersection.coords[-1])]
    elif intersection.geom_type == "MultiLineString":
        first = list(intersection.geoms)[0]
        last = list(intersection.geoms)[-1]
        points = [Point(first.coords[0]), Point(last.coords[-1])]
    else:
        points = [intersection.representative_point(), intersection.representative_point()]

    entry_idx = distance_to_route_point(points[0], route_df)
    exit_idx = distance_to_route_point(points[-1], route_df)
    low, high = sorted([entry_idx, exit_idx])
    affected = route_df.iloc[low:high + 1]
    entry_time = affected["eta"].iloc[0]
    exit_time = affected["eta"].iloc[-1]
    flight_level = int(round(affected["fl"].dropna().mean())) if affected["fl"].notna().any() else None
    level_match = fl_matches(flight_level, notam.fl_min, notam.fl_max)
    time_match = bool(notam.active_start and notam.active_end and overlaps(entry_time, exit_time, notam.active_start, notam.active_end))

    return {
        "impacted": geo_match and level_match and time_match,
        "restriction_type": "Polygon",
        "affected": f"{len(notam.polygon_points)}-point polygon",
        "entry_time": entry_time,
        "exit_time": exit_time,
        "flight_level": flight_level,
        "geo_match": geo_match,
        "level_match": level_match,
        "time_match": time_match,
        "reason": "Route intersects the restricted polygon.",
    }


def check_radius_impact(notam: NotamRestriction, route: Optional[LineString], route_df: pd.DataFrame) -> dict:
    if not notam.radius_center or not notam.radius_nm:
        return {"impacted": False, "reason": "No radius restriction was detected in the NOTAM."}
    if not route or route_df.empty:
        return {"impacted": False, "reason": "Route geometry could not be built from waypoint coordinates."}

    radius_degrees = notam.radius_nm / 60
    area = Point(notam.radius_center).buffer(radius_degrees)
    pseudo = NotamRestriction(
        raw_text=notam.raw_text,
        restriction_type="Radius",
        airway=notam.airway,
        segment_start=notam.segment_start,
        segment_end=notam.segment_end,
        fl_min=notam.fl_min,
        fl_max=notam.fl_max,
        fl_text=notam.fl_text,
        active_start=notam.active_start,
        active_end=notam.active_end,
        polygon=area,
        polygon_points=[],
        reference=notam.reference,
        radius_center=notam.radius_center,
        radius_nm=notam.radius_nm,
    )
    result = check_polygon_impact(pseudo, route, route_df)
    result["restriction_type"] = "Radius"
    result["affected"] = f"{notam.radius_nm:g} NM radius"
    return result


def calculate_etd_shift(result: dict, notam: NotamRestriction, etd: datetime, safety_margin: int) -> dict:
    if not result.get("entry_time") or not result.get("exit_time") or not notam.active_start or not notam.active_end:
        return {}

    margin = timedelta(minutes=safety_margin)
    if not result.get("impacted"):
        return {
            "minimum_delay": timedelta(0),
            "minimum_advance": timedelta(0),
            "recommendation": "none",
            "recommended_etd": etd,
        }

    delay = max(timedelta(0), notam.active_end + margin - result["entry_time"])
    advance = max(timedelta(0), result["exit_time"] - (notam.active_start - margin))
    recommendation = "delay" if delay <= advance else "advance"
    recommended_shift = delay if recommendation == "delay" else -advance
    return {
        "minimum_delay": delay,
        "minimum_advance": advance,
        "recommendation": recommendation,
        "recommended_etd": etd + recommended_shift,
    }


def calculate_etd_options(result: dict, notam: NotamRestriction, etd: datetime, safety_margin: int) -> dict:
    return calculate_etd_shift(result, notam, etd, safety_margin)


def generate_fl_options(result: dict, notam: NotamRestriction) -> list[dict]:
    return flight_level_resolution_options(result, notam)


def generate_route_alternatives(result: dict, route_df: pd.DataFrame) -> list[dict]:
    return []


def fmt_dt(value: Optional[datetime]) -> str:
    return value.strftime("%H:%M UTC") if value else "Not detected"


def fmt_duration(delta: Optional[timedelta]) -> str:
    if delta is None:
        return "N/A"
    minutes = int(math.ceil(delta.total_seconds() / 60))
    return f"{minutes} min"


def fmt_fl(value: Optional[int]) -> str:
    return f"{value:03d}" if value is not None else "Not detected"


def action_label(resolution: dict, shift: dict) -> str:
    if resolution.get("suggested_fl") is not None:
        return f"FL{resolution['suggested_fl']:03d}"
    if resolution.get("resolution") == "ETD Change Recommended" and shift:
        direction = shift.get("recommendation", "delay").title()
        return f"{direction} ETD {fmt_duration(shift.get('minimum_delay' if direction == 'Delay' else 'minimum_advance'))}"
    if resolution.get("resolution") == "Reroute Required":
        return "Reroute"
    return "No Action Required"


def backup_label(shift: dict) -> str:
    if not shift:
        return "Not available"
    if shift.get("recommendation") == "none":
        return "No ETD change"
    if shift.get("recommendation") == "delay":
        return f"Delay ETD {fmt_duration(shift.get('minimum_delay'))}"
    return f"Advance ETD {fmt_duration(shift.get('minimum_advance'))}"


def render_decision_card(result: dict, available_resolutions: list[dict], affected_count: int = 0):
    impacted = result.get("impacted")
    status = "IMPACTED" if impacted else "NOT IMPACTED"
    border = "#dc2626" if impacted else "#16a34a"
    background = "#fff7ed" if impacted else "#f0fdf4"
    count = len([option for option in available_resolutions if option.get("type") != "none"])
    summary = (
        f"{status} | {affected_count} NOTAM{'s' if affected_count != 1 else ''} affected | {count} resolution{'s' if count != 1 else ''}"
        if impacted
        else f"{status} | No action required"
    )

    st.markdown(
        f"""
        <div class="decision-card" style="border-left-color:{border}; background:{background};">
          <span class="decision-line">{summary}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt_notam_time(notam: NotamRestriction) -> str:
    if not notam.active_start or not notam.active_end:
        return "Time not detected"
    return f"{notam.active_start.strftime('%H%M')}-{notam.active_end.strftime('%H%M')}"


def affected_area_label(notam: NotamRestriction, result: dict) -> str:
    if result.get("restriction_type") == "Polygon":
        return "Polygon area"
    if result.get("restriction_type") == "Radius":
        return result.get("affected", "Radius area")
    return result.get("affected", "Affected area")


def render_affected_notams(affected: list[dict]):
    if not affected:
        return
    st.markdown("<div class='section-label'>AFFECTED NOTAMS</div>", unsafe_allow_html=True)
    for analysis in affected:
        notam = analysis["notam"]
        result = analysis["result"]
        st.markdown(
            f"""
            <div class="affected-notam">
              <strong>{notam.reference}</strong> - {affected_area_label(notam, result)} - {notam.fl_text} - {fmt_notam_time(notam)}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_route_alternatives(route_suggestion: RouteSuggestionResult, impacted: bool):
    if not impacted:
        return
    st.markdown("<div class='section-label'>ROUTE ALTERNATIVES TO VALIDATE IN LIDO</div>", unsafe_allow_html=True)
    if not route_suggestion.available:
        st.warning(route_suggestion.message)
        return
    for idx, candidate in enumerate(route_suggestion.candidates):
        st.write(f"Route {chr(65 + idx)}: {candidate}")


def render_compact_decision_ui(
    analyses: list[dict],
    available_resolutions: list[dict],
    route_suggestion: RouteSuggestionResult,
    safety_margin: int,
    baseline_etd: datetime,
    navlog_df: pd.DataFrame,
    coord_df: pd.DataFrame,
    route_df: pd.DataFrame,
    route_text: str,
    route_segments: list[dict],
    ofp_sections: dict,
    ofp_metadata: dict,
    status: dict,
):
    affected = [analysis for analysis in analyses if analysis["result"].get("impacted")]
    aggregate_result = {"impacted": bool(affected)}
    route_table = standardize_route_dataframe(route_df)
    result_table = results_dataframe(analyses)
    render_decision_card(aggregate_result, available_resolutions, len(affected))
    render_available_resolutions(available_resolutions, bool(affected))
    render_affected_notams(affected)
    render_parsing_status(status)
    render_route_alternatives(route_suggestion, bool(affected))

    st.markdown("<div class='section-label'>ROUTE TABLE</div>", unsafe_allow_html=True)
    st.dataframe(route_table, use_container_width=True, height=260)

    st.markdown("<div class='section-label'>NOTAM RESULTS</div>", unsafe_allow_html=True)
    st.dataframe(result_table, use_container_width=True, height=240)
    export_cols = st.columns(2)
    export_cols[0].download_button(
        "Export results CSV",
        result_table.to_csv(index=False).encode("utf-8"),
        file_name="notam_impact_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
    export_cols[1].download_button(
        "Export parsed route CSV",
        route_table.to_csv(index=False).encode("utf-8"),
        file_name="parsed_route.csv",
        mime="text/csv",
        use_container_width=True,
    )

    render_route_map(route_table, analyses)

    with st.expander("Operational Details"):
        rows = []
        for analysis in analyses:
            notam = analysis["notam"]
            result = analysis["result"]
            rows.append({
                "NOTAM": notam.reference,
                "Status": "Impacted" if result.get("impacted") else "Not impacted",
                "Type": result.get("restriction_type", notam.restriction_type),
                "Affected": result.get("affected", "Not established"),
                "Entry": fmt_dt(result.get("entry_time")),
                "Exit": fmt_dt(result.get("exit_time")),
                "FL": fmt_fl(result.get("flight_level")),
                "Restricted FL": notam.fl_text,
                "Active": f"{fmt_dt(notam.active_start)} - {fmt_dt(notam.active_end)}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with st.expander("NOTAM-by-NOTAM Analysis"):
        for analysis in analyses:
            notam = analysis["notam"]
            result = analysis["result"]
            st.write(f"**{notam.reference}** - {'Impacted' if result.get('impacted') else 'Not impacted'}")
            st.write(result.get("reason", "No reason available."))

    with st.expander("Dispatcher Calculations"):
        st.write(f"Safety margin: {safety_margin} min")
        for analysis in analyses:
            result = analysis["result"]
            st.write(
                f"{analysis['notam'].reference}: geography={bool(result.get('geo_match'))}, "
                f"FL={bool(result.get('level_match'))}, time={bool(result.get('time_match'))}"
            )

    with st.expander("Parsed OFP Data"):
        st.write(f"Baseline OFP ETD: {baseline_etd.strftime('%H:%M UTC')}")
        st.write(f"Route used: {route_text or 'Not extracted'}")
        st.write(f"Detected OFP sections: {', '.join(ofp_sections.keys()) if ofp_sections else 'None detected'}")
        st.json(ofp_metadata)
        st.dataframe(pd.DataFrame(route_segments), use_container_width=True)
        st.dataframe(navlog_df, use_container_width=True)
        st.dataframe(coord_df, use_container_width=True)
        if not route_df.empty:
            st.dataframe(route_df, use_container_width=True)

    with st.expander("Raw Parsed NOTAM Data"):
        st.dataframe(pd.DataFrame([analysis["notam"].__dict__ for analysis in analyses]), use_container_width=True)


def render_parsing_status(status: dict):
    st.markdown("<div class='section-label'>PARSING STATUS</div>", unsafe_allow_html=True)
    items = list(status.items())
    cols = st.columns(5)
    for idx, (label, value) in enumerate(items):
        cols[idx % 5].metric(label, value)


def render_route_map(route_table: pd.DataFrame, analyses: list[dict]):
    coord_rows = route_table.dropna(subset=["Latitude", "Longitude"]) if not route_table.empty else pd.DataFrame()
    if coord_rows.empty:
        st.info("Map unavailable: waypoint coordinates missing. Polygon/radius analysis is disabled; text-based airway, waypoint, FIR, FL and timing checks remain available.")
        return
    try:
        import pydeck as pdk

        route_points = coord_rows.rename(columns={"Latitude": "lat", "Longitude": "lon"})
        layers = [
            pdk.Layer(
                "PathLayer",
                data=[{"path": route_points[["lon", "lat"]].values.tolist()}],
                get_path="path",
                get_width=3,
                get_color=[20, 80, 180],
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=route_points,
                get_position="[lon, lat]",
                get_radius=25000,
                get_fill_color=[20, 80, 180],
                pickable=True,
            ),
        ]
        view_state = pdk.ViewState(latitude=route_points["lat"].mean(), longitude=route_points["lon"].mean(), zoom=3)
        st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, tooltip={"text": "{Waypoint}"}))
    except Exception as exc:
        st.warning(f"Map could not be rendered: {exc}")


def render_available_resolutions(available_resolutions: list[dict], impacted: bool):
    st.markdown("<div class='section-label'>AVAILABLE RESOLUTIONS</div>", unsafe_allow_html=True)
    fl_options = [option for option in available_resolutions if option["type"] == "fl"]
    timing_options = [option for option in available_resolutions if option["type"] in {"etd_advance", "etd_delay"}]
    other_options = [option for option in available_resolutions if option["type"] not in {"fl", "etd_advance", "etd_delay"}]

    def render_card(option: dict, container):
        new_etd = option.get("new_etd")
        subtitle = option.get("subtitle", "")
        badge = {
            "fl": "Vertical",
            "etd_advance": "Timing",
            "etd_delay": "Timing",
            "reroute": "Lateral",
            "none": "Status",
        }.get(option.get("type"), "Option")
        etd_html = f"<div class='resolution-etd'>New ETD: {fmt_dt(new_etd)}</div>" if new_etd else ""
        subtitle_html = f"<div class='resolution-subtitle'>{subtitle}</div>" if subtitle else ""
        container.markdown(
            f"""
            <div class="resolution-card">
              <div class="resolution-title-row">
                <span class="resolution-title">{option['label']}</span>
                <span class="type-badge">Type: {badge}</span>
              </div>
              {subtitle_html}
              {etd_html}
              <div class="resolution-detail">{option['detail']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if fl_options:
        cols = st.columns(len(fl_options))
        for col, option in zip(cols, fl_options):
            render_card(option, col)

    if timing_options:
        timing_order = {"etd_advance": 0, "etd_delay": 1}
        timing_options = sorted(timing_options, key=lambda option: timing_order.get(option["type"], 99))
        cols = st.columns(2)
        for col, option in zip(cols, timing_options):
            render_card(option, col)

    if other_options:
        cols = st.columns(min(2, len(other_options)))
        for idx, option in enumerate(other_options):
            render_card(option, cols[idx % len(cols)])

    if impacted:
        st.info("Dispatcher decision required. Listed options are feasible alternatives, not an automatic ranking.")


def render_results(
    result: dict,
    notam: NotamRestriction,
    resolution: dict,
    shift: dict,
    available_resolutions: list[dict],
    safety_margin: int,
    baseline_etd: datetime,
    navlog_df: pd.DataFrame,
    coord_df: pd.DataFrame,
    route_df: pd.DataFrame,
    route_text: str,
    route_segments: list[dict],
):
    impacted = result.get("impacted", False)
    render_decision_card(result, available_resolutions)
    render_available_resolutions(available_resolutions, impacted)

    with st.expander("Operational Details"):
        cols = st.columns(4)
        cols[0].metric("Status", "Impacted" if impacted else "Not impacted")
        cols[1].metric("Restriction", result.get("restriction_type", notam.restriction_type))
        cols[2].metric("Segment / Area", result.get("affected", "Not established"))
        cols[3].metric("Current FL", fmt_fl(result.get("flight_level")))
        cols = st.columns(4)
        cols[0].metric("Entry", fmt_dt(result.get("entry_time")))
        cols[1].metric("Exit", fmt_dt(result.get("exit_time")))
        cols[2].metric("Restricted FL", notam.fl_text)
        cols[3].metric("NOTAM Active", f"{fmt_dt(notam.active_start)} - {fmt_dt(notam.active_end)}")

    with st.expander("Alternative Solutions"):
        fl_options = [option["label"] for option in available_resolutions if option["type"] == "fl"]
        advance_options = [option for option in available_resolutions if option["type"] == "etd_advance"]
        delay_options = [option for option in available_resolutions if option["type"] == "etd_delay"]
        cols = st.columns(3)
        cols[0].metric("FL options", ", ".join(fl_options) if fl_options else "No FL option")
        cols[1].metric("Delay option", fmt_duration(shift.get("minimum_delay")) if shift else "N/A")
        cols[2].metric("Advance option", fmt_duration(shift.get("minimum_advance")) if shift else "N/A")
        for option in [*fl_options, *[o["label"] for o in advance_options], *[o["label"] for o in delay_options]]:
            st.write(f"- {option}")
        if not impacted:
            st.info("No operational change is required for the current scenario.")

    with st.expander("Dispatcher Calculations"):
        cols = st.columns(4)
        cols[0].metric("Safety margin", f"{safety_margin} min")
        cols[1].metric("Geography", "Match" if result.get("geo_match") else "No match")
        cols[2].metric("FL", "Match" if result.get("level_match") else "No match")
        cols[3].metric("Time", "Overlap" if result.get("time_match") else "No overlap")
        if shift:
            cols = st.columns(3)
            cols[0].metric("Minimum delay", fmt_duration(shift.get("minimum_delay")))
            cols[1].metric("Minimum advance", fmt_duration(shift.get("minimum_advance")))
            cols[2].metric("Recommended ETD", fmt_dt(shift.get("recommended_etd")))
        st.write(
            f"Logic: the tool checks route/area, FL band, and shifted crossing time. "
            f"When impacted, it lists all calculated vertical and timing resolutions without selecting one. "
            f"Reason: {resolution.get('reason', result.get('reason', 'No additional reason detected.'))}"
        )

    with st.expander("Raw Parsed Data"):
        st.write("NOTAM", notam)
        st.write(f"Baseline OFP ETD: {baseline_etd.strftime('%H:%M UTC')}")
        st.write(f"Route used: {route_text or 'Not extracted'}")
        st.dataframe(pd.DataFrame(route_segments), use_container_width=True)
        st.dataframe(navlog_df, use_container_width=True)
        st.dataframe(coord_df, use_container_width=True)
        if not route_df.empty:
            st.dataframe(route_df[["waypoint", "eta", "offset", "fl", "lat", "lon"]], use_container_width=True)

    return


SAMPLES = {
    "ofp": {
        "notam": "",
        "full_ofp": (
            "LIDO OFP SAMPLE\n"
            "ETD 0500\n"
            "ATC ROUTE: BTO W271 NIXUK TESEN GOVSA ATSAV EGILO LADIX DONVO REBTI\n\n"
            "NAVLOG\n"
            "BTO      0550  FL350\n"
            "NIXUK    0620  FL350\n"
            "TESEN    0642  FL350\n"
            "GOVSA    0656  FL350\n"
            "ATSAV    0710  FL350\n"
            "EGILO    0725  FL350\n"
            "LADIX    0740  FL350\n"
            "DONVO    0755  FL350\n"
            "REBTI    0810  FL350\n\n"
            "WAYPOINT COORDINATES\n"
            "BTO 300000N 1200000E\n"
            "NIXUK 315000N 1233000E\n"
            "TESEN 323000N 1242000E\n"
            "GOVSA 324000N 1233000E\n"
            "ATSAV 330000N 1240000E\n"
            "EGILO 334000N 1243000E\n"
            "LADIX 340000N 1250000E\n"
            "DONVO 344000N 1253000E\n"
            "REBTI 350000N 1260000E"
        ),
        "etd": "0500",
    },
    "segment": {
        "notam": "W271 NIXUK-TESEN CLSD FL290-FL410 0613-0723",
        "full_ofp": (
            "ETD 0500\n"
            "ATC ROUTE: BTO W271 NIXUK TESEN LAMEN\n\n"
            "NAVLOG\n"
            "BTO      0550  FL350\n"
            "NIXUK    0620  FL350\n"
            "TESEN    0642  FL350\n"
            "LAMEN    0710  FL350\n\n"
            "WAYPOINT COORDINATES\n"
            "BTO 300000N 1200000E\n"
            "NIXUK 315000N 1233000E\n"
            "TESEN 323000N 1242000E\n"
            "LAMEN 335000N 1250000E"
        ),
        "etd": "0500",
    },
    "fl": {
        "notam": "A599 SEGMENT NIXUK-TESEN NOT AVBL FL380 AND ABOVE 0613-0723",
        "full_ofp": (
            "ETD 0500\n"
            "ROUTE: BTO A599 NIXUK TESEN LAMEN\n\n"
            "BTO 0550 FL390\n"
            "NIXUK 0620 FL390\n"
            "TESEN 0642 FL390\n"
            "LAMEN 0710 FL390\n\n"
            "BTO N300000 E1200000\n"
            "NIXUK N315000 E1233000\n"
            "TESEN N323000 E1242000\n"
            "LAMEN N335000 E1250000"
        ),
        "etd": "0500",
    },
    "polygon": {
        "notam": "B2345/26 TEMPORARY RESTRICTED AREA WITHIN N310000 E1225000 - N330000 E1225000 - N330000 E1245000 - N310000 E1245000 FL290-FL410 0613-0723",
        "full_ofp": (
            "ETD 0500\n"
            "ROUTE: BTO W271 NIXUK TESEN LAMEN\n\n"
            "BTO 0550 FL350\n"
            "NIXUK 0620 FL350\n"
            "TESEN 0642 FL350\n"
            "LAMEN 0710 FL350\n\n"
            "BTO 300000N 1200000E\n"
            "NIXUK 315000N 1233000E\n"
            "TESEN 323000N 1242000E\n"
            "LAMEN 335000N 1250000E"
        ),
        "etd": "0500",
    },
    "multi": {
        "notam": (
            "A1234/26 W271 NIXUK-TESEN CLSD FL290-FL410 0613-0723\n\n"
            "B2345/26 TEMPORARY RESTRICTED AREA WITHIN N310000 E1225000 - N330000 E1225000 - N330000 E1245000 - N310000 E1245000 FL290-FL410 0613-0723\n\n"
            "C3456/26 AREA WITHIN 25NM OF N323000 E1242000 SFC-UNL 0900-1000"
        ),
        "full_ofp": (
            "ETD 0500\n"
            "ATC ROUTE: BTO W271 NIXUK TESEN GOVSA ATSAV EGILO LADIX DONVO REBTI\n\n"
            "BTO      0550  FL350\n"
            "NIXUK    0620  FL350\n"
            "TESEN    0642  FL350\n"
            "GOVSA    0656  FL350\n"
            "ATSAV    0710  FL350\n"
            "EGILO    0725  FL350\n"
            "LADIX    0740  FL350\n"
            "DONVO    0755  FL350\n"
            "REBTI    0810  FL350\n\n"
            "BTO 300000N 1200000E\n"
            "NIXUK 315000N 1233000E\n"
            "TESEN 323000N 1242000E\n"
            "GOVSA 324000N 1233000E\n"
            "ATSAV 330000N 1240000E\n"
            "EGILO 334000N 1243000E\n"
            "LADIX 340000N 1250000E\n"
            "DONVO 344000N 1253000E\n"
            "REBTI 350000N 1260000E"
        ),
        "etd": "0500",
    },
}


def load_sample(key: str):
    for field, value in SAMPLES[key].items():
        st.session_state[field] = value
    st.session_state["etd_input"] = SAMPLES[key]["etd"]
    st.session_state["manual_route"] = ""
    st.session_state["baseline_etd"] = SAMPLES[key]["etd"]
    st.session_state["navlog_signature"] = navlog_signature(SAMPLES[key]["full_ofp"])


def load_notam_sample(key: str):
    st.session_state["notam"] = SAMPLES[key]["notam"]


def main():
    st.set_page_config(page_title="China NOTAM Timing Impact Tool", layout="wide")
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
        div[data-testid="stVerticalBlock"] { gap: 0.35rem; }
        .decision-card {
            border-left: 6px solid;
            border-radius: 4px;
            padding: 6px 10px;
            margin: 0.12rem 0 0.18rem 0;
            box-shadow: none;
            min-height: 34px;
            max-height: 40px;
            display: flex;
            align-items: center;
        }
        .decision-line {
            display: block;
            font-size: 0.92rem;
            font-weight: 800;
            letter-spacing: 0;
            line-height: 1.05;
            color: #0f172a;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .section-label {
            margin: 0.15rem 0 0.1rem 0;
            font-size: 0.9rem;
            font-weight: 850;
            color: #0f172a;
            letter-spacing: 0;
        }
        .resolution-card {
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 0.45rem 0.65rem;
            margin: 0.1rem 0 0.2rem 0;
            background: #ffffff;
            min-height: 88px;
        }
        .resolution-title-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.4rem;
        }
        .resolution-title {
            font-weight: 800;
            font-size: 0.98rem;
            color: #0f172a;
            line-height: 1.15;
        }
        .type-badge {
            border: 1px solid #cbd5e1;
            border-radius: 999px;
            padding: 1px 7px;
            background: #f8fafc;
            color: #334155;
            font-size: 0.72rem;
            font-weight: 750;
            white-space: nowrap;
        }
        .resolution-subtitle {
            margin-top: 0.12rem;
            font-size: 1.08rem;
            font-weight: 800;
            color: #1d4ed8;
            line-height: 1.15;
        }
        .resolution-etd {
            margin-top: 0.1rem;
            font-weight: 650;
            color: #334155;
        }
        .resolution-detail {
            margin-top: 0.1rem;
            color: #475569;
            font-size: 0.86rem;
            line-height: 1.2;
        }
        .affected-notam {
            border-left: 4px solid #f97316;
            background: #fff7ed;
            border-radius: 4px;
            padding: 5px 8px;
            margin: 0.12rem 0;
            color: #0f172a;
            font-size: 0.86rem;
            line-height: 1.2;
        }
        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.45rem 0.6rem;
        }
        div[data-testid="stMetricLabel"] { font-size: 0.78rem; }
        div[data-testid="stMetricValue"] { font-size: 1.05rem; }
        textarea { min-height: 120px !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Operational NOTAM Impact Assessment Tool")
    st.caption("China use case / decision-support prototype")
    st.warning(
        "Decision-support only. Dispatcher must validate NOTAM, OFP data, FL, timing and final operational decision."
    )

    sample_cols = st.columns(5)
    sample_cols[0].button("Load realistic OFP sample", on_click=load_sample, args=("ofp",), use_container_width=True)
    sample_cols[1].button("Load segment NOTAM sample", on_click=load_notam_sample, args=("segment",), use_container_width=True)
    sample_cols[2].button("Load segment + FL NOTAM sample", on_click=load_notam_sample, args=("fl",), use_container_width=True)
    sample_cols[3].button("Load polygon + FL NOTAM sample", on_click=load_notam_sample, args=("polygon",), use_container_width=True)
    sample_cols[4].button("Load multi-NOTAM bulletin sample", on_click=load_sample, args=("multi",), use_container_width=True)

    left, right = st.columns([1, 1.35])
    with left:
        notam_text = st.text_area("NOTAM text", key="notam", height=170)
        manual_route = st.text_input("Manual route input (optional)", key="manual_route")
    with right:
        uploaded_ofp_pdf = st.file_uploader("OFP PDF upload", type=["pdf"])
        pdf_text = extract_text_from_pdf(uploaded_ofp_pdf) if uploaded_ofp_pdf else ""
        if uploaded_ofp_pdf and not pdf_text:
            st.warning("PDF text not readable. Please use text fallback.")
        with st.expander("Manual OFP text fallback / debug mode"):
            fallback_ofp_text = st.text_area("Full OFP text", key="full_ofp", height=220)
        full_ofp_text = pdf_text or fallback_ofp_text
        extracted_etd = extract_etd_from_ofp(full_ofp_text) if full_ofp_text else None
        etd_default = st.session_state.get("etd", extracted_etd or "")
        etd_text = st.text_input("ETD UTC HHMM (optional override)", key="etd_input", value=etd_default)
        safety_margin = st.number_input("Safety margin in minutes", min_value=0, max_value=240, value=10, step=5)

    if st.button("Assess NOTAM impact", type="primary"):
        if uploaded_ofp_pdf and not pdf_text:
            st.warning("PDF text not readable. Please use text fallback.")
        extracted_etd = extract_etd_from_ofp(full_ofp_text)
        scenario_etd_text = etd_text.strip() or extracted_etd

        extracted_route = extract_route_from_ofp(full_ofp_text)
        route_text = extracted_route or manual_route.strip()

        etd = parse_hhmm(scenario_etd_text) if scenario_etd_text else None
        parse_etd = etd or parse_hhmm("0000")
        source_label = "PDF text extracted" if uploaded_ofp_pdf and pdf_text else "Fallback text used"

        baseline_etd = ensure_baseline_etd(full_ofp_text, parse_etd, extracted_etd) if full_ofp_text else parse_etd
        parsed_ofp = parse_ofp(full_ofp_text, baseline_etd, route_text)
        route_segments = parsed_ofp["route_segments"]
        original_navlog_df = parsed_ofp["navlog"]
        navlog_df = shift_navlog_to_etd(original_navlog_df, etd) if etd else original_navlog_df
        coord_df = parsed_ofp["coordinates"]
        route = parsed_ofp["route_geometry"]
        route_df = parsed_ofp["route_df"].copy()
        if etd and not route_df.empty and "offset" in route_df:
            route_df["eta"] = route_df["offset"].apply(lambda offset: etd + offset if pd.notna(offset) else pd.NaT)
            route_df["eto_utc"] = route_df["eta"].apply(lambda value: value.strftime("%H:%M UTC") if pd.notna(value) else "")
        notams = parse_notams(notam_text)

        engine1_ofp, engine1_diag, engine1_pass, engine1_status = engine1_ofp_output(
            parsed_ofp,
            source_label,
            full_ofp_text,
            route_text,
            scenario_etd_text or "",
        )
        engine2_notams, engine2_diag, engine2_pass, engine2_debug = engine2_notam_output(notams, notam_text)
        if scenario_etd_text and not etd:
            invalid_etd = pd.DataFrame([{
                "Engine": "Engine 1 - OFP Parser",
                "Severity": "CRITICAL",
                "Expected": "ETD in UTC HHMM format",
                "Found": scenario_etd_text,
                "Why parsing failed": "The ETD override or extracted ETD did not match HHMM format.",
            }])
            engine1_diag = pd.concat([engine1_diag, invalid_etd], ignore_index=True)
            engine1_pass = False

        outputs = {
            "engine1_status": engine1_status,
            "engine1_ofp": engine1_ofp,
            "engine1_parser_tables": parsed_ofp.get("parser_tables", {}),
            "engine1_quality": parsed_ofp.get("parser_quality", {}),
            "engine1_diagnostics": engine1_diag,
            "engine2_notams": engine2_notams,
            "engine2_diagnostics": engine2_diag,
            "engine2_debug": engine2_debug,
            "engine3_matching": pd.DataFrame(),
            "engine4_geo": pd.DataFrame(),
            "engine5_time": pd.DataFrame(),
            "engine6_fl": pd.DataFrame(),
            "engine7_decision": pd.DataFrame(),
        }

        if not engine1_pass or not engine2_pass:
            st.error("Pipeline stopped before impact analysis. Engine 1 or Engine 2 did not parse successfully.")
            render_engine_debug(outputs)
            return

        analyses = [
            analyze_notam(notam, navlog_df, route_df, route, route_segments, etd, int(safety_margin))
            for notam in notams
        ]

        available_resolutions = aggregate_available_resolutions(analyses, etd)
        route_suggestion = RouteEngine().suggest_alternatives(analyses, route_df)
        status = parsing_status(full_ofp_text, source_label, route_df, coord_df, route_segments, notams)
        outputs["engine3_matching"] = engine3_matching_output(notams, analyses, route_df, route_segments)
        outputs["engine4_geo"] = engine4_geospatial_output(notams, analyses, route, route_df)
        outputs["engine5_time"] = engine5_time_output(analyses, etd, int(safety_margin))
        outputs["engine6_fl"] = engine6_fl_output(analyses)
        outputs["engine7_decision"] = engine7_decision_output(analyses, available_resolutions)

        result_tab, debug_tab = st.tabs(["Pipeline Results", "Debug"])
        with result_tab:
            st.markdown("<div class='section-label'>ENGINE 7 - DECISION ENGINE</div>", unsafe_allow_html=True)
            st.dataframe(outputs["engine7_decision"], use_container_width=True, height=220)
            render_compact_decision_ui(
                analyses,
                available_resolutions,
                route_suggestion,
                int(safety_margin),
                baseline_etd,
                navlog_df,
                coord_df,
                route_df,
                route_text,
                route_segments,
                parsed_ofp["sections"],
                parsed_ofp["metadata"],
                status,
            )
        with debug_tab:
            render_engine_debug(outputs)


if __name__ == "__main__":
    main()
