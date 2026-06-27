"""Output schema and allowed enum values from problem_statement.md."""

OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

CLAIM_STATUSES = {"supported", "contradicted", "not_enough_information"}
SEVERITIES = {"none", "low", "medium", "high", "unknown"}
ISSUE_TYPES = {
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
}
RISK_FLAGS = {
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
}
CAR_PARTS = {
    "front_bumper",
    "rear_bumper",
    "door",
    "hood",
    "windshield",
    "side_mirror",
    "headlight",
    "taillight",
    "fender",
    "quarter_panel",
    "body",
    "unknown",
}
LAPTOP_PARTS = {
    "screen",
    "keyboard",
    "trackpad",
    "hinge",
    "lid",
    "corner",
    "port",
    "base",
    "body",
    "unknown",
}
PACKAGE_PARTS = {
    "box",
    "package_corner",
    "package_side",
    "seal",
    "label",
    "contents",
    "item",
    "unknown",
}


def parts_for_object(claim_object: str) -> set[str]:
    mapping = {
        "car": CAR_PARTS,
        "laptop": LAPTOP_PARTS,
        "package": PACKAGE_PARTS,
    }
    return mapping.get(claim_object, {"unknown"})


def normalize_bool(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return "true"
    if text in {"false", "0", "no"}:
        return "false"
    return text


def normalize_risk_flags(flags) -> str:
    if flags is None:
        return "none"
    if isinstance(flags, str):
        items = [f.strip() for f in flags.split(";") if f.strip()]
    elif isinstance(flags, list):
        items = [str(f).strip() for f in flags if str(f).strip()]
    else:
        items = []

    cleaned = []
    for item in items:
        if item == "none":
            continue
        if item in RISK_FLAGS:
            cleaned.append(item)

    return ";".join(dict.fromkeys(cleaned)) if cleaned else "none"


def normalize_supporting_ids(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, list):
        items = [str(v).strip() for v in value if str(v).strip()]
    else:
        items = [v.strip() for v in str(value).split(";") if v.strip()]
    if not items or items == ["none"]:
        return "none"
    return ";".join(items)


def clamp_enum(value: str, allowed: set[str], default: str = "unknown") -> str:
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if text in allowed:
        return text
    aliases = {
        "shattered": "glass_shatter",
        "shatter": "glass_shatter",
        "broken": "broken_part",
        "missing": "missing_part",
        "crushed": "crushed_packaging",
        "torn": "torn_packaging",
        "water": "water_damage",
        "liquid": "water_damage",
        "not_enough_info": "not_enough_information",
        "insufficient": "not_enough_information",
    }
    if text in aliases and aliases[text] in allowed:
        return aliases[text]
    return default if default in allowed else next(iter(allowed))
