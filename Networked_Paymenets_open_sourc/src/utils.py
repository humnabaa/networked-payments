"""Utility functions: SIC code mappings, industry categorization, quarter parsing."""

import yaml
import re
from pathlib import Path

# Full 2-digit SIC code to industry name mapping (UK SIC 2007)
SIC_INDUSTRY_NAMES = {
    0: "Unknown or Unclassified",
    1: "Crop and Animal Production",
    2: "Forestry and Logging",
    3: "Fishing and Aquaculture",
    5: "Mining of Coal",
    6: "Extraction of Crude Petroleum and Gas",
    7: "Mining of Metal Ores",
    8: "Other Mining and Quarrying",
    9: "Mining Support Services",
    10: "Food Products",
    11: "Beverages",
    12: "Tobacco Products",
    13: "Textiles",
    14: "Wearing Apparel",
    15: "Leather Products",
    16: "Wood Products",
    17: "Paper Products",
    18: "Printing and Reproduction",
    19: "Coke and Refined Petroleum",
    20: "Chemicals",
    21: "Pharmaceuticals",
    22: "Rubber and Plastic Products",
    23: "Other Non-Metallic Minerals",
    24: "Basic Metals",
    25: "Fabricated Metal Products",
    26: "Computer and Electronic Products",
    27: "Electrical Equipment",
    28: "Machinery and Equipment",
    29: "Motor Vehicles",
    30: "Other Transport Equipment",
    31: "Furniture",
    32: "Other Manufacturing",
    33: "Repair and Installation of Machinery",
    35: "Electricity, Gas, Steam",
    36: "Water Collection and Supply",
    37: "Sewerage",
    38: "Waste Collection and Treatment",
    39: "Remediation Activities",
    41: "Construction of Buildings",
    42: "Civil Engineering",
    43: "Specialised Construction",
    45: "Wholesale and Retail of Motor Vehicles",
    46: "Wholesale Trade",
    47: "Retail Trade",
    49: "Land Transport",
    50: "Water Transport",
    51: "Air Transport",
    52: "Warehousing and Transport Support",
    53: "Postal and Courier Activities",
    55: "Accommodation",
    56: "Food and Beverage Service",
    58: "Publishing Activities",
    59: "Motion Picture and TV",
    60: "Broadcasting",
    61: "Telecommunications",
    62: "Computer Programming and Consultancy",
    63: "Information Service Activities",
    64: "Financial Services (excl. Insurance)",
    65: "Insurance and Pension Funding",
    66: "Activities Auxiliary to Financial Services",
    68: "Real Estate Activities",
    69: "Legal and Accounting",
    70: "Head Offices and Management Consultancy",
    71: "Architecture and Engineering",
    72: "Scientific Research and Development",
    73: "Advertising and Market Research",
    74: "Other Professional and Technical",
    75: "Veterinary Activities",
    77: "Rental and Leasing",
    78: "Employment Activities",
    79: "Travel Agency and Tour Operator",
    80: "Security and Investigation",
    81: "Services to Buildings and Landscape",
    82: "Office Administration and Business Support",
    84: "Public Administration and Defence",
    85: "Education",
    86: "Human Health Activities",
    87: "Residential Care Activities",
    88: "Social Work Without Accommodation",
    90: "Creative Arts and Entertainment",
    91: "Libraries, Archives, Museums",
    92: "Gambling and Betting",
    93: "Sports and Recreation",
    94: "Activities of Membership Organisations",
    95: "Repair of Computers and Personal Goods",
    96: "Other Personal Service Activities",
    97: "Activities of Households as Employers",
    98: "Undifferentiated Goods and Services by Households",
    99: "Extraterritorial Organisations",
}

# Grouped industry names used in the paper (SIC ranges → label)
SIC_GROUPED_NAMES = {
    "Financial Services (64-66)": [64, 65, 66],
    "Wholesale Trade (46)": [46],
    "Manufacturing (10-33)": list(range(10, 34)),
    "Real Estate Activities (68)": [68],
    "Professional Services (69-71)": [69, 70, 71],
    "Retail Trade (47)": [47],
    "Construction (41-43)": [41, 42, 43],
    "Information & Communication (58-63)": [58, 59, 60, 61, 62, 63],
    "Administrative Services (77-82)": [77, 78, 79, 80, 81, 82],
    "Transportation & Storage (49-53)": [49, 50, 51, 52, 53],
}


def load_config(config_path="config/settings.yaml"):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_industry_name(sic_code):
    """Get industry name from SIC code."""
    return SIC_INDUSTRY_NAMES.get(sic_code, f"SIC {sic_code}")


def get_industry_category(sic_code, config):
    """Determine which category an industry belongs to based on SIC code."""
    categories = config.get("industry_categories", {})
    for cat_name, cat_info in categories.items():
        for sic_range in cat_info.get("sic_ranges", []):
            if len(sic_range) == 2 and sic_range[0] <= sic_code <= sic_range[1]:
                return cat_name
    return "Other Services"


def get_category_color(category, config):
    """Get color for an industry category."""
    categories = config.get("industry_categories", {})
    if category in categories:
        return categories[category]["color"]
    return "#999999"


def parse_quarter(quarter_str):
    """Parse quarter string like '2017-Q1' or '2017Q1' into (year, quarter) tuple."""
    match = re.match(r"(\d{4})[- ]?Q(\d)", str(quarter_str))
    if match:
        return int(match.group(1)), int(match.group(2))
    raise ValueError(f"Cannot parse quarter string: {quarter_str}")


def quarter_to_str(year, quarter):
    """Convert (year, quarter) tuple to string."""
    return f"{year}-Q{quarter}"


def quarter_to_index(quarter_str):
    """Convert quarter string to a sortable integer index."""
    year, q = parse_quarter(quarter_str)
    return year * 4 + q


def get_period_label(quarter_str, config):
    """Determine which analysis period a quarter belongs to."""
    periods = config.get("periods", {})
    q_idx = quarter_to_index(quarter_str)
    for period_name, (start, end) in periods.items():
        if quarter_to_index(start) <= q_idx <= quarter_to_index(end):
            return period_name
    return "unknown"
