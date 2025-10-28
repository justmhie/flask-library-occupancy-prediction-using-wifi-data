"""
AP MAC to Library Location Mapping
Maps Access Point MAC addresses to library locations
"""

# AP MAC to Library mapping
# Based on actual library AP deployments
AP_LOCATION_MAP = {
    # Miguel Pro Library
    'miguel_pro': [
        '10:F0:68:29:66:70',  # AP-202209-000005 (confirmed)
        '10:F0:68:28:3C:D0',  # AP-202209-000006 (confirmed)
        '10:F0:68:38:BD:40',  # AP-202209-000007 (confirmed)
        # Additional APs that may be in Miguel Pro
        '80:BC:37:20:8A:20',  # High usage AP (likely Miguel Pro)
        '10:F0:68:28:3A:80',  # Similar naming pattern
    ],

    # American Corner
    'american_corner': [
        '34:15:93:01:25:40',  # AP-202209-000062 (confirmed)
        'C0:C7:0A:32:62:10',  # High usage AP (likely American Corner)
        'C0:C7:0A:29:C0:70',  # Similar vendor
    ],

    # Gisbert 2nd Floor
    'gisbert_2nd': [
        '10:F0:68:29:68:20',  # AP-202209-000009 (confirmed)
        '5C:DF:89:07:69:30',  # High usage, likely Gisbert
        '5C:DF:89:07:62:A0',  # Similar vendor/naming
    ],

    # Gisbert 3rd Floor
    'gisbert_3rd': [
        '5C:DF:89:07:3A:80',  # AP 89 (confirmed)
        '5C:DF:89:07:58:D0',  # Same vendor series
        '00:33:58:29:52:F0',  # High usage AP
    ],

    # Gisbert 4th Floor
    'gisbert_4th': [
        '10:F0:68:29:21:60',  # AP-202209-000010 (confirmed)
        '80:BC:37:18:95:60',  # High usage AP
        '00:33:58:11:D3:D0',  # Likely Gisbert 4th
    ],

    # Gisbert 5th Floor
    'gisbert_5th': [
        '10:F0:68:28:76:50',  # AP-202209-000011 (confirmed)
    ],
}

# Library display names
LIBRARY_NAMES = {
    'miguel_pro': 'Miguel Pro Library',
    'gisbert_2nd': 'Gisbert 2nd Floor',
    'american_corner': 'American Corner',
    'gisbert_3rd': 'Gisbert 3rd Floor',
    'gisbert_4th': 'Gisbert 4th Floor',
    'gisbert_5th': 'Gisbert 5th Floor',
}

def get_location_from_ap(ap_mac):
    """Get library location from AP MAC address"""
    for location, ap_list in AP_LOCATION_MAP.items():
        if ap_mac in ap_list:
            return location
    return 'unknown'

def get_all_locations():
    """Get list of all library locations"""
    return list(AP_LOCATION_MAP.keys())

def get_display_name(location_id):
    """Get display name for location"""
    return LIBRARY_NAMES.get(location_id, location_id.replace('_', ' ').title())
