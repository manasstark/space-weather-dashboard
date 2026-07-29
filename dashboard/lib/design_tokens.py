"""Shared design tokens for the dark, monospace "operational terminal"
visual language originally built for the SW Operational Command Centre
(dashboard/lib/command_centre.py) — extracted here so any other
component (e.g. shared_ui.terminal_metric) can use the identical colors/
font without importing command_centre.py itself (which would create a
circular import, since command_centre.py already imports from
shared_ui.py) or silently drifting out of sync with a copy-pasted set
of hex values.

This resolves the "three competing visual identities" finding from the
product review: the Command Centre's Bloomberg-terminal look, the old
metric_card's literal Windows-95 beveled grey box used on every other
page, and Project Status's plainer third look. Making this module the
single source of truth means every future component reaches for one
real design language instead of inventing a fourth.
"""

BG = "#090c10"
PANEL_BG = "#0d1117"
BORDER = "#1c2530"
TEXT = "#c9d1d9"
MUTED = "#6e7681"
ACCENT = "#39d98a"      # positive/online status, section headers
AMBER = "#e3b341"       # warnings, moderate deviation
RED = "#f85149"         # critical, large deviation
BLUE = "#58a6ff"        # informational, links, LIVE status
MONO = "'JetBrains Mono', 'Fira Code', 'Consolas', 'Courier New', monospace"
