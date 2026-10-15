"""Storage triage: evidence-backed suggestions for which data on /mmfs1 and
/mmfs2 can be deleted. Read-only by design -- this package never deletes or
modifies anything it inspects; it writes only its own sqlite file and report
directory. The PI performs every delete.

Deliberately imports nothing from `opym` (its `__init__` eagerly pulls in
napari-adjacent modules); the registry/triage/auto-triage/seglab schemas are
read as contracts, the same convention opym-dashboard follows.
See docs/STORAGE_TRIAGE_RULES.md for every rule and the incident behind it.
"""
