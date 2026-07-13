"""Space Weather Physics Engine — generalized opt-in feature registry.

Every research laboratory (IMF, Kp, AE) already used the same pattern
before this migration: a {display_name: function} registry plus a
{display_name: [prerequisite_names]} dependency map, so a researcher can
toggle each physics feature on/off independently while prerequisite
columns still get computed silently in the background when needed. This
module generalizes that existing pattern into one shared, tested
implementation — it does not replace or redesign it; each lab keeps its
own registry dict (its own available features), only the resolution
LOOP that walks a registry and applies requested features is now shared.
"""

import pandas as pd


def apply_requested_features(
    frame: pd.DataFrame,
    requested: dict[str, bool],
    function_registry: dict[str, callable],
    dependency_registry: dict[str, list[str]] = None,
) -> list[str]:
    """Applies every requested (True-valued) feature in `requested`,
    resolving prerequisites from `dependency_registry` first.

    For each requested feature, any prerequisite listed in
    `dependency_registry` that is NOT itself independently requested gets
    computed too (its columns are added to `frame`, but NOT included in
    the returned exposed-feature list) — so a feature's hidden
    prerequisites never leak into a model's feature set unless the
    researcher also explicitly enabled them.

    `dependency_registry` entries must list ALL transitively-required
    prerequisites, in the order they must run (this resolver is one
    level deep by design — it does not recurse — so a feature whose
    prerequisite has its own prerequisite must flatten the full chain in
    its own dependency_registry entry). This exactly preserves the
    resolution behavior every lab already had before this migration.

    Returns the list of exposed feature column names, in
    function_registry's iteration order (insertion order) — same
    convention every lab's own hand-rolled loop already followed.
    """
    dependency_registry = dependency_registry or {}
    exposed_columns: list[str] = []

    for name in function_registry:
        if not requested.get(name, False):
            continue
        for dependency in dependency_registry.get(name, []):
            if dependency not in exposed_columns and requested.get(dependency, False) is False:
                function_registry[dependency](frame)
        exposed_columns += function_registry[name](frame)

    return exposed_columns
