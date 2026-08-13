import re

BRITISH_VARIANTS = {
    "colour": "color",
    "colours": "color",
    "flavour": "flavor",
    "flavours": "flavor",
    "behaviour": "behavior",
    "honour": "honor",
    "honours": "honor",
    "labour": "labor",
    "labours": "labor",
    "favourite": "favorite",
    "favourites": "favorite",
    "neighbour": "neighbor",
    "neighbours": "neighbor",
    "centre": "center",
    "metre": "meter",
    "theatre": "theater",
    "analyse": "analyze",
    "organise": "organize",
    "organised": "organized",
    "organising": "organizing",
    "defence": "defense",
    "licence": "license",
    "pretence": "pretense",
    "travelling": "traveling",
    "traveller": "traveler",
    "grey": "gray",
}

def normalize_concept_name(word: str) -> str:
    """
    Normalize a lexical concept candidate.

    Applies conservative plural reduction without damaging singular words
    ending in s, ss, us, is, ous, ics, or ness.
    """

    w = str(word or "").strip().lower()

    if not w:
        return ""

    # Remove leading articles.
    w = re.sub(
        r"^(a|an|the)\s+",
        "",
        w,
    )

    # Remove unsupported punctuation while preserving spaces and hyphens.
    w = re.sub(
        r"[^a-z0-9\- ]+",
        "",
        w,
    )

    w = re.sub(
        r"\s+",
        " ",
        w,
    ).strip()

    if not w:
        return ""

    # Normalize spelling before plural handling.
    if w in BRITISH_VARIANTS:
        w = BRITISH_VARIANTS[w]

    # Words that resemble plurals but are normally singular or invariant.
    invariant_words = {
        "species",
        "series",
        "news",
        "gas",
    }

    if w not in invariant_words:
        if w.endswith("ies") and len(w) > 4:
            # properties → property
            w = w[:-3] + "y"

        elif w.endswith(("ches", "shes", "xes", "zes")):
            # branches → branch; boxes → box
            w = w[:-2]

        elif w.endswith("sses"):
            # classes → class; glasses → glass
            w = w[:-2]

        elif (
            w.endswith("s")
            and len(w) > 3
            and not w.endswith((
                "ss",
                "us",
                "is",
                "ous",
                "ics",
                "ness",
            ))
        ):
            # lemons → lemon, colours → colour
            w = w[:-1]

    # Apply spelling normalization again after singularization.
    if w in BRITISH_VARIANTS:
        w = BRITISH_VARIANTS[w]

    return w.strip()