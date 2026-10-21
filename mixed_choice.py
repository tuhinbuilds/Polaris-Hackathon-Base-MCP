"""Mixed-choice shopping query parser and candidate interleaver."""
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

GARMENT_KEYWORDS: Set[str] = {
    'jean', 'jeans', 'trousers', 'trouser', 'pants', 'pant', 'skirt', 'skirts', 'shorts', 'short',
    'dress', 'dresses', 't-shirt', 't-shirts', 'tshirt', 'tshirts', 'tee', 'tees', 'top', 'tops',
    'shirt', 'shirts', 'blouse', 'blouses', 'sweater', 'sweaters', 'cardigan', 'cardigans',
    'hoodie', 'hoodies', 'sweatshirt', 'sweatshirts', 'jacket', 'jackets', 'coat', 'coats',
    'blazer', 'blazers', 'suit', 'suits', 'vest', 'vests', 'sweatpants', 'joggers', 'jogger',
    'leggings', 'bra', 'underwear', 'pyjamas', 'pajamas', 'nightwear', 'robe', 'swimwear',
    'bikini', 'swimsuit', 'shoes', 'sneakers', 'trainers', 'boots', 'sandals', 'heels',
    'bag', 'bags', 'handbag', 'handbags', 'backpack', 'tote', 'scarf', 'scarves', 'hat', 'beanie',
    'belt', 'belts', 'tie', 'ties', 'sunglasses', 'necklace', 'earrings', 'polo', 'polos',
    'chinos', 'chino', 'jumpsuit', 'playsuit', 'pumps', 'flats', 'slippers'
}

COMPLEMENT_INDICATORS = re.compile(
    r'\b(?:with\s+(?:a\s+)?matching|with\s+(?:a\s+)?complementary|to\s+go\s+with|to\s+match\s+with|accompanying)\b',
    re.IGNORECASE
)

SPLIT_CONJUNCTIONS = re.compile(
    r'\s*(?:,\s*(?:and|or)\s*|,\s*|\s+(?:and|plus|\&|or)\s+)',
    re.IGNORECASE
)

PRICE_PATTERN = re.compile(
    r'\b(?:under|below|less\s+than|up\s+to|at\s+most|max(?:imum)?)\s+(?:inr\s*|rs\.?\s*|₹\s*)?(\d+)\b',
    re.IGNORECASE
)


@dataclass
class SubIntent:
    raw_span: str
    query_text: str
    inferred_color: Optional[str] = None
    inferred_max_price: Optional[float] = None
    inferred_min_price: Optional[float] = None


def parse_mixed_choices(
    query_str: str,
    catalog_colors: Set[str],
) -> Optional[List[SubIntent]]:
    """Decomposes a query into multiple shopping intents if it represents a valid mixed choice."""
    if not query_str or not query_str.strip():
        return None

    # Complement requests (e.g. 'dress with a matching bag') must NOT be treated as mixed choice
    if COMPLEMENT_INDICATORS.search(query_str):
        return None

    parts = [p.strip() for p in SPLIT_CONJUNCTIONS.split(query_str) if p.strip()]
    if len(parts) < 2:
        return None

    # Ellipsis resolution for 'or' alternatives (e.g. 'black or white dress' -> 'black dress', 'white dress')
    if len(parts) == 2 and re.search(r'\b(?:or)\b', query_str, re.IGNORECASE):
        words0 = set(re.findall(r'\w+', parts[0].casefold()))
        words1 = set(re.findall(r'\w+', parts[1].casefold()))
        g0 = words0 & GARMENT_KEYWORDS
        g1 = words1 & GARMENT_KEYWORDS
        if not g0 and g1:
            shared_noun = sorted(list(g1), key=lambda x: parts[1].casefold().find(x))[0]
            parts[0] = f"{parts[0]} {shared_noun}"

    # Validate that every part contains at least one distinct garment product type
    parsed_parts: List[Tuple[str, Set[str]]] = []
    for p in parts:
        words = set(re.findall(r'\w+', p.casefold()))
        garments = words & GARMENT_KEYWORDS
        if not garments:
            return None  # One part has no product type (e.g. 'black and white dress') -> abort split
        parsed_parts.append((p.strip(), garments))

    # Check if a global shared budget exists at the end of the full query
    global_budget_match = PRICE_PATTERN.search(query_str)
    shared_max_price = float(global_budget_match.group(1)) if global_budget_match else None

    sub_intents: List[SubIntent] = []
    for text_span, _ in parsed_parts:
        cleaned = text_span
        # Check local price
        local_match = PRICE_PATTERN.search(cleaned)
        if local_match:
            local_max_p = float(local_match.group(1))
            cleaned = cleaned[:local_match.start()] + ' ' + cleaned[local_match.end():]
        else:
            local_max_p = shared_max_price

        # Check local color
        inferred_col: Optional[str] = None
        for word in re.findall(r'\w+', cleaned.casefold()):
            if word.capitalize() in catalog_colors and not inferred_col:
                inferred_col = word.capitalize()
                break

        sub_intents.append(SubIntent(
            raw_span=text_span,
            query_text=cleaned.strip(),
            inferred_color=inferred_col,
            inferred_max_price=local_max_p,
        ))

    return sub_intents if len(sub_intents) >= 2 else None


def interleave_candidates(
    intent_candidate_lists: List[List[int]],
    limit: int,
    offset: int = 0,
) -> List[int]:
    """Deterministically interleaves candidates from multiple intents with catalog ID deduplication."""
    interleaved: List[int] = []
    seen_ids: Set[int] = set()

    max_len = max((len(lst) for lst in intent_candidate_lists), default=0)

    for rank in range(max_len):
        for intent_list in intent_candidate_lists:
            if rank < len(intent_list):
                cand_idx = intent_list[rank]
                if cand_idx not in seen_ids:
                    seen_ids.add(cand_idx)
                    interleaved.append(cand_idx)
                    if len(interleaved) >= offset + limit:
                        return interleaved[offset:offset + limit]

    return interleaved[offset:offset + limit]
