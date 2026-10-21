"""Natural language query preprocessing, constraint extraction, and typo handling."""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

STOPWORDS: Set[str] = {
    'a', 'an', 'the', 'for', 'with', 'in', 'of', 'at', 'to', 'from', 'on', 'by',
    'and', 'or', 'is', 'it', 'my', 'me', 'i', 'looking', 'look', 'want', 'need',
    'find', 'get', 'show', 'please', 'something', 's', 'pair', 'pairs', 'set',
    'piece', 'pieces', 'item', 'items'
}

SYNONYMS: Dict[str, str] = {
    'sneaker': 'sneakers',
    'trainer': 'trainers',
    'shades': 'sunglasses',
    'sunglass': 'sunglasses',
    'pant': 'trousers',
    'pants': 'trousers',
    'jogger': 'joggers',
    'sweatpant': 'sweatpants',
    'hoodies': 'hoodie',
    'suits': 'suit',
    'skirts': 'skirt',
    'jackets': 'jacket',
    'coats': 'coat',
    'dresses': 'dress',
    'sweaters': 'sweater',
    'cardigans': 'cardigan',
    'blouses': 'blouse',
    'shirts': 'shirt',
    'tees': 'tee',
    'jean': 'jeans',
    'boot': 'boots',
    'sandal': 'sandals',
    'heel': 'heels',
    'pajamas': 'pyjamas',
    'pajama': 'pyjamas',
    'pyjama': 'pyjamas',
}

OCCASION_INTENTS: List[Tuple[re.Pattern, List[str]]] = [
    (re.compile(r'\b(?:lazy\s+sunday|sunday\s+lounge|relaxing\s+at\s+home|cozy\s+loungewear|loungewear|lounge\s+wear|homewear)\b', re.IGNORECASE),
     ['pyjamas', 'joggers', 'hoodie', 'soft', 'sweatshirt']),
    (re.compile(r'\b(?:dinner\s+date|evening\s+out|date\s+night|dinner|evening)\b', re.IGNORECASE),
     ['dress', 'blazer', 'shirt', 'evening', 'party']),
    (re.compile(r'\b(?:summer\s+vacation|vacation|holiday|resort|beach\s+holiday)\b', re.IGNORECASE),
     ['summer', 'dress', 'linen', 'shorts', 'swimwear', 'sandals', 'sunglasses']),
    (re.compile(r'\b(?:business\s+casual|office\s+wear|workwear|work\s+attire|for\s+work|office\s+look)\b', re.IGNORECASE),
     ['blazer', 'trousers', 'shirt', 'blouse', 'tailored']),
    (re.compile(r'\b(?:wedding\s+guest|wedding\s+attire|wedding\s+outfit|wedding\s+look|wedding)\b', re.IGNORECASE),
     ['dress', 'cocktail', 'evening', 'suit', 'blazer', 'formal']),
    (re.compile(r'\b(?:warm\s+winter|winter\s+warm|winter\s+coat|winter\s+jacket|cold\s+winter\s+day|winter\s+day|cold\s+weather|freezing|chilly|cold\s+winter)\b', re.IGNORECASE),
     ['coat', 'jacket', 'sweater', 'wool', 'warm', 'padded']),
    (re.compile(r'\b(?:sleeping|sleep\s+clothes|sleepwear)\b', re.IGNORECASE),
     ['pyjamas', 'nightwear', 'robe']),
    (re.compile(r'\b(?:rainy\s+day|rain\s+weather|monsoon)\b', re.IGNORECASE),
     ['waterproof', 'jacket', 'coat']),
    (re.compile(r'\b(?:party\s+look|party\s+wear|night\s+out|clubbing)\b', re.IGNORECASE),
     ['party', 'dress', 'sequin', 'glitter', 'sparkly']),
    (re.compile(r'\b(?:gym\s+wear|workout|fitness|running)\b', re.IGNORECASE),
     ['sport', 'running', 'tights']),
    (re.compile(r'\b(?:beach\s+wear|swimming|poolside)\b', re.IGNORECASE),
     ['swimwear', 'sandals', 'shorts']),
    (re.compile(r'\b(?:formal\s+suit|formal\s+wear|formal\s+outfit|formal)\b', re.IGNORECASE),
     ['suit', 'blazer', 'trousers', 'tailored']),
    (re.compile(r'\b(?:retro|vintage)\b', re.IGNORECASE),
     ['retro', 'vintage']),
]

GENERIC_SHOPPING_WORDS: Set[str] = {
    'outfit', 'outfits', 'clothes', 'clothing', 'attire', 'apparel',
    'pieces', 'items', 'wear', 'look', 'style', 'day', 'home'
}

OCCASION_WORDS: Set[str] = {
    'dinner', 'evening', 'party', 'wedding', 'date', 'prom', 'cocktail',
    'sleeping', 'sleep', 'lazy', 'sunday', 'relaxing', 'gym', 'workout',
    'running', 'beach', 'poolside', 'office', 'work', 'casual', 'formal',
    'summer', 'winter', 'vacation', 'holiday', 'resort', 'loungewear',
    'clothes', 'outfit', 'apparel', 'wear', 'look', 'style', 'attire'
}

AUDIENCE_MAP: Dict[str, str] = {
    'women': 'Ladieswear',
    'woman': 'Ladieswear',
    'ladies': 'Ladieswear',
    'ladieswear': 'Ladieswear',
    'female': 'Ladieswear',
    'men': 'Menswear',
    'man': 'Menswear',
    'menswear': 'Menswear',
    'male': 'Menswear',
    'baby': 'Baby Sizes 50-98',
    'babies': 'Baby Sizes 50-98',
    'infant': 'Baby Sizes 50-98',
    'toddler': 'Children Sizes 92-140',
    'kids': 'Children Sizes 92-140',
    'kid': 'Children Sizes 92-140',
    'children': 'Children Sizes 92-140',
    'boys': 'Children Sizes 92-140',
    'boy': 'Children Sizes 92-140',
    'girls': 'Children Sizes 92-140',
    'girl': 'Children Sizes 92-140',
}

PRICE_PATTERNS = [
    (re.compile(r'\b(?:between|from)\s+(?:inr\s*|rs\.?\s*|₹\s*)?(\d+)\s+(?:and|to)\s+(?:inr\s*|rs\.?\s*|₹\s*)?(\d+)\b', re.IGNORECASE), 'range'),
    (re.compile(r'\b(?:under|below|less\s+than|up\s+to|at\s+most|max(?:imum)?)\s+(?:inr\s*|rs\.?\s*|₹\s*)?(\d+)\b', re.IGNORECASE), 'max'),
    (re.compile(r'\b(?:inr\s*|rs\.?\s*|₹\s*)?(\d+)\s*(?:and\s+under|or\s+less|and\s+below)\b', re.IGNORECASE), 'max'),
    (re.compile(r'\b(?:over|above|more\s+than|at\s+least|min(?:imum)?)\s+(?:inr\s*|rs\.?\s*|₹\s*)?(\d+)\b', re.IGNORECASE), 'min'),
    (re.compile(r'\b(?:inr\s*|rs\.?\s*|₹\s*)?(\d+)\s*(?:and\s+over|or\s+more|and\s+above)\b', re.IGNORECASE), 'min'),
]


@dataclass
class PreprocessedQuery:
    raw_query: str
    cleaned_text: str
    tokens: Tuple[str, ...]
    expanded_tokens: Tuple[str, ...]
    inferred_min_price: Optional[float] = None
    inferred_max_price: Optional[float] = None
    inferred_color: Optional[str] = None
    inferred_audience: Optional[str] = None
    is_impossible: bool = False


KNOWN_FASHION_WORDS: Set[str] = {
    'formal', 'casual', 'dinner', 'date', 'party', 'cocktail', 'office', 'work',
    'wedding', 'prom', 'gym', 'sleep', 'lazy', 'sunday', 'smart', 'evening',
    'summer', 'winter', 'spring', 'autumn', 'fall', 'cozy', 'soft', 'comfortable',
    'loungewear', 'sleepwear', 'activewear', 'basics', 'classic', 'vintage', 'retro',
    'cotton', 'linen', 'silk', 'wool', 'denim', 'leather', 'suede', 'lace',
    'suit', 'suits', 'blazer', 'blazers', 'dress', 'dresses', 'tshirt', 't-shirt',
    'shirt', 'shirts', 'trousers', 'trouser', 'pants', 'jeans', 'jean', 'skirt', 'skirts',
    'shorts', 'sweater', 'sweaters', 'cardigan', 'hoodie', 'jacket', 'coat', 'tie',
    'bag', 'bags', 'shoes', 'boots', 'sandals', 'sneakers', 'trainers', 'scarf', 'hat', 'belt',
    'tee', 'tees'
}


def damerau_levenshtein_1(word: str) -> Set[str]:
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def damerau_levenshtein_2(word: str) -> Set[str]:
    return set(e2 for e1 in damerau_levenshtein_1(word) for e2 in damerau_levenshtein_1(e1))


def correct_typo(token: str, vocab: Dict[str, int]) -> str:
    """Corrects token ONLY if it is completely absent from the catalog vocabulary."""
    if vocab.get(token, 0) > 0:
        return token
    if len(token) < 4:
        return token
    # Try distance 1
    d1 = [w for w in damerau_levenshtein_1(token) if vocab.get(w, 0) >= 1]
    if d1:
        return max(d1, key=lambda w: vocab[w])
    # Try distance 2 for longer words
    if len(token) >= 6:
        d2 = [w for w in damerau_levenshtein_2(token) if vocab.get(w, 0) >= 1]
        if d2:
            return max(d2, key=lambda w: vocab[w])
    return token


def preprocess_query(
    query_str: str,
    explicit_filters: Dict[str, Any],
    catalog_vocab: Dict[str, int],
    catalog_colors: Set[str],
) -> PreprocessedQuery:
    if not query_str or not query_str.strip():
        return PreprocessedQuery(
            raw_query="",
            cleaned_text="",
            tokens=(),
            expanded_tokens=(),
        )

    text = query_str.strip()

    # 1. Extract Price Constraints from Query
    inferred_min_price: Optional[float] = None
    inferred_max_price: Optional[float] = None

    for pat, kind in PRICE_PATTERNS:
        match = pat.search(text)
        if match:
            if kind == 'range':
                inferred_min_price = float(match.group(1))
                inferred_max_price = float(match.group(2))
            elif kind == 'max':
                inferred_max_price = float(match.group(1))
            elif kind == 'min':
                inferred_min_price = float(match.group(1))
            text = text[:match.start()] + ' ' + text[match.end():]
            break

    # Strip leftover INR/rupees labels
    text = re.sub(r'\b(?:inr|rupees?|rs\.?|₹)\b', ' ', text, flags=re.IGNORECASE)

    # 2. Check Constraint Precedence against Explicit Filters
    explicit_min = explicit_filters.get('min_price', 0)
    explicit_max = explicit_filters.get('max_price', float('inf'))

    eff_min = max(explicit_min, inferred_min_price or 0)
    eff_max = min(explicit_max, inferred_max_price if inferred_max_price is not None else float('inf'))

    is_impossible = eff_min > eff_max

    # 3. Occasion / Shopping Intent Detection
    expanded_intent_tokens: List[str] = []
    for pat, intent_words in OCCASION_INTENTS:
        if pat.search(text):
            expanded_intent_tokens.extend(intent_words)

    # 4. Tokenization & Token Processing
    # Handle possessives e.g. women's -> women
    cleaned_punc = re.sub(r"['’]s\b", '', text, flags=re.IGNORECASE)
    raw_tokens = tuple(re.findall(r'[a-z0-9]+', cleaned_punc.casefold()))

    inferred_color: Optional[str] = None
    inferred_audience: Optional[str] = None
    processed_tokens: List[str] = []
    audience_words_seen: Set[str] = set()

    for t in raw_tokens:
        # Check audience mapping
        if t in AUDIENCE_MAP and not inferred_audience:
            inferred_audience = AUDIENCE_MAP[t]
            audience_words_seen.add(t)

        # Check color match
        if t.capitalize() in catalog_colors and not inferred_color:
            inferred_color = t.capitalize()

        # Check synonym mapping
        resolved = SYNONYMS.get(t, t)

        # Check typo correction if unknown
        if catalog_vocab:
            resolved = correct_typo(resolved, catalog_vocab)

        processed_tokens.append(resolved)

    # 5. Filter Stopwords
    meaningful_tokens = [t for t in processed_tokens if t not in STOPWORDS]
    # If other product terms exist besides generic shopping terms, strip them
    if len(meaningful_tokens) > 1:
        without_generic = [t for t in meaningful_tokens if t not in GENERIC_SHOPPING_WORDS]
        if without_generic:
            meaningful_tokens = without_generic

    # If other product terms exist besides the audience word, strip the audience word from text tokens
    if len(meaningful_tokens) > 1 and audience_words_seen:
        meaningful_tokens = [t for t in meaningful_tokens if t not in audience_words_seen]

    if not meaningful_tokens and processed_tokens:
        meaningful_tokens = [t for t in processed_tokens if t not in {'s'}] or processed_tokens

    # Ensure uniqueness while preserving order
    seen = set()
    final_tokens = []
    for t in meaningful_tokens:
        if t not in seen:
            seen.add(t)
            final_tokens.append(t)

    # Deduplicate expanded intent tokens
    intent_deduped = []
    for t in expanded_intent_tokens:
        if t not in seen:
            seen.add(t)
            intent_deduped.append(t)

    return PreprocessedQuery(
        raw_query=query_str,
        cleaned_text=' '.join(final_tokens),
        tokens=tuple(final_tokens),
        expanded_tokens=tuple(intent_deduped),
        inferred_min_price=inferred_min_price,
        inferred_max_price=inferred_max_price,
        inferred_color=inferred_color,
        inferred_audience=inferred_audience,
        is_impossible=is_impossible,
    )
