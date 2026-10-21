"""Catalog-grounded complementary recommendation engine with category affinity graph."""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

COMPLEMENT_KEYWORDS_MAP: Dict[str, Tuple[str, ...]] = {
    'bag': ('Bag',),
    'bags': ('Bag',),
    'handbag': ('Bag',),
    'handbags': ('Bag',),
    'clutch': ('Bag',),
    'tote': ('Bag',),
    'backpack': ('Bag',),
    'purse': ('Bag',),
    'tie': ('Tie',),
    'ties': ('Tie',),
    'bowtie': ('Tie',),
    'cravat': ('Tie',),
    'scarf': ('Scarf',),
    'scarves': ('Scarf',),
    'shawl': ('Scarf',),
    'belt': ('Belt',),
    'belts': ('Belt',),
    'earring': ('Earring',),
    'earrings': ('Earring',),
    'necklace': ('Necklace',),
    'necklaces': ('Necklace',),
    'bracelet': ('Bracelet',),
    'bracelets': ('Bracelet',),
    'ring': ('Ring',),
    'rings': ('Ring',),
    'sunglasses': ('Sunglasses',),
    'shades': ('Sunglasses',),
    'hat': ('Hat/beanie', 'Hat/brim', 'Cap/peaked'),
    'hats': ('Hat/beanie', 'Hat/brim', 'Cap/peaked'),
    'beanie': ('Hat/beanie',),
    'beanies': ('Hat/beanie',),
    'cap': ('Cap/peaked',),
    'caps': ('Cap/peaked',),
    'gloves': ('Gloves',),
    'mittens': ('Gloves',),
    'shoe': ('Sneakers', 'Boots', 'Sandals', 'Heeled sandals', 'Pumps', 'Ballerinas'),
    'shoes': ('Sneakers', 'Boots', 'Sandals', 'Heeled sandals', 'Pumps', 'Ballerinas'),
    'sneakers': ('Sneakers',),
    'trainers': ('Sneakers',),
    'boots': ('Boots',),
    'sandals': ('Sandals',),
    'heels': ('Heeled sandals', 'Pumps'),
    'pumps': ('Pumps',),
    'socks': ('Socks',),
    'tights': ('Leggings/Tights',),
    'wallet': ('Wallet',),
    'watch': ('Watch',),
}

ALL_COMPLEMENT_KEYWORDS: Set[str] = set(COMPLEMENT_KEYWORDS_MAP.keys())

COMPLEMENT_SPLIT_REGEX = re.compile(
    r'\b(?:with\s+(?:a\s+)?(?:matching\s+|complementary\s+|coordinating\s+)?|to\s+go\s+with\s+(?:a\s+)?|to\s+match(?:\s+with)?\s+(?:a\s+)?|accompanied\s+by\s+(?:a\s+)?|(?:and|\+)\s+(?:a\s+)?(?:matching\s+|complementary\s+))',
    re.IGNORECASE
)

PRICE_REGEX = re.compile(
    r'\b(?:under|below|less\s+than|up\s+to|at\s+most|max(?:imum)?)\s+(?:inr\s*|rs\.?\s*|₹\s*)?(\d+)\b',
    re.IGNORECASE
)

AFFINITY_TABLE: Dict[str, List[str]] = {
    # Full body
    'Garment Full body': ['Bag', 'Earring', 'Necklace', 'Heeled sandals', 'Sunglasses', 'Belt'],
    'Dress': ['Bag', 'Earring', 'Necklace', 'Heeled sandals', 'Sunglasses', 'Belt'],
    'Jumpsuit/Playsuit': ['Bag', 'Earring', 'Necklace', 'Heeled sandals', 'Belt'],

    # Formal / Suiting
    'Suit': ['Tie', 'Shirt', 'Belt', 'Other shoe'],
    'Blazer': ['Tie', 'Shirt', 'Belt', 'Bag'],

    # Upper body
    'Garment Upper body': ['Trousers', 'Shorts', 'Skirt', 'Bag', 'Sneakers'],
    'T-shirt': ['Trousers', 'Shorts', 'Sneakers', 'Cap/peaked', 'Bag'],
    'Shirt': ['Trousers', 'Tie', 'Belt', 'Shoes'],
    'Blouse': ['Skirt', 'Trousers', 'Earring', 'Bag'],
    'Sweater': ['Trousers', 'Scarf', 'Boots', 'Bag'],
    'Cardigan': ['T-shirt', 'Trousers', 'Bag', 'Necklace'],
    'Hoodie': ['Trousers', 'Sneakers', 'Cap/peaked', 'Backpack'],

    # Lower body
    'Garment Lower body': ['Belt', 'T-shirt', 'Top', 'Sneakers', 'Boots'],
    'Trousers': ['Belt', 'Shirt', 'T-shirt', 'Sneakers'],
    'Jeans': ['Belt', 'T-shirt', 'Sneakers', 'Jacket'],
    'Shorts': ['T-shirt', 'Sandals', 'Sunglasses', 'Cap/peaked'],
    'Skirt': ['Top', 'Blouse', 'Heels', 'Bag', 'Earring'],

    # Outerwear
    'Jacket': ['Scarf', 'Hat/beanie', 'Gloves', 'Bag', 'Boots'],
    'Coat': ['Scarf', 'Gloves', 'Hat/beanie', 'Boots', 'Bag'],

    # Swimwear
    'Swimwear': ['Sunglasses', 'Hat/brim', 'Bag', 'Sandals'],
    'Bikini': ['Sunglasses', 'Hat/brim', 'Sandals', 'Bag'],
    'Swimsuit': ['Sunglasses', 'Hat/brim', 'Sandals', 'Bag'],

    # Nightwear
    'Nightwear': ['Slippers', 'Robe', 'Socks'],
    'Pyjamas': ['Slippers', 'Robe', 'Socks'],

    # Footwear
    'Shoes': ['Socks', 'Bag', 'Belt'],
    'Sneakers': ['Socks', 'Cap/peaked', 'Backpack'],
}

AUDIENCE_CLUSTERS: List[Set[str]] = [
    {'Ladieswear', 'Divided', 'Ladies Accessories', 'Lingeries/Tights'},
    {'Menswear', 'Men'},
    {'Children Sizes 92-140', 'Children Sizes 134-170', 'Children Accessories, Swimwear'},
    {'Baby Sizes 50-98'},
]

NEUTRAL_COLORS: Set[str] = {'Black', 'White', 'Grey', 'Dark Grey', 'Light Grey', 'Beige', 'Navy', 'Dark Blue', 'Silver'}


@dataclass
class ComplementRequest:
    raw_span: str
    target_types: List[str] = field(default_factory=list)
    max_price: Optional[float] = None
    is_matching: bool = False


def split_complement_query(query_str: str) -> Tuple[str, Optional[ComplementRequest]]:
    """Splits an input query into primary query and complement request if complement clause exists."""
    if not query_str or not query_str.strip():
        return query_str, None

    m = COMPLEMENT_SPLIT_REGEX.search(query_str)
    if not m:
        return query_str, None

    after = query_str[m.end():].strip()
    words = set(re.findall(r'\w+', after.casefold()))
    matched_kw = words & ALL_COMPLEMENT_KEYWORDS
    if not matched_kw:
        return query_str, None

    before = query_str[:m.start()].strip()
    price_m = PRICE_REGEX.search(after)
    max_p = float(price_m.group(1)) if price_m else None

    target_types: List[str] = []
    for kw in matched_kw:
        for t in COMPLEMENT_KEYWORDS_MAP[kw]:
            if t not in target_types:
                target_types.append(t)

    is_matching = bool(re.search(r'\b(?:matching|coordinate|coordinating|match)\b', query_str[m.start():m.end()], re.IGNORECASE))

    return before, ComplementRequest(
        raw_span=after,
        target_types=target_types,
        max_price=max_p,
        is_matching=is_matching,
    )


class CategoryAffinityRecommender:
    def __init__(self, products: List[Dict[str, Any]]):
        self.products = products
        self.product_map = {p['id']: p for p in products}

        # Fast group by product_type (case-folded)
        self.by_product_type: Dict[str, List[Dict[str, Any]]] = {}
        for p in self.products:
            pt = p.get('product_type')
            if pt:
                key = pt.casefold()
                if key not in self.by_product_type:
                    self.by_product_type[key] = []
                self.by_product_type[key].append(p)

        # Pre-sort each type pool by rating average and review count
        for key, items in self.by_product_type.items():
            items.sort(key=lambda x: (
                -(x['rating']['average'] or 0.0),
                -x['rating']['count'],
                x['id']
            ))

    def _get_audience_cluster(self, audience: Optional[str]) -> Optional[Set[str]]:
        if not audience:
            return None
        aud_clean = audience.strip()
        for cluster in AUDIENCE_CLUSTERS:
            if aud_clean in cluster:
                return cluster
        return None

    def generate_recommendations(
        self,
        results: List[Dict[str, Any]],
        arguments: Dict[str, Any],
        complement_request: Optional[ComplementRequest] = None,
    ) -> List[Dict[str, Any]]:
        """Generates 1 to 5 catalog-grounded complementary recommendations referencing primary products."""
        if not arguments.get('include_recommendations', False) or not results:
            return []

        primary_ids = {r['id'] for r in results}
        selected_rec_ids: Set[str] = set()
        recommendations: List[Dict[str, Any]] = []

        # Target number of recommendations is min(5, len(results))
        max_recs = min(5, len(results))

        for rank, primary_res in enumerate(results[:max_recs]):
            p_id = primary_res['id']
            primary_product = self.product_map.get(p_id)
            if not primary_product:
                continue

            primary_type = str(primary_product.get('product_type') or 'item')
            primary_name = str(primary_product.get('name') or primary_type)
            primary_color = str(primary_product.get('color') or '').strip()
            primary_aud = str(primary_product.get('audience') or '').strip()
            aud_cluster = self._get_audience_cluster(primary_aud)

            # Determine candidate types for this complement
            if complement_request and complement_request.target_types:
                target_types = complement_request.target_types
            else:
                target_types = AFFINITY_TABLE.get(
                    primary_product.get('product_type', ''),
                    AFFINITY_TABLE.get(primary_product.get('category', ''), ['Bag', 'Scarf', 'Sunglasses'])
                )

            # Find candidates across target types
            best_cand: Optional[Dict[str, Any]] = None
            best_score = -1e9

            for t in target_types:
                candidates = self.by_product_type.get(t.casefold(), [])
                for cand in candidates:
                    cand_id = cand['id']
                    # Invariant 1: Disjoint from primary results
                    if cand_id in primary_ids:
                        continue
                    # Invariant 2: Unique among recommendations
                    if cand_id in selected_rec_ids:
                        continue
                    # Invariant 3: Price constraint
                    cand_price = cand['price']['amount']
                    if complement_request and complement_request.max_price is not None:
                        if cand_price > complement_request.max_price:
                            continue

                    # Score candidate affinity
                    cand_aud = str(cand.get('audience') or '').strip()
                    cand_col = str(cand.get('color') or '').strip()

                    # Invariant 4: Audience compatibility (never cross gender/age demographics)
                    if aud_cluster and cand_aud and cand_aud not in aud_cluster:
                        continue

                    score = 0.0
                    # Audience match bonus
                    if cand_aud == primary_aud:
                        score += 25.0
                    elif aud_cluster and cand_aud in aud_cluster:
                        score += 15.0

                    # Color coordination bonus
                    if primary_color and cand_col:
                        if cand_col.casefold() == primary_color.casefold():
                            score += 12.0
                        elif cand_col in NEUTRAL_COLORS or primary_color in NEUTRAL_COLORS:
                            score += 5.0

                    # Rating and reviews
                    avg_r = cand['rating']['average']
                    if avg_r is not None:
                        score += avg_r * 2.0
                    score += min(5.0, cand['rating']['count'] / 50.0)

                    if score > best_score:
                        best_score = score
                        best_cand = cand

                if best_cand:
                    break

            if best_cand:
                selected_rec_ids.add(best_cand['id'])
                comp_name = str(best_cand.get('name') or best_cand.get('product_type') or 'accessory')
                comp_col = str(best_cand.get('color') or '')
                comp_type = str(best_cand.get('product_type') or 'accessory')

                # Generate concise, factual reason (1 to 300 characters)
                if comp_col and comp_col.casefold() == primary_color.casefold():
                    reason = f"A matching {comp_col} {comp_name} ({comp_type}) to complement the selected {primary_name}."
                elif comp_col:
                    reason = f"A stylish {comp_col} {comp_name} ({comp_type}) that coordinates with the selected {primary_name}."
                else:
                    reason = f"A classic {comp_name} to complement the selected {primary_name}."

                # Enforce schema constraint: length <= 300
                if len(reason) > 300:
                    reason = reason[:297] + "..."

                recommendations.append({
                    'id': best_cand['id'],
                    'for_product_id': p_id,
                    'reason': reason,
                })

                if len(recommendations) >= 5:
                    break

        return recommendations
