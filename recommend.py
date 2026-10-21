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
    'Garment Set': ['Sneakers', 'Cap/peaked', 'Bag', 'Socks'],

    # Formal / Suiting
    'Suit': ['Tie', 'Shirt', 'Belt', 'Other shoe'],
    'Blazer': ['Tie', 'Shirt', 'Belt', 'Bag', 'Other shoe'],

    # Upper body
    'Garment Upper body': ['Trousers', 'Shorts', 'Skirt', 'Bag', 'Sneakers'],
    'T-shirt': ['Trousers', 'Shorts', 'Sneakers', 'Cap/peaked', 'Bag'],
    'Top': ['Trousers', 'Shorts', 'Skirt', 'Bag', 'Cardigan', 'Sneakers'],
    'Vest top': ['Trousers', 'Shorts', 'Skirt', 'Cardigan', 'Sunglasses'],
    'Shirt': ['Trousers', 'Tie', 'Belt', 'Sneakers', 'Other shoe'],
    'Blouse': ['Skirt', 'Trousers', 'Earring', 'Bag', 'Necklace'],
    'Sweater': ['Trousers', 'Scarf', 'Boots', 'Bag'],
    'Cardigan': ['T-shirt', 'Trousers', 'Bag', 'Necklace'],
    'Hoodie': ['Trousers', 'Sneakers', 'Cap/peaked', 'Bag'],
    'Bodysuit': ['Trousers', 'Skirt', 'Cardigan', 'Bag'],

    # Lower body
    'Garment Lower body': ['Belt', 'T-shirt', 'Top', 'Sneakers', 'Boots'],
    'Trousers': ['Belt', 'Shirt', 'T-shirt', 'Sneakers', 'Bag'],
    'Jeans': ['Belt', 'T-shirt', 'Sneakers', 'Jacket', 'Bag'],
    'Shorts': ['T-shirt', 'Sandals', 'Sunglasses', 'Cap/peaked'],
    'Skirt': ['Top', 'Blouse', 'Heeled sandals', 'Bag', 'Earring'],
    'Leggings/Tights': ['Top', 'Sweater', 'T-shirt', 'Sneakers', 'Boots'],

    # Outerwear
    'Jacket': ['Scarf', 'Hat/beanie', 'Gloves', 'Bag', 'Boots'],
    'Coat': ['Scarf', 'Gloves', 'Hat/beanie', 'Boots', 'Bag'],

    # Swimwear
    'Swimwear': ['Sunglasses', 'Hat/brim', 'Bag', 'Sandals'],
    'Bikini': ['Sunglasses', 'Hat/brim', 'Sandals', 'Bag'],
    'Bikini top': ['Swimwear bottom', 'Sunglasses', 'Hat/brim', 'Sandals'],
    'Swimwear bottom': ['Bikini top', 'Sunglasses', 'Hat/brim', 'Sandals'],
    'Swimsuit': ['Sunglasses', 'Hat/brim', 'Sandals', 'Bag'],

    # Nightwear
    'Nightwear': ['Slippers', 'Robe', 'Socks'],
    'Pyjamas': ['Slippers', 'Robe', 'Socks'],
    'Pyjama set': ['Slippers', 'Robe', 'Socks'],

    # Footwear
    'Shoes': ['Socks', 'Bag', 'Belt'],
    'Sneakers': ['Socks', 'Cap/peaked', 'Bag'],
    'Boots': ['Trousers', 'Coat', 'Jacket', 'Socks', 'Bag'],
    'Sandals': ['Dress', 'Sunglasses', 'Bag', 'Hat/brim'],
    'Heeled sandals': ['Dress', 'Bag', 'Earring', 'Necklace'],

    # Accessories
    'Accessories': ['Bag', 'Scarf', 'Sunglasses', 'Earring'],
    'Bag': ['Wallet', 'Scarf', 'Sunglasses', 'Hat/beanie'],
    'Earring': ['Necklace', 'Bracelet', 'Ring', 'Bag'],
    'Necklace': ['Earring', 'Bracelet', 'Ring', 'Bag'],
    'Scarf': ['Coat', 'Jacket', 'Hat/beanie', 'Gloves'],
    'Underwear': ['Socks', 'Robe', 'Pyjama set'],
    'Socks & Tights': ['Sneakers', 'Boots', 'Slippers'],
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

        # Check explicit filter audience constraint
        filter_audience = arguments.get('filters', {}).get('audience')
        filter_aud_cluster = self._get_audience_cluster(filter_audience) if filter_audience else None

        # Check if query or primary results have formality indicators
        query_text = (arguments.get('query') or '').casefold()
        is_formal = any(w in query_text for w in ('evening', 'cocktail', 'formal', 'silk', 'gown', 'prom', 'wedding', 'suit', 'blazer'))
        if not is_formal:
            for r in results[:5]:
                p = self.product_map.get(r['id'])
                if p:
                    text = (str(p.get('name') or '') + ' ' + str(p.get('product_type') or '')).casefold()
                    if any(w in text for w in ('evening', 'cocktail', 'formal', 'silk', 'gown', 'blazer', 'suit')):
                        is_formal = True
                        break

        used_complement_types: Set[str] = set()

        # Two-pass recommendation generation:
        # Pass 1: Prioritize distinct complementary categories for a complete, coordinated outfit
        # Pass 2: Fill remaining slots up to 5 if needed
        for pass_num in (1, 2):
            if len(recommendations) >= 5:
                break

            for rank, primary_res in enumerate(results[:5]):
                if len(recommendations) >= 5:
                    break

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
                        AFFINITY_TABLE.get(primary_product.get('category', ''), ['Bag', 'Scarf', 'Sunglasses', 'Earring', 'Belt'])
                    )

                # Find best candidate across affinity types
                best_cand: Optional[Dict[str, Any]] = None
                best_cand_type: Optional[str] = None
                best_score = -1e9

                for t in target_types:
                    t_folded = t.casefold()
                    if pass_num == 1 and not (complement_request and complement_request.target_types):
                        if t_folded in used_complement_types:
                            continue

                    candidates = self.by_product_type.get(t_folded, [])
                    for cand in candidates:
                        cand_id = cand['id']
                        if cand_id in primary_ids or cand_id in selected_rec_ids:
                            continue

                        cand_price = cand['price']['amount']
                        if complement_request and complement_request.max_price is not None:
                            if cand_price > complement_request.max_price:
                                continue

                        cand_aud = str(cand.get('audience') or '').strip()
                        cand_col = str(cand.get('color') or '').strip()

                        # Invariant: Global audience filter compliance
                        if filter_aud_cluster and cand_aud and cand_aud not in filter_aud_cluster:
                            continue

                        # Invariant: Primary product audience compatibility
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
                                score += 15.0
                            elif cand_col in NEUTRAL_COLORS or primary_color in NEUTRAL_COLORS:
                                score += 6.0

                        # Formality / Style consistency
                        cand_name_lower = (cand.get('name') or '').casefold()
                        if is_formal:
                            if any(w in cand_name_lower for w in ('backpack', 'back pack', 'shopper', 'canvas', 'gym', 'sport', 'tote')):
                                score -= 30.0
                            elif any(w in cand_name_lower for w in ('clutch', 'evening', 'chain', 'silk', 'leather', 'sparkle', 'glitter', 'pearl', 'crossbody', 'shoulder bag')):
                                score += 20.0

                        # Rating and reviews
                        avg_r = cand['rating']['average']
                        if avg_r is not None:
                            score += avg_r * 2.0
                        score += min(5.0, cand['rating']['count'] / 50.0)

                        if score > best_score:
                            best_score = score
                            best_cand = cand
                            best_cand_type = t_folded

                    if best_cand:
                        break

                if best_cand:
                    selected_rec_ids.add(best_cand['id'])
                    if best_cand_type:
                        used_complement_types.add(best_cand_type)

                    comp_name = str(best_cand.get('name') or best_cand.get('product_type') or 'accessory')
                    comp_col = str(best_cand.get('color') or '')
                    comp_type = str(best_cand.get('product_type') or 'accessory')

                    if comp_col and comp_col.casefold() == primary_color.casefold():
                        reason = f"A matching {comp_col} {comp_name} ({comp_type}) to complement the selected {primary_name}."
                    elif comp_col:
                        reason = f"A stylish {comp_col} {comp_name} ({comp_type}) that coordinates with the selected {primary_name}."
                    else:
                        reason = f"A classic {comp_name} to complement the selected {primary_name}."

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
