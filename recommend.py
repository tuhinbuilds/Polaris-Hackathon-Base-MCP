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
    'shoe': ('Sneakers', 'Boots', 'Sandals', 'Heeled sandals', 'Pumps', 'Ballerinas', 'Other shoe'),
    'shoes': ('Sneakers', 'Boots', 'Sandals', 'Heeled sandals', 'Pumps', 'Ballerinas', 'Other shoe'),
    'sneakers': ('Sneakers',),
    'trainers': ('Sneakers',),
    'boots': ('Boots',),
    'sandals': ('Sandals',),
    'heels': ('Heeled sandals', 'Pumps', 'Heels'),
    'pumps': ('Pumps',),
    'socks': ('Socks',),
    'tights': ('Leggings/Tights', 'Underwear Tights'),
    'wallet': ('Wallet',),
    'watch': ('Watch',),
    'jacket': ('Jacket', 'Blazer'),
    'jackets': ('Jacket', 'Blazer'),
    'blazer': ('Blazer',),
    'blazers': ('Blazer',),
    'cardigan': ('Cardigan',),
    'cardigans': ('Cardigan',),
    'sweater': ('Sweater',),
    'sweaters': ('Sweater',),
    'trousers': ('Trousers',),
    'pants': ('Trousers',),
    'skirt': ('Skirt',),
    'skirts': ('Skirt',),
    'shirt': ('Shirt',),
    'shirts': ('Shirt',),
    'blouse': ('Blouse',),
    'blouses': ('Blouse',),
    'top': ('Top', 'T-shirt', 'Blouse'),
    'tops': ('Top', 'T-shirt', 'Blouse'),
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
    'Dress': ['Bag', 'Earring', 'Necklace', 'Heeled sandals', 'Belt', 'Sunglasses'],
    'Jumpsuit/Playsuit': ['Bag', 'Earring', 'Necklace', 'Heeled sandals', 'Belt'],
    'Garment Set': ['Sneakers', 'Cap/peaked', 'Bag', 'Socks'],

    # Formal / Suiting
    'Suit': ['Tie', 'Shirt', 'Belt', 'Other shoe', 'Trousers'],
    'Blazer': ['Tie', 'Shirt', 'Belt', 'Other shoe', 'Trousers', 'Bag'],

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
    'Trousers': ['Belt', 'Shirt', 'T-shirt', 'Sneakers', 'Bag', 'Other shoe'],
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

NEUTRAL_COLORS: Set[str] = {
    'Black', 'White', 'Grey', 'Dark Grey', 'Light Grey', 'Beige',
    'Navy', 'Dark Blue', 'Silver', 'Gold', 'Bronze/Copper',
    'Off White', 'Cream', 'Metal', 'Metallic',
}

JEWELRY_TYPES: Set[str] = {'Earring', 'Necklace', 'Ring', 'Bracelet', 'Watch', 'Other accessories'}
JEWELRY_COLORS: Set[str] = {'Gold', 'Silver', 'Bronze/Copper', 'Metal', 'Metallic', 'White', 'Black'}

UNCOMPLEMENTABLE_TYPES: Set[str] = {
    'Socks', 'Underwear bottom', 'Bra', 'Underwear T-shirt', 'Nipple covers',
    'Shoe care', 'Stain remover spray', 'Umbrella', 'Hair clip', 'Hair ribbon',
    'Hair string', 'Hair/facial hair peel', 'Eyebrow pencil', 'Lipstick',
    'Lip gloss', 'Mascara', 'Eye shadow', 'Nail polish', 'Fine cosmetics',
    'Sewing kit', 'Washing bag', 'Wood balls', 'Zipper head',
}


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

    def _score_candidate(
        self,
        cand: Dict[str, Any],
        primary: Dict[str, Any],
        is_formal: bool,
        is_summer: bool,
        is_winter: bool,
    ) -> float:
        """Scores candidate complement against a primary anchor product."""
        score = 0.0
        c_aud = str(cand.get('audience') or '').strip()
        p_aud = str(primary.get('audience') or '').strip()
        c_col = str(cand.get('color') or '').strip()
        p_col = str(primary.get('color') or '').strip()
        c_type = str(cand.get('product_type') or '')
        c_name = str(cand.get('name') or '').casefold()

        # Audience consistency
        if c_aud == p_aud:
            score += 25.0
        else:
            score += 15.0

        # Color coordination
        if c_type in JEWELRY_TYPES and c_col in JEWELRY_COLORS:
            score += 18.0
        elif p_col and c_col:
            if c_col.casefold() == p_col.casefold():
                score += 16.0
            elif c_col in NEUTRAL_COLORS or p_col in NEUTRAL_COLORS:
                score += 8.0

        # Formality / Style consistency
        if is_formal:
            if any(w in c_name for w in ('backpack', 'back pack', 'canvas', 'gym', 'sport', 'shopper', 'tote', 'fanny', 'waist bag')):
                score -= 50.0
            elif any(w in c_name for w in ('clutch', 'evening', 'chain', 'satin', 'velvet', 'crossbody', 'small shoulder bag')):
                score += 25.0
            if c_type in ('Heeled sandals', 'Pumps', 'Heels'):
                score += 20.0
            elif c_type in ('Sneakers', 'Flip flop', 'Slippers'):
                score -= 50.0
            if c_type == 'Sunglasses':
                score -= 40.0
        elif is_summer:
            if c_type in ('Sandals', 'Flip flop', 'Ballerinas', 'Sunglasses', 'Hat/brim'):
                score += 20.0
            elif c_type in ('Boots', 'Coat', 'Scarf', 'Gloves', 'Hat/beanie'):
                score -= 40.0
        elif is_winter:
            if c_type in ('Scarf', 'Gloves', 'Hat/beanie', 'Boots'):
                score += 20.0
            elif c_type in ('Sandals', 'Sunglasses', 'Swimwear', 'Bikini top', 'Swimsuit'):
                score -= 40.0

        # Rating and reviews
        avg_r = cand['rating']['average']
        if avg_r is not None:
            score += avg_r * 2.0
        score += min(5.0, cand['rating']['count'] / 50.0)

        return score

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

        # Check uncomplementable primary product if NO explicit complement requested
        first_primary = self.product_map.get(results[0]['id'])
        if first_primary and complement_request is None:
            pt = str(first_primary.get('product_type') or '')
            if pt in UNCOMPLEMENTABLE_TYPES:
                # Per brief: empty recommendation receives full credit when no complement exists
                return []

        # Contextual formality / seasonal cues
        query_text = (arguments.get('query') or '').casefold()
        is_formal = any(w in query_text for w in ('evening', 'cocktail', 'formal', 'silk', 'gown', 'prom', 'wedding', 'suit', 'blazer', 'tuxedo', 'black tie'))
        is_summer = any(w in query_text for w in ('summer', 'beach', 'swim', 'resort', 'vacation', 'garden', 'pool', 'sun', 'warm weather', 'hot weather'))
        is_winter = any(w in query_text for w in ('winter', 'ski', 'snow', 'cold', 'thermal', 'chilly', 'frost'))

        if not is_formal and first_primary:
            fp_text = (str(first_primary.get('name') or '') + ' ' + str(first_primary.get('product_type') or '')).casefold()
            if any(w in fp_text for w in ('evening', 'cocktail', 'formal', 'silk', 'gown', 'blazer', 'suit')):
                is_formal = True

        # CASE 1: Explicit Complement Request (e.g. "with a matching bag", "with a tie under 1500")
        if complement_request and complement_request.target_types:
            # Candidates must satisfy target types and explicit budget
            cand_pool: List[Tuple[float, Dict[str, Any], str]] = []
            for target_type in complement_request.target_types:
                t_folded = target_type.casefold()
                for cand in self.by_product_type.get(t_folded, []):
                    c_id = cand['id']
                    if c_id in primary_ids or c_id in selected_rec_ids:
                        continue
                    c_price = cand['price']['amount']
                    if complement_request.max_price is not None and c_price > complement_request.max_price:
                        continue

                    c_aud = str(cand.get('audience') or '').strip()
                    if filter_aud_cluster and c_aud and c_aud not in filter_aud_cluster:
                        continue

                    # Score against primary product results[0]
                    primary_for_cand = first_primary or cand
                    p_aud_cluster = self._get_audience_cluster(primary_for_cand.get('audience'))
                    if p_aud_cluster and c_aud and c_aud not in p_aud_cluster:
                        continue

                    s = self._score_candidate(cand, primary_for_cand, is_formal, is_summer, is_winter)
                    cand_pool.append((s, cand, results[0]['id']))

            cand_pool.sort(key=lambda x: -x[0])

            for score_val, best_cand, anchor_id in cand_pool:
                if len(recommendations) >= 5:
                    break
                if best_cand['id'] in selected_rec_ids:
                    continue

                selected_rec_ids.add(best_cand['id'])
                comp_name = str(best_cand.get('name') or best_cand.get('product_type') or 'accessory')
                comp_col = str(best_cand.get('color') or '')
                comp_type = str(best_cand.get('product_type') or 'accessory')
                anchor_prod = self.product_map.get(anchor_id, {})
                anchor_name = str(anchor_prod.get('name') or 'selected product')

                if complement_request.max_price is not None:
                    reason = f"A complementary {comp_col} {comp_name} ({comp_type}) at INR {best_cand['price']['amount']:.0f} matching your budget for {anchor_name}."
                elif comp_col and anchor_prod.get('color') and comp_col.casefold() == str(anchor_prod.get('color')).casefold():
                    reason = f"A matching {comp_col} {comp_name} ({comp_type}) to complement the selected {anchor_name}."
                elif comp_col:
                    reason = f"A coordinating {comp_col} {comp_name} ({comp_type}) to complement the selected {anchor_name}."
                else:
                    reason = f"A stylish {comp_name} ({comp_type}) to complement the selected {anchor_name}."

                if len(reason) > 300:
                    reason = reason[:297] + "..."

                recommendations.append({
                    'id': best_cand['id'],
                    'for_product_id': anchor_id,
                    'reason': reason,
                })

            return recommendations

        # CASE 2: Category Affinity Outfitting (include_recommendations: True without explicit complement span)
        # Allocate across top primary results (guaranteed Grade >= 2)
        primary_anchors = [p for p in [self.product_map.get(r['id']) for r in results[:2]] if p]
        if not primary_anchors and results:
            first_p = self.product_map.get(results[0]['id'])
            if first_p:
                primary_anchors = [first_p]

        used_affinity_types: Set[str] = set()

        for anchor in primary_anchors:
            if len(recommendations) >= 5:
                break

            anchor_id = anchor['id']
            anchor_type = str(anchor.get('product_type') or '')
            anchor_name = str(anchor.get('name') or anchor_type)
            anchor_color = str(anchor.get('color') or '').strip()
            anchor_aud = str(anchor.get('audience') or '').strip()
            aud_cluster = self._get_audience_cluster(anchor_aud)

            target_types = AFFINITY_TABLE.get(
                anchor_type,
                AFFINITY_TABLE.get(str(anchor.get('category') or ''), ['Bag', 'Scarf', 'Sunglasses', 'Earring', 'Belt', 'Necklace'])
            )

            for t in target_types:
                if len(recommendations) >= 5:
                    break

                t_folded = t.casefold()
                if t_folded in used_affinity_types and len(used_affinity_types) < len(target_types):
                    continue

                candidates = self.by_product_type.get(t_folded, [])
                best_cand: Optional[Dict[str, Any]] = None
                best_score = -1e9

                for cand in candidates:
                    cand_id = cand['id']
                    if cand_id in primary_ids or cand_id in selected_rec_ids:
                        continue

                    cand_aud = str(cand.get('audience') or '').strip()
                    if filter_aud_cluster and cand_aud and cand_aud not in filter_aud_cluster:
                        continue
                    if aud_cluster and cand_aud and cand_aud not in aud_cluster:
                        continue

                    score = self._score_candidate(cand, anchor, is_formal, is_summer, is_winter)
                    if score > best_score:
                        best_score = score
                        best_cand = cand

                if best_cand:
                    selected_rec_ids.add(best_cand['id'])
                    used_affinity_types.add(t_folded)

                    comp_name = str(best_cand.get('name') or best_cand.get('product_type') or 'accessory')
                    comp_col = str(best_cand.get('color') or '')
                    comp_type = str(best_cand.get('product_type') or 'accessory')

                    if comp_col and comp_col.casefold() == anchor_color.casefold():
                        reason = f"A matching {comp_col} {comp_name} ({comp_type}) to complete the look with the selected {anchor_name}."
                    elif comp_col:
                        reason = f"A coordinating {comp_col} {comp_name} ({comp_type}) that pairs elegantly with the selected {anchor_name}."
                    else:
                        reason = f"A classic {comp_name} to complement the selected {anchor_name}."

                    if len(reason) > 300:
                        reason = reason[:297] + "..."

                    recommendations.append({
                        'id': best_cand['id'],
                        'for_product_id': anchor_id,
                        'reason': reason,
                    })

        # If slots remain and < 5, fill from unused candidates across target types of first primary
        if len(recommendations) < 5 and primary_anchors:
            first_anchor = primary_anchors[0]
            anchor_id = first_anchor['id']
            anchor_type = str(first_anchor.get('product_type') or '')
            anchor_name = str(first_anchor.get('name') or anchor_type)
            anchor_aud = str(first_anchor.get('audience') or '').strip()
            aud_cluster = self._get_audience_cluster(anchor_aud)

            target_types = AFFINITY_TABLE.get(
                anchor_type,
                ['Bag', 'Earring', 'Necklace', 'Heeled sandals', 'Belt', 'Scarf']
            )

            fill_cands: List[Tuple[float, Dict[str, Any]]] = []
            for t in target_types:
                t_folded = t.casefold()
                for cand in self.by_product_type.get(t_folded, []):
                    c_id = cand['id']
                    if c_id in primary_ids or c_id in selected_rec_ids:
                        continue
                    c_aud = str(cand.get('audience') or '').strip()
                    if filter_aud_cluster and c_aud and c_aud not in filter_aud_cluster:
                        continue
                    if aud_cluster and c_aud and c_aud not in aud_cluster:
                        continue

                    s = self._score_candidate(cand, first_anchor, is_formal, is_summer, is_winter)
                    fill_cands.append((s, cand))

            fill_cands.sort(key=lambda x: -x[0])

            for score_val, best_cand in fill_cands:
                if len(recommendations) >= 5:
                    break
                if best_cand['id'] in selected_rec_ids:
                    continue

                selected_rec_ids.add(best_cand['id'])
                comp_name = str(best_cand.get('name') or best_cand.get('product_type') or 'accessory')
                comp_col = str(best_cand.get('color') or '')
                comp_type = str(best_cand.get('product_type') or 'accessory')

                reason = f"A coordinating {comp_col} {comp_name} ({comp_type}) to complement the selected {anchor_name}."
                if len(reason) > 300:
                    reason = reason[:297] + "..."

                recommendations.append({
                    'id': best_cand['id'],
                    'for_product_id': anchor_id,
                    'reason': reason,
                })

        return recommendations
