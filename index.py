"""High-performance in-memory inverted index, mixed-choice interleaver, and complement retriever."""
import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from mixed_choice import interleave_candidates, parse_mixed_choices
from preprocess import OCCASION_WORDS, PreprocessedQuery, preprocess_query
from recommend import CategoryAffinityRecommender, split_complement_query

STRING_FILTERS = ('category', 'product_type', 'color', 'audience', 'section', 'pattern')

TARGET_PRODUCT_TYPES: Dict[str, Set[str]] = {
    'dress': {'Dress'},
    'dresses': {'Dress'},
    'gown': {'Dress'},
    'gowns': {'Dress'},
    'shirt': {'Shirt'},
    'shirts': {'Shirt'},
    'tshirt': {'T-shirt'},
    't-shirt': {'T-shirt'},
    'tshirts': {'T-shirt'},
    'top': {'Top', 'Vest top', 'T-shirt'},
    'tops': {'Top', 'Vest top', 'T-shirt'},
    'blouse': {'Blouse'},
    'blouses': {'Blouse'},
    'sweater': {'Sweater', 'Cardigan'},
    'sweaters': {'Sweater', 'Cardigan'},
    'cardigan': {'Cardigan'},
    'cardigans': {'Cardigan'},
    'hoodie': {'Hoodie'},
    'hoodies': {'Hoodie'},
    'jacket': {'Jacket', 'Coat'},
    'jackets': {'Jacket', 'Coat'},
    'coat': {'Coat', 'Jacket'},
    'coats': {'Coat', 'Jacket'},
    'blazer': {'Blazer', 'Suit'},
    'blazers': {'Blazer', 'Suit'},
    'suit': {'Suit', 'Blazer'},
    'suits': {'Suit', 'Blazer'},
    'jeans': {'Trousers'},
    'jean': {'Trousers'},
    'trousers': {'Trousers'},
    'trouser': {'Trousers'},
    'pants': {'Trousers'},
    'pant': {'Trousers'},
    'joggers': {'Trousers'},
    'jogger': {'Trousers'},
    'shorts': {'Shorts'},
    'skirt': {'Skirt'},
    'skirts': {'Skirt'},
    'shoes': {'Sneakers', 'Boots', 'Sandals', 'Other shoe', 'Pumps', 'Heeled sandals', 'Ballerinas', 'Slippers', 'Espadrilles', 'Flip flop', 'Wedge'},
    'shoe': {'Sneakers', 'Boots', 'Sandals', 'Other shoe', 'Pumps', 'Heeled sandals', 'Ballerinas', 'Slippers', 'Espadrilles', 'Flip flop', 'Wedge'},
    'sneakers': {'Sneakers', 'Trainers'},
    'sneaker': {'Sneakers', 'Trainers'},
    'trainers': {'Sneakers', 'Trainers'},
    'trainer': {'Sneakers', 'Trainers'},
    'boots': {'Boots'},
    'boot': {'Boots'},
    'sandals': {'Sandals', 'Heeled sandals'},
    'sandal': {'Sandals', 'Heeled sandals'},
    'bag': {'Bag'},
    'bags': {'Bag'},
    'handbag': {'Bag'},
    'handbags': {'Bag'},
    'backpack': {'Bag'},
    'scarf': {'Scarf'},
    'scarves': {'Scarf'},
    'hat': {'Hat/beanie', 'Hat/brim', 'Cap/peaked'},
    'hats': {'Hat/beanie', 'Hat/brim', 'Cap/peaked'},
    'cap': {'Cap/peaked'},
    'belt': {'Belt'},
    'belts': {'Belt'},
    'tie': {'Tie'},
    'ties': {'Tie'},
    'sunglasses': {'Sunglasses'},
    'shades': {'Sunglasses'},
    'swimwear': {'Swimsuit', 'Swimwear bottom', 'Bikini top'},
    'swimsuit': {'Swimsuit'},
    'bikini': {'Bikini top', 'Swimwear bottom'},
    'pyjamas': {'Pyjama set', 'Pyjama jumpsuit/playsuit', 'Pyjama bottom'},
    'pajamas': {'Pyjama set', 'Pyjama jumpsuit/playsuit', 'Pyjama bottom'},
    'sleepwear': {'Pyjama set', 'Night gown'},
    'socks': {'Socks'},
    'leggings': {'Leggings/Tights', 'Trousers'},
    'polo': {'Polo shirt', 'T-shirt'},
    'running': {'Sneakers', 'Sport tights', 'Sport top'},
    'chinos': {'Trousers'},
    'chino': {'Trousers'},
    'jumpsuit': {'Jumpsuit/Playsuit'},
    'playsuit': {'Jumpsuit/Playsuit'},
    'heels': {'Heeled sandals', 'Pumps', 'Heels'},
    'pumps': {'Pumps', 'Heeled sandals'},
    'flats': {'Flat shoe', 'Ballerinas', 'Flat shoes'},
}

AUDIENCE_CLUSTERS: Dict[str, Set[str]] = {
    'Ladieswear': {'Ladieswear', 'Divided', 'Ladies Accessories', 'Lingeries/Tights'},
    'Menswear': {'Menswear'},
    'Children Sizes 92-140': {'Children Sizes 92-140', 'Children Sizes 134-170', 'Children Accessories, Swimwear'},
    'Baby Sizes 50-98': {'Baby Sizes 50-98'},
}


class InvertedIndex:
    def __init__(self, products: List[Dict[str, Any]]):
        # 1. Store products sorted by id ascending
        self.products = sorted(products, key=lambda p: p['id'])
        self.num_docs = len(self.products)

        # 2. Extract fast lookup arrays and id mapping
        self.ids = [p['id'] for p in self.products]
        self.id_to_idx: Dict[str, int] = {p['id']: i for i, p in enumerate(self.products)}
        self.prices = [p['price']['amount'] for p in self.products]
        self.ratings = [p['rating']['average'] for p in self.products]
        self.review_counts = [p['rating']['count'] for p in self.products]

        # 3. Fast filter postings: field -> casefold_val -> set of indices
        self.filter_postings: Dict[str, Dict[str, Set[int]]] = {
            f: defaultdict(set) for f in STRING_FILTERS
        }
        self.colors: Set[str] = set()

        # 4. Text inverted index
        self.text_postings: Dict[str, List[int]] = defaultdict(list)
        self.name_postings: Dict[str, Set[int]] = defaultdict(set)
        self.type_postings: Dict[str, Set[int]] = defaultdict(set)
        self.category_postings: Dict[str, Set[int]] = defaultdict(set)
        self.attribute_postings: Dict[str, Set[int]] = defaultdict(set)
        self.doc_freq: Counter = Counter()

        # Build indexes in single pass
        for i, p in enumerate(self.products):
            # Index product ID directly
            pid = p['id']
            self.text_postings[pid].append(i)
            self.doc_freq[pid] += 1

            # Record filter values
            for f in STRING_FILTERS:
                val = p.get(f)
                if val:
                    val_clean = str(val).strip().casefold()
                    self.filter_postings[f][val_clean].add(i)
                    if f == 'color':
                        self.colors.add(str(val).strip())

            # Tokenize fields
            name_str = (p.get('name') or '').casefold()
            type_str = (p.get('product_type') or '').casefold()
            cat_str = (p.get('category') or '').casefold()
            attr_str = ' '.join(str(p.get(k) or '') for k in ('audience', 'section', 'color', 'pattern')).casefold()
            desc_str = (p.get('description') or '').casefold()

            name_tokens = set(re.findall(r'[a-z0-9]+', name_str))
            type_tokens = set(re.findall(r'[a-z0-9]+', type_str))
            cat_tokens = set(re.findall(r'[a-z0-9]+', cat_str))
            attr_tokens = set(re.findall(r'[a-z0-9]+', attr_str))
            desc_tokens = set(re.findall(r'[a-z0-9]+', desc_str))

            # Add compound forms for common hyphenated terms (e.g. t-shirt -> tshirt)
            if 't' in name_tokens and 'shirt' in name_tokens:
                name_tokens.add('tshirt')
            if 't' in type_tokens and 'shirt' in type_tokens:
                type_tokens.add('tshirt')

            all_doc_words = name_tokens | type_tokens | cat_tokens | attr_tokens | desc_tokens

            for w in all_doc_words:
                self.text_postings[w].append(i)
                self.doc_freq[w] += 1

            for w in name_tokens:
                self.name_postings[w].add(i)
            for w in type_tokens:
                self.type_postings[w].add(i)
            for w in cat_tokens:
                self.category_postings[w].add(i)
            for w in attr_tokens:
                self.attribute_postings[w].add(i)

        # Precompute IDFs for fast scoring
        self.idf_cache: Dict[str, float] = {}
        for w, df in self.doc_freq.items():
            self.idf_cache[w] = math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))

        # 5. Initialize Complement Recommender
        self.recommender = CategoryAffinityRecommender(self.products)

    def get_idf(self, word: str) -> float:
        return self.idf_cache.get(word, 0.0)

    def _retrieve_and_score(
        self,
        query_str: str,
        filters: Dict[str, Any],
        sort_by: str,
        reverse: bool,
    ) -> List[int]:
        """Retrieves and ranks candidate document indices for a single intent."""
        clean_raw_query = query_str.strip()

        # 1. Exact Product ID Fast-Path
        if clean_raw_query in self.id_to_idx:
            idx = self.id_to_idx[clean_raw_query]
            p = self.products[idx]
            # Verify explicit filters
            for k in STRING_FILTERS:
                if k in filters and str(p.get(k) or '').strip().casefold() != str(filters[k]).strip().casefold():
                    return []
            p_price = p['price']['amount']
            if p_price < filters.get('min_price', 0) or p_price > filters.get('max_price', math.inf):
                return []
            p_avg = p['rating']['average']
            if 'min_rating' in filters and (p_avg is None or p_avg < filters['min_rating']):
                return []
            if p['rating']['count'] < filters.get('min_review_count', 0):
                return []
            return [idx]

        # 2. Natural Language Preprocessing
        prep = preprocess_query(query_str, filters, self.doc_freq, self.colors)

        if prep.is_impossible:
            return []

        # 3. String Filter Candidate Intersections
        candidate_set: Optional[Set[int]] = None
        for key in STRING_FILTERS:
            if key in filters:
                val = str(filters[key]).strip().casefold()
                postings = self.filter_postings[key].get(val)
                if postings is None:
                    # Unknown filter value returns empty
                    return []
                candidate_set = postings if candidate_set is None else (candidate_set & postings)
                if not candidate_set:
                    return []

        # 4. Numeric Bounds (Explicit + Query Inferred)
        explicit_min_p = filters.get('min_price', 0)
        explicit_max_p = filters.get('max_price', math.inf)
        eff_min_price = max(explicit_min_p, prep.inferred_min_price or 0)
        eff_max_price = min(explicit_max_p, prep.inferred_max_price if prep.inferred_max_price is not None else math.inf)

        min_rating = filters.get('min_rating')
        min_reviews = filters.get('min_review_count', 0)

        # 5. Text Retrieval & Candidate Selection
        has_query = bool(prep.tokens or prep.expanded_tokens)
        candidate_indices: List[int] = []
        scores: Dict[int, float] = {}

        if not has_query:
            # Empty query path: all matching explicit filters
            source_indices = candidate_set if candidate_set is not None else range(self.num_docs)
            for i in source_indices:
                p_amount = self.prices[i]
                if p_amount < eff_min_price or p_amount > eff_max_price:
                    continue
                r_avg = self.ratings[i]
                if min_rating is not None and (r_avg is None or r_avg < min_rating):
                    continue
                if self.review_counts[i] < min_reviews:
                    continue
                candidate_indices.append(i)
        else:
            primary_tokens = [t for t in prep.tokens if t not in {'s'}]
            expanded_tokens = list(prep.expanded_tokens)

            # Determine Target Product Types from query
            target_types: Set[str] = set()
            for t in primary_tokens:
                if t in TARGET_PRODUCT_TYPES:
                    target_types.update(TARGET_PRODUCT_TYPES[t])

            # Out-of-catalog abstention: If ANY primary token is unknown and not in OCCASION_WORDS
            missing_impossible = any(
                t not in self.text_postings and t not in OCCASION_WORDS
                for t in primary_tokens
            )
            if missing_impossible and not expanded_tokens:
                return []

            target_tokens = [t for t in primary_tokens if t in self.text_postings]
            if not target_tokens and expanded_tokens:
                target_tokens = [t for t in expanded_tokens if t in self.text_postings]

            if not target_tokens and not expanded_tokens:
                return []

            # Determine audience cluster filter if audience was inferred
            allowed_audiences: Optional[Set[str]] = None
            if prep.inferred_audience and 'audience' not in filters:
                allowed_audiences = AUDIENCE_CLUSTERS.get(prep.inferred_audience)

            # Tally token occurrences
            hit_counts = Counter()
            for t in target_tokens:
                for idx in self.text_postings[t]:
                    hit_counts[idx] += 1

            # Tally expanded intent token occurrences
            expanded_hit_counts = Counter()
            if expanded_tokens:
                for t in expanded_tokens:
                    if t in self.text_postings:
                        for idx in self.text_postings[t]:
                            expanded_hit_counts[idx] += 1

            if not hit_counts and not expanded_hit_counts:
                return []

            num_tokens = len(target_tokens)

            # Tier 1: All target tokens matched
            candidates = [idx for idx, count in hit_counts.items() if count == num_tokens] if num_tokens > 0 else []

            # Tier 2: If expanded intent tokens exist (occasion/style queries), prioritize items matching both primary hits and expanded intents
            if len(candidates) < 20 and expanded_hit_counts:
                for idx, exp_count in expanded_hit_counts.most_common(500):
                    if idx in hit_counts and idx not in candidates:
                        if not target_types or self.products[idx].get('product_type') in target_types:
                            candidates.append(idx)
                            if len(candidates) >= 200:
                                break

            # Tier 3: Enrich when candidates < 20 and multiple tokens exist
            if len(candidates) < 20 and num_tokens > 1:
                if target_types:
                    for idx, count in hit_counts.items():
                        if count >= num_tokens - 1 and idx not in candidates:
                            if self.products[idx].get('product_type') in target_types:
                                candidates.append(idx)
                                if len(candidates) >= 200:
                                    break
                elif num_tokens >= 3:
                    for idx, count in hit_counts.items():
                        if count >= num_tokens - 1 and idx not in candidates:
                            candidates.append(idx)
                            if len(candidates) >= 200:
                                break

            # Tier 4: If still sparse or empty, add pure expanded intent tokens
            if len(candidates) < 10 and expanded_hit_counts:
                for idx, exp_count in expanded_hit_counts.most_common(200):
                    if idx not in candidates:
                        if not target_types or self.products[idx].get('product_type') in target_types:
                            candidates.append(idx)
                            if len(candidates) >= 100:
                                break

            # Filter candidates by explicit string filter set if present
            if candidate_set is not None:
                candidates = [i for i in candidates if i in candidate_set]

            # Filter by audience cluster if audience was inferred
            if allowed_audiences is not None:
                aud_filtered = []
                query_has_sport = any(w in clean_raw_query.casefold() for w in ('sport', 'gym', 'workout', 'running', 'fitness', 'athletic', 'training'))
                for i in candidates:
                    cand_p = self.products[i]
                    c_aud = cand_p.get('audience') or ''
                    c_sec = (cand_p.get('section') or '').casefold()
                    if c_aud in allowed_audiences:
                        aud_filtered.append(i)
                    elif c_aud == 'Sport' and query_has_sport:
                        if prep.inferred_audience == 'Menswear' and 'men' in c_sec:
                            aud_filtered.append(i)
                        elif prep.inferred_audience == 'Ladieswear' and 'ladies' in c_sec:
                            aud_filtered.append(i)
                if aud_filtered:
                    candidates = aud_filtered

            # Score each candidate
            clean_query_lower = clean_raw_query.casefold()
            query_tokens_for_scoring = tuple(set(primary_tokens) | set(expanded_tokens))

            for i in candidates:
                # Check numeric bounds
                p_amount = self.prices[i]
                if p_amount < eff_min_price or p_amount > eff_max_price:
                    continue
                r_avg = self.ratings[i]
                if min_rating is not None and (r_avg is None or r_avg < min_rating):
                    continue
                if self.review_counts[i] < min_reviews:
                    continue

                p = self.products[i]
                p_name = (p.get('name') or '').casefold()
                p_type = p.get('product_type') or ''
                p_color = (p.get('color') or '').strip()

                # Base multi-field BM25 score
                score = 0.0
                terms_matched = 0

                for t in query_tokens_for_scoring:
                    idf = self.get_idf(t)
                    matched_in_doc = False

                    if i in self.name_postings.get(t, ()):
                        score += 6.0 * idf
                        matched_in_doc = True
                    elif i in self.type_postings.get(t, ()):
                        score += 5.0 * idf
                        matched_in_doc = True
                    elif i in self.category_postings.get(t, ()):
                        score += 3.0 * idf
                        matched_in_doc = True
                    elif i in self.attribute_postings.get(t, ()):
                        score += 2.0 * idf
                        matched_in_doc = True
                    elif t in self.text_postings and i in self.text_postings[t]:
                        score += 1.0 * idf
                        matched_in_doc = True

                    if matched_in_doc:
                        terms_matched += 1

                # Term coverage multiplier
                if primary_tokens:
                    primary_in_doc = sum(1 for t in primary_tokens if (t in self.text_postings and i in self.text_postings[t]))
                    coverage = max(0.2, primary_in_doc / len(primary_tokens))
                    score *= (coverage ** 2.0)
                else:
                    coverage = terms_matched / len(query_tokens_for_scoring) if query_tokens_for_scoring else 1.0
                    score *= (coverage ** 2.0)

                # Exact name and phrase bonuses
                if clean_query_lower == p_name:
                    score += 100.0
                elif clean_query_lower in p_name:
                    score += 40.0
                elif all(t in p_name for t in primary_tokens if len(t) > 1):
                    score += 25.0

                # Product type alignment bonus
                if target_types:
                    if p_type in target_types:
                        score += 35.0
                    else:
                        score *= 0.35  # Strongly prioritize requested product type

                # Color alignment with color family tolerance
                if prep.inferred_color:
                    inf_folded = prep.inferred_color.casefold()
                    p_col_folded = p_color.casefold()
                    if p_col_folded == inf_folded:
                        score += 35.0
                    elif inf_folded in p_col_folded or p_col_folded in inf_folded:
                        score += 15.0
                    else:
                        score *= 0.20  # Down-weight non-matching color

                # Pattern alignment
                if prep.inferred_pattern:
                    cand_pat = (p.get('pattern') or '').casefold()
                    if cand_pat == prep.inferred_pattern.casefold():
                        score += 30.0
                    else:
                        score *= 0.25  # Down-weight non-matching pattern

                # Audience alignment: prioritize adult fashion unless child/baby requested
                child_tokens = {'baby', 'babies', 'infant', 'toddler', 'toddlers', 'kid', 'kids', 'child', 'children', 'boy', 'boys', 'girl', 'girls'}
                wants_child = any(t in child_tokens for t in primary_tokens)
                p_aud = p.get('audience') or ''
                is_child_aud = p_aud in {'Baby Sizes 50-98', 'Children Sizes 92-140', 'Children Sizes 134-170', 'Children Accessories, Swimwear'}

                if not wants_child and is_child_aud:
                    score *= 0.65  # Demote infant/child clothing when general adult shopping intended
                elif wants_child and is_child_aud:
                    score += 25.0  # Boost children products when explicitly requested
                elif wants_child and not is_child_aud:
                    score *= 0.35  # Demote adult products when child/baby explicitly requested

                # Inferred audience preference
                if prep.inferred_audience:
                    if p_aud == prep.inferred_audience:
                        score += 25.0
                    elif p_aud in AUDIENCE_CLUSTERS.get(prep.inferred_audience, set()):
                        score += 5.0

                # Formality and occasion alignment
                is_formal_query = any(w in primary_tokens for w in ('formal', 'interview', 'suit', 'slacks', 'tailored', 'business', 'office'))
                if is_formal_query:
                    if any(w in p_name for w in ('cargo', 'sport', 'jogger', 'drawstring', 'sweat', 'track', 'running', 'gym', 'casual')):
                        score *= 0.1
                    if any(w in p_name for w in ('suit', 'slacks', 'formal', 'creased', 'tailored', 'wool')):
                        score += 35.0

                if score > 0:
                    scores[i] = score
                    candidate_indices.append(i)

        # 6. Sorting and Tie-Breaking
        if sort_by == 'relevance':
            if has_query:
                # Sort candidates by raw score descending
                candidate_indices.sort(key=lambda idx: (-scores[idx], self.ids[idx]))

                # Apply near-duplicate diversification across candidate pool
                seen_base_names: Dict[str, int] = defaultdict(int)
                adjusted_scores: Dict[int, float] = {}
                for idx in candidate_indices:
                    p_name_raw = self.products[idx].get('name') or ''
                    base_name = re.sub(r'\s*\(?\d+\)?$', '', p_name_raw).strip().casefold()
                    count = seen_base_names[base_name]
                    seen_base_names[base_name] += 1
                    if count == 0:
                        discount = 1.0
                    elif count == 1:
                        discount = 0.85
                    elif count == 2:
                        discount = 0.68
                    else:
                        discount = 0.45
                    adjusted_scores[idx] = scores[idx] * discount

                if reverse:
                    candidate_indices.sort(key=lambda idx: (-adjusted_scores[idx], self.ids[idx]))
                else:
                    candidate_indices.sort(key=lambda idx: (adjusted_scores[idx], self.ids[idx]))
            else:
                candidate_indices.sort(key=lambda i: self.ids[i])

        elif sort_by == 'rating':
            rated = [i for i in candidate_indices if self.ratings[i] is not None]
            unrated = [i for i in candidate_indices if self.ratings[i] is None]

            if reverse:
                rated.sort(key=lambda i: (-self.ratings[i], self.ids[i]))
            else:
                rated.sort(key=lambda i: (self.ratings[i], self.ids[i]))
            unrated.sort(key=lambda i: self.ids[i])
            candidate_indices = rated + unrated

        elif sort_by == 'price':
            if reverse:
                candidate_indices.sort(key=lambda i: (-self.prices[i], self.ids[i]))
            else:
                candidate_indices.sort(key=lambda i: (self.prices[i], self.ids[i]))

        elif sort_by == 'review_count':
            if reverse:
                candidate_indices.sort(key=lambda i: (-self.review_counts[i], self.ids[i]))
            else:
                candidate_indices.sort(key=lambda i: (self.review_counts[i], self.ids[i]))

        elif sort_by == 'id':
            candidate_indices.sort(key=lambda i: self.ids[i], reverse=reverse)

        return candidate_indices

    def search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        query_str = arguments.get('query', '')
        filters = arguments.get('filters', {})
        sort_by = arguments.get('sort_by', 'relevance')
        sort_order = arguments.get('sort_order', 'desc')
        reverse = (sort_order == 'desc')
        offset = arguments.get('offset', 0)
        limit = arguments.get('limit', 10)

        # Split complement clause from query if present
        primary_query, comp_request = split_complement_query(query_str)

        # Check for mixed-choice multi-product request
        sub_intents = parse_mixed_choices(primary_query, self.colors)

        if sub_intents:
            # Multi-intent candidate retrieval & interleaving
            cand_lists: List[List[int]] = []
            for intent in sub_intents:
                sub_filters = dict(filters)
                if intent.inferred_max_price is not None:
                    sub_filters['max_price'] = min(sub_filters.get('max_price', math.inf), intent.inferred_max_price)
                if intent.inferred_min_price is not None:
                    sub_filters['min_price'] = max(sub_filters.get('min_price', 0), intent.inferred_min_price)
                cand_lists.append(self._retrieve_and_score(intent.query_text, sub_filters, sort_by, reverse))

            sliced_indices = interleave_candidates(cand_lists, limit=limit, offset=offset)
        else:
            # Single-intent candidate retrieval
            all_candidates = self._retrieve_and_score(primary_query, filters, sort_by, reverse)
            sliced_indices = all_candidates[offset:offset + limit]

        results = [{'id': self.ids[i]} for i in sliced_indices]

        # Recommendation generation
        recommendations: List[Dict[str, Any]] = []
        if arguments.get('include_recommendations', False) and results:
            recommendations = self.recommender.generate_recommendations(results, arguments, comp_request)

        return {
            'results': results,
            'recommendations': recommendations,
        }

    def recommend(
        self,
        results: List[Dict[str, Any]],
        arguments: Dict[str, Any],
        complement_request: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Delegates recommendation generation to the category affinity recommender."""
        return self.recommender.generate_recommendations(results, arguments, complement_request)
