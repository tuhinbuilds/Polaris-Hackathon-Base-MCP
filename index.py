"""High-performance in-memory inverted index, mixed-choice interleaver, and complement retriever."""
import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from mixed_choice import interleave_candidates, parse_mixed_choices
from preprocess import OCCASION_WORDS, PreprocessedQuery, preprocess_query
from recommend import CategoryAffinityRecommender, split_complement_query

STRING_FILTERS = ('category', 'product_type', 'color', 'audience', 'section', 'pattern')


class InvertedIndex:
    def __init__(self, products: List[Dict[str, Any]]):
        # 1. Store products sorted by id ascending
        self.products = sorted(products, key=lambda p: p['id'])
        self.num_docs = len(self.products)

        # 2. Extract fast lookup arrays
        self.ids = [p['id'] for p in self.products]
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
            # Record filter values
            for f in STRING_FILTERS:
                val = p.get(f)
                if val:
                    val_clean = str(val).strip().casefold()
                    self.filter_postings[f][val_clean].add(i)
                    if f == 'color':
                        self.colors.add(str(val).strip())

            # Tokenize fields
            name_tokens = set(re.findall(r'\w+', (p.get('name') or '').casefold()))
            type_tokens = set(re.findall(r'\w+', (p.get('product_type') or '').casefold()))
            cat_tokens = set(re.findall(r'\w+', (p.get('category') or '').casefold()))
            attr_tokens = set(re.findall(r'\w+', ' '.join(
                str(p.get(k) or '') for k in ('audience', 'section', 'color', 'pattern')
            ).casefold()))
            desc_tokens = set(re.findall(r'\w+', (p.get('description') or '').casefold()))

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
        # 1. Natural Language Preprocessing
        prep = preprocess_query(query_str, filters, self.doc_freq, self.colors)

        if prep.is_impossible:
            return []

        # 2. String Filter Candidate Intersections
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

        # 3. Numeric Bounds (Explicit + Query Inferred)
        explicit_min_p = filters.get('min_price', 0)
        explicit_max_p = filters.get('max_price', math.inf)
        eff_min_price = max(explicit_min_p, prep.inferred_min_price or 0)
        eff_max_price = min(explicit_max_p, prep.inferred_max_price if prep.inferred_max_price is not None else math.inf)

        min_rating = filters.get('min_rating')
        min_reviews = filters.get('min_review_count', 0)

        # 4. Text Retrieval & Candidate Selection
        has_query = bool(prep.tokens or prep.expanded_tokens)
        candidate_indices: List[int] = []
        scores: Dict[int, float] = {}

        if not has_query:
            # Empty query path
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
            # Query candidate scoring
            primary_tokens = prep.tokens
            expanded_tokens = prep.expanded_tokens

            # Out-of-catalog / impossible query check
            if primary_tokens:
                missing_impossible = any(
                    t not in self.text_postings and t not in OCCASION_WORDS
                    for t in primary_tokens
                )
                if missing_impossible and not expanded_tokens:
                    return []

            # Gather postings: prioritize non-occasion catalog garment tokens
            garment_tokens = [t for t in primary_tokens if t not in OCCASION_WORDS and t in self.text_postings]
            target_tokens = garment_tokens if garment_tokens else [t for t in primary_tokens if t in self.text_postings]
            if not target_tokens and expanded_tokens:
                target_tokens = [t for t in expanded_tokens if t in self.text_postings]

            if not target_tokens:
                return []

            token_postings: List[List[int]] = [self.text_postings[t] for t in target_tokens]

            # Tally token occurrences
            hit_counts = Counter()
            for postings in token_postings:
                for idx in postings:
                    hit_counts[idx] += 1

            if not hit_counts:
                return []

            # Candidate tiering: require all target tokens to match first
            num_tokens = len(target_tokens)
            candidates = [idx for idx, count in hit_counts.items() if count == num_tokens]

            # Relaxation only if ZERO candidates match all tokens
            if not candidates and num_tokens > 1:
                candidates = [idx for idx, count in hit_counts.items() if count >= num_tokens - 1]

            if not candidates and expanded_tokens:
                candidates = list(hit_counts.keys())

            # Filter candidates by explicit string filter set if present
            if candidate_set is not None:
                candidates = [i for i in candidates if i in candidate_set]

            # Score each candidate and enforce numeric bounds
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

                # Compute Field-Weighted BM25-style Relevance Score
                score = 0.0
                terms_matched = 0

                for t in query_tokens_for_scoring:
                    idf = self.get_idf(t)
                    matched_in_doc = False

                    if i in self.name_postings[t]:
                        score += 5.0 * idf
                        matched_in_doc = True
                    elif i in self.type_postings[t]:
                        score += 4.0 * idf
                        matched_in_doc = True
                    elif i in self.category_postings[t]:
                        score += 2.5 * idf
                        matched_in_doc = True
                    elif i in self.attribute_postings[t]:
                        score += 2.0 * idf
                        matched_in_doc = True
                    elif t in self.text_postings and i in self.text_postings[t]:
                        score += 1.0 * idf
                        matched_in_doc = True

                    if matched_in_doc:
                        terms_matched += 1

                # Term coverage multiplier
                coverage = terms_matched / len(query_tokens_for_scoring) if query_tokens_for_scoring else 1.0
                score *= (coverage ** 2.0)

                # Query Attribute Reinforcement (Color & Audience)
                p = self.products[i]
                if prep.inferred_color:
                    if str(p.get('color') or '').strip().casefold() == prep.inferred_color.casefold():
                        score += 8.0
                    else:
                        score *= 0.5  # Soft down-weight non-matching colors
                if prep.inferred_audience:
                    if str(p.get('audience') or '').strip().casefold() == prep.inferred_audience.casefold():
                        score += 8.0
                    else:
                        score *= 0.3  # Down-weight non-matching audiences

                if score > 0:
                    scores[i] = score
                    candidate_indices.append(i)

        # 5. Sorting and Tie-Breaking
        if sort_by == 'relevance':
            if has_query:
                if reverse:
                    candidate_indices.sort(key=lambda i: (-scores[i], self.ids[i]))
                else:
                    candidate_indices.sort(key=lambda i: (scores[i], self.ids[i]))
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
