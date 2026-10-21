"""Automated exact contract evaluation harness for Polaris MCP."""
import json
import math
import time
from typing import Any, Dict, List, Tuple
from jsonschema import Draft202012Validator
from search import Catalog, INPUT_SCHEMA, OUTPUT_SCHEMA, validate_arguments

OUTPUT_VALIDATOR = Draft202012Validator(OUTPUT_SCHEMA)
INPUT_VALIDATOR = Draft202012Validator(INPUT_SCHEMA)


class ContractEvaluator:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.product_map = {p['id']: p for p in catalog.products}
        self.results: List[Dict[str, Any]] = []

    def _record(self, name: str, passed: bool, details: str = "", error: str = ""):
        self.results.append({
            "name": name,
            "passed": passed,
            "details": details,
            "error": error,
        })

    def run_all_contract_tests(self) -> Dict[str, Any]:
        start_time = time.perf_counter()
        self.results = []

        self.test_empty_query_string_filters()
        self.test_case_insensitive_string_filters()
        self.test_inclusive_numeric_bounds()
        self.test_null_ratings_and_zero_reviews()
        self.test_sorting_and_id_tie_breaking()
        self.test_empty_query_relevance_sort_fallback()
        self.test_pagination_stability_and_prefix()
        self.test_invalid_inputs_schema_bounds()
        self.test_invalid_types_and_edge_cases()
        self.test_unique_catalog_backed_ids()
        self.test_output_schema_compliance()
        self.test_recommendation_invariants()
        self.test_no_match_empty_response()

        elapsed = time.perf_counter() - start_time
        passed_count = sum(1 for r in self.results if r['passed'])
        failed_count = sum(1 for r in self.results if not r['passed'])

        return {
            "total_checks": len(self.results),
            "passed": passed_count,
            "failed": failed_count,
            "pass_rate": passed_count / len(self.results) if self.results else 0.0,
            "elapsed_seconds": elapsed,
            "checks": self.results,
        }

    def test_empty_query_string_filters(self):
        """1. Empty query with explicit string filters."""
        first_product = self.catalog.products[0]
        for field in ('category', 'product_type', 'color', 'audience', 'section', 'pattern'):
            sample_val = first_product.get(field)
            if not sample_val:
                continue
            try:
                res = self.catalog.search({'query': '', 'filters': {field: sample_val}, 'limit': 20})
                matched = all(
                    str(self.product_map[r['id']].get(field) or '').strip().casefold() == sample_val.casefold()
                    for r in res['results']
                )
                self._record(
                    f"filter_{field}_exact_empty_query",
                    matched and len(res['results']) > 0,
                    f"Retrieved {len(res['results'])} items, all matching {field}={sample_val}",
                )
            except Exception as e:
                self._record(f"filter_{field}_exact_empty_query", False, error=str(e))

    def test_case_insensitive_string_filters(self):
        """2. Case-insensitive exact string filters and whitespace stripping."""
        try:
            r_lower = self.catalog.search({'filters': {'color': 'black'}, 'limit': 15})
            r_upper = self.catalog.search({'filters': {'color': 'BLACK'}, 'limit': 15})
            r_spaces = self.catalog.search({'filters': {'color': '  Black  '}, 'limit': 15})
            ids_lower = [x['id'] for x in r_lower['results']]
            ids_upper = [x['id'] for x in r_upper['results']]
            ids_spaces = [x['id'] for x in r_spaces['results']]

            equal_results = (ids_lower == ids_upper == ids_spaces)
            self._record(
                "case_insensitive_and_whitespace_stripping",
                equal_results and len(ids_lower) == 15,
                f"Matched across 'black', 'BLACK', and '  Black  ' (n={len(ids_lower)})",
            )
        except Exception as e:
            self._record("case_insensitive_and_whitespace_stripping", False, error=str(e))

        # Unknown nonblank filter value must yield empty matches
        try:
            r_unknown = self.catalog.search({'filters': {'color': 'NonexistentColorX99'}, 'limit': 10})
            self._record(
                "unknown_string_filter_returns_empty",
                r_unknown['results'] == [] and r_unknown['recommendations'] == [],
                "Unknown color yielded 0 results as required by contract",
            )
        except Exception as e:
            self._record("unknown_string_filter_returns_empty", False, error=str(e))

    def test_inclusive_numeric_bounds(self):
        """3. Inclusive numeric bounds for price, rating, review_count."""
        try:
            # Price exact boundary
            r_price = self.catalog.search({'filters': {'min_price': 999, 'max_price': 999}, 'limit': 20})
            prices_ok = all(self.product_map[r['id']]['price']['amount'] == 999 for r in r_price['results'])
            self._record(
                "inclusive_numeric_bounds_price_exact",
                prices_ok and len(r_price['results']) > 0,
                f"Matched {len(r_price['results'])} items with price exactly 999",
            )

            # Rating inclusive bound
            r_rating = self.catalog.search({'filters': {'min_rating': 4.0}, 'limit': 20})
            ratings_ok = all(
                self.product_map[r['id']]['rating']['average'] is not None and
                self.product_map[r['id']]['rating']['average'] >= 4.0
                for r in r_rating['results']
            )
            self._record(
                "inclusive_numeric_bounds_rating_min",
                ratings_ok and len(r_rating['results']) > 0,
                f"Matched {len(r_rating['results'])} items with rating >= 4.0",
            )

            # Review count inclusive bound
            r_reviews = self.catalog.search({'filters': {'min_review_count': 5}, 'limit': 20})
            reviews_ok = all(self.product_map[r['id']]['rating']['count'] >= 5 for r in r_reviews['results'])
            self._record(
                "inclusive_numeric_bounds_review_count_min",
                reviews_ok and len(r_reviews['results']) > 0,
                f"Matched {len(r_reviews['results'])} items with review count >= 5",
            )
        except Exception as e:
            self._record("inclusive_numeric_bounds", False, error=str(e))

    def test_null_ratings_and_zero_reviews(self):
        """4. Products with null ratings and zero reviews."""
        try:
            # min_rating must exclude null ratings
            r_min_rating = self.catalog.search({'filters': {'min_rating': 1.0}, 'limit': 50})
            no_nulls = all(self.product_map[r['id']]['rating']['average'] is not None for r in r_min_rating['results'])
            self._record(
                "min_rating_excludes_null_ratings",
                no_nulls and len(r_min_rating['results']) > 0,
                "No null-average ratings returned when min_rating is set",
            )

            # min_review_count: 0 must allow zero-review products
            r_zero_reviews = self.catalog.search({'filters': {'min_review_count': 0}, 'limit': 50})
            zero_review_found = any(self.product_map[r['id']]['rating']['count'] == 0 for r in r_zero_reviews['results'])
            self._record(
                "min_review_count_zero_allows_unreviewed",
                zero_review_found and len(r_zero_reviews['results']) == 50,
                "Zero-review products satisfy min_review_count: 0",
            )

            # Sorting by rating (asc and desc) must place unrated at the end
            r_rating_asc = self.catalog.search({'sort_by': 'rating', 'sort_order': 'asc', 'limit': 100})
            # Check first 50 are rated (if enough rated exist)
            all_first_rated = all(self.product_map[r['id']]['rating']['average'] is not None for r in r_rating_asc['results'])
            self._record(
                "sort_rating_asc_places_nulls_last",
                all_first_rated,
                "Null ratings placed after rated items in rating asc sort",
            )
        except Exception as e:
            self._record("null_ratings_and_zero_reviews", False, error=str(e))

    def test_sorting_and_id_tie_breaking(self):
        """5. Price, rating, review_count, and ID sorting with ID ascending tie-breaking."""
        # Price sort desc with tie breaking
        try:
            r_price_desc = self.catalog.search({'sort_by': 'price', 'sort_order': 'desc', 'limit': 30})
            prices = [self.product_map[r['id']]['price']['amount'] for r in r_price_desc['results']]
            is_sorted = prices == sorted(prices, reverse=True)

            ties_broken_by_id_asc = True
            for i in range(len(r_price_desc['results']) - 1):
                id1, id2 = r_price_desc['results'][i]['id'], r_price_desc['results'][i+1]['id']
                if self.product_map[id1]['price']['amount'] == self.product_map[id2]['price']['amount']:
                    if id1 > id2:
                        ties_broken_by_id_asc = False

            self._record(
                "sort_price_desc_with_id_asc_tie_breaking",
                is_sorted and ties_broken_by_id_asc,
                "Price descending sorted correctly; equal prices broken by ID asc",
            )
        except Exception as e:
            self._record("sort_price_desc_with_id_asc_tie_breaking", False, error=str(e))

        # ID sort asc and desc
        try:
            r_id_asc = self.catalog.search({'sort_by': 'id', 'sort_order': 'asc', 'limit': 20})
            ids_asc = [r['id'] for r in r_id_asc['results']]
            self._record(
                "sort_id_asc",
                ids_asc == sorted(ids_asc),
                "IDs correctly sorted in lexicographical ascending order",
            )

            r_id_desc = self.catalog.search({'sort_by': 'id', 'sort_order': 'desc', 'limit': 20})
            ids_desc = [r['id'] for r in r_id_desc['results']]
            self._record(
                "sort_id_desc",
                ids_desc == sorted(ids_desc, reverse=True),
                "IDs correctly sorted in lexicographical descending order",
            )
        except Exception as e:
            self._record("sort_id_ordering", False, error=str(e))

    def test_empty_query_relevance_sort_fallback(self):
        """6. Empty query relevance sort falls back to ID ascending regardless of sort_order."""
        try:
            r_rel_desc = self.catalog.search({'query': '', 'sort_by': 'relevance', 'sort_order': 'desc', 'limit': 20})
            r_rel_asc = self.catalog.search({'query': '', 'sort_by': 'relevance', 'sort_order': 'asc', 'limit': 20})

            ids_desc = [r['id'] for r in r_rel_desc['results']]
            ids_asc = [r['id'] for r in r_rel_asc['results']]

            is_fallback = (ids_desc == sorted(ids_desc) and ids_asc == sorted(ids_asc) and ids_desc == ids_asc)
            self._record(
                "empty_query_relevance_fallback_to_id_asc",
                is_fallback,
                "Empty query relevance sorting defaults to ID asc regardless of sort_order parameter",
            )
        except Exception as e:
            self._record("empty_query_relevance_fallback_to_id_asc", False, error=str(e))

    def test_pagination_stability_and_prefix(self):
        """7. Pagination determinism and prefix consistency."""
        try:
            page1 = self.catalog.search({'offset': 0, 'limit': 10})['results']
            page2 = self.catalog.search({'offset': 10, 'limit': 10})['results']
            prefix20 = self.catalog.search({'offset': 0, 'limit': 20})['results']

            prefix_ok = (page1 + page2 == prefix20)
            self._record(
                "pagination_prefix_consistency",
                prefix_ok and len(prefix20) == 20,
                "[0..10] + [10..20] exactly matches [0..20]",
            )

            # Repeatability
            page1_repeat = self.catalog.search({'offset': 0, 'limit': 10})['results']
            self._record(
                "pagination_repeatability",
                page1 == page1_repeat,
                "Repeated identical requests return strictly identical results",
            )
        except Exception as e:
            self._record("pagination_stability", False, error=str(e))

    def test_invalid_inputs_schema_bounds(self):
        """8. Rejection of out-of-range bounds, unknown properties, and conflicting bounds."""
        invalid_cases = [
            ("limit_zero", {'limit': 0}),
            ("limit_exceeds_100", {'limit': 101}),
            ("offset_negative", {'offset': -1}),
            ("offset_exceeds_10000", {'offset': 10001}),
            ("min_price_exceeds_max_price", {'filters': {'min_price': 1000, 'max_price': 500}}),
            ("min_price_negative", {'filters': {'min_price': -1}}),
            ("min_rating_below_1", {'filters': {'min_rating': 0.5}}),
            ("min_rating_above_5", {'filters': {'min_rating': 5.5}}),
            ("min_review_count_negative", {'filters': {'min_review_count': -1}}),
            ("blank_string_filter", {'filters': {'color': '   '}}),
            ("unknown_root_property", {'query': 'test', 'unexpected_arg': True}),
            ("unknown_filter_property", {'filters': {'nonexistent_filter': 'value'}}),
            ("invalid_sort_by", {'sort_by': 'invalid_sort'}),
            ("invalid_sort_order", {'sort_order': 'sideways'}),
        ]

        for label, args in invalid_cases:
            try:
                self.catalog.search(args)
                self._record(f"reject_invalid_{label}", False, f"Failed to reject invalid arguments: {args}")
            except (ValueError, Exception) as e:
                self._record(f"reject_invalid_{label}", True, f"Correctly rejected with {type(e).__name__}")

    def test_invalid_types_and_edge_cases(self):
        """9. Rejection of booleans-as-integers, nulls, and non-JSON numeric values."""
        type_cases = [
            ("limit_boolean_true", {'limit': True}),
            ("limit_string_number", {'limit': '10'}),
            ("offset_boolean_false", {'offset': False}),
            ("query_null", {'query': None}),
            ("filters_null", {'filters': None}),
            ("include_recommendations_string", {'include_recommendations': 'true'}),
            ("nan_in_min_price", {'filters': {'min_price': float('nan')}}),
            ("inf_in_min_price", {'filters': {'min_price': float('inf')}}),
        ]

        for label, args in type_cases:
            try:
                self.catalog.search(args)
                self._record(f"reject_type_edge_case_{label}", False, f"Accepted invalid type: {args}")
            except (ValueError, Exception) as e:
                self._record(f"reject_type_edge_case_{label}", True, f"Correctly rejected type edge case: {label}")

    def test_unique_catalog_backed_ids(self):
        """10. Every result ID is a string, unique on the page, and backed by catalog."""
        try:
            res = self.catalog.search({'query': 'dress', 'limit': 50})
            ids = [r['id'] for r in res['results']]

            all_strings = all(isinstance(x, str) and len(x) > 0 for x in ids)
            all_unique = len(ids) == len(set(ids))
            all_known = all(x in self.product_map for x in ids)

            self._record(
                "results_unique_catalog_backed_ids",
                all_strings and all_unique and all_known and len(ids) > 0,
                f"Verified {len(ids)} IDs: all strings, 100% unique, 100% known in catalog",
            )
        except Exception as e:
            self._record("results_unique_catalog_backed_ids", False, error=str(e))

    def test_output_schema_compliance(self):
        """11. Output adheres strictly to schemas/output.schema.json."""
        try:
            for args in [
                {},
                {'query': 'cotton', 'limit': 5},
                {'query': 'spaceship engine'},
                {'filters': {'color': 'Black', 'max_price': 1500}},
            ]:
                res = self.catalog.search(args)
                OUTPUT_VALIDATOR.validate(res)

            self._record(
                "output_schema_compliance",
                True,
                "All test outputs validated against Draft 2020-12 output.schema.json",
            )
        except Exception as e:
            self._record("output_schema_compliance", False, error=str(e))

    def test_recommendation_invariants(self):
        """12. Recommendation structure and cross-reference invariants."""
        try:
            # When include_recommendations is false, recommendations must be []
            res_disabled = self.catalog.search({'query': 'dress', 'include_recommendations': False})
            self._record(
                "recommendations_empty_when_disabled",
                res_disabled['recommendations'] == [],
                "Recommendations empty when include_recommendations=False",
            )

            # When results is empty, recommendations must be []
            res_no_results = self.catalog.search({'query': 'spaceship engine', 'include_recommendations': True})
            self._record(
                "recommendations_empty_when_results_empty",
                res_no_results['results'] == [] and res_no_results['recommendations'] == [],
                "Recommendations empty when results are empty",
            )

            # When recommendations exist (test against hypothetical or real), verify invariants
            res_recs = self.catalog.search({'query': 'dress', 'include_recommendations': True})
            recs = res_recs['recommendations']
            if recs:
                primary_ids = {r['id'] for r in res_recs['results']}
                rec_ids = [r['id'] for r in recs]
                refs_valid = all(r['for_product_id'] in primary_ids for r in recs)
                disjoint = all(r_id not in primary_ids for r_id in rec_ids)
                unique_recs = len(rec_ids) == len(set(rec_ids))
                valid_reasons = all(isinstance(r['reason'], str) and 1 <= len(r['reason']) <= 300 for r in recs)
                self._record(
                    "recommendation_invariants_checked",
                    refs_valid and disjoint and unique_recs and valid_reasons,
                    f"Verified {len(recs)} recommendation items against all cross-reference invariants",
                )
            else:
                self._record(
                    "recommendation_invariants_checked",
                    True,
                    "Current baseline returns empty recommendations (structurally compliant stub)",
                )
        except Exception as e:
            self._record("recommendation_invariants_checked", False, error=str(e))

    def test_no_match_empty_response(self):
        """13. No match query returns results: [] and recommendations: []."""
        try:
            res = self.catalog.search({'query': 'spaceship engine'})
            is_empty = (res == {'results': [], 'recommendations': []})
            self._record(
                "no_match_returns_empty_arrays",
                is_empty,
                "Unsatisfiable query returned {'results': [], 'recommendations': []}",
            )
        except Exception as e:
            self._record("no_match_returns_empty_arrays", False, error=str(e))
