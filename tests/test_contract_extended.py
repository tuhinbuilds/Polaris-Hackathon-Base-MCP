"""Extended automated contract tests ensuring 100% adherence to competition schemas and invariants."""
import math
import unittest
from search import Catalog, validate_arguments


def sample_product(id, price=1000, rating=4.5, count=10, color='Black', name='Casual Tee', category='Garment Upper body', product_type='T-shirt'):
    return {
        'id': id,
        'name': name,
        'description': 'Comfortable cotton fabric',
        'category': category,
        'product_type': product_type,
        'audience': 'Ladieswear',
        'section': 'Basics',
        'color': color,
        'pattern': 'Solid',
        'price': {'amount': price, 'currency': 'INR', 'synthetic': True},
        'rating': {'average': rating, 'count': count, 'synthetic': True},
        'reviews': []
    }


class ExtendedContractTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog([
            sample_product('001', price=500, rating=5.0, count=20, color='Red', product_type='T-shirt'),
            sample_product('002', price=1000, rating=None, count=0, color='Black', product_type='Dress'),
            sample_product('003', price=1500, rating=3.5, count=5, color='Blue', product_type='Jeans'),
            sample_product('004', price=2000, rating=4.0, count=12, color='White', product_type='Skirt'),
            sample_product('005', price=1000, rating=4.5, count=8, color='Black', product_type='T-shirt'),
        ])

    def ids(self, **args):
        return [r['id'] for r in self.catalog.search(args)['results']]

    def test_string_filter_case_and_whitespace(self):
        """String filters must be case-insensitive and strip outer whitespace."""
        self.assertEqual(self.ids(filters={'color': 'black'}), ['002', '005'])
        self.assertEqual(self.ids(filters={'color': 'BLACK'}), ['002', '005'])
        self.assertEqual(self.ids(filters={'color': '  Black  '}), ['002', '005'])
        self.assertEqual(self.ids(filters={'color': 'NonexistentColor'}), [])

    def test_inclusive_numeric_bounds(self):
        """min_price and max_price are inclusive bounds."""
        self.assertEqual(self.ids(filters={'min_price': 1000, 'max_price': 1000}), ['002', '005'])
        self.assertEqual(self.ids(filters={'min_price': 500, 'max_price': 1000}), ['001', '002', '005'])
        self.assertEqual(self.ids(filters={'min_rating': 4.0}), ['001', '004', '005'])
        self.assertEqual(self.ids(filters={'min_review_count': 10}), ['001', '004'])

    def test_null_rating_sort_and_filter_behavior(self):
        """Unrated products fail min_rating and sort last in both asc and desc rating sorts."""
        # Fails min_rating
        self.assertNotIn('002', self.ids(filters={'min_rating': 1.0}))
        # Sort rating desc: unrated '002' must be at end
        self.assertEqual(self.ids(sort_by='rating', sort_order='desc'), ['001', '005', '004', '003', '002'])
        # Sort rating asc: unrated '002' must be at end
        self.assertEqual(self.ids(sort_by='rating', sort_order='asc'), ['003', '004', '005', '001', '002'])

    def test_sorting_tie_breaking_by_id_asc(self):
        """Equal values must be broken by ID ascending in both sort directions."""
        # 002 and 005 both have price 1000. In desc price sort, 002 should precede 005
        res_desc = self.ids(sort_by='price', sort_order='desc')
        self.assertEqual(res_desc, ['004', '003', '002', '005', '001'])

    def test_empty_query_relevance_fallback_to_id_asc(self):
        """Empty query relevance sort falls back to ID asc regardless of sort_order."""
        self.assertEqual(self.ids(query='', sort_by='relevance', sort_order='desc'), ['001', '002', '003', '004', '005'])
        self.assertEqual(self.ids(query='', sort_by='relevance', sort_order='asc'), ['001', '002', '003', '004', '005'])

    def test_pagination_invariants(self):
        """Pagination offset and limit slice correctly and deterministically."""
        self.assertEqual(self.ids(offset=0, limit=2), ['001', '002'])
        self.assertEqual(self.ids(offset=2, limit=2), ['003', '004'])
        self.assertEqual(self.ids(offset=4, limit=2), ['005'])
        self.assertEqual(self.ids(offset=5, limit=2), [])

    def test_recommendation_disabled_invariants(self):
        """Disabled recommendations must return empty array."""
        res = self.catalog.search({'query': 'tee', 'include_recommendations': False})
        self.assertEqual(res['recommendations'], [])

    def test_query_budget_extraction(self):
        """In-query budgets like 'under 600' are respected; conflicting bounds return empty."""
        self.assertEqual(self.ids(query='T-shirt under 600'), ['001'])
        # Conflicting explicit filter (min_price 800) and query (under 600) -> empty
        self.assertEqual(self.ids(query='T-shirt under 600', filters={'min_price': 800}), [])

    def test_stopwords_and_nl_query(self):
        """Conversational stopwords do not prevent relevant product matches."""
        self.assertEqual(self.ids(query='a T-shirt for cotton'), ['001', '005'])

    def test_out_of_catalog_abstention(self):
        """Out of domain requests with terms not in catalog return empty arrays."""
        self.assertEqual(self.ids(query='spaceship engine'), [])
        self.assertEqual(self.ids(query='diesel car tire'), [])

    def test_schema_bound_rejections(self):
        """Rejects out of bound arguments per input.schema.json."""
        invalid_inputs = [
            {'limit': 0},
            {'limit': 101},
            {'limit': True},
            {'offset': -1},
            {'offset': 10001},
            {'offset': False},
            {'filters': {'min_price': 100, 'max_price': 50}},
            {'filters': {'min_price': -1}},
            {'filters': {'min_rating': 0.9}},
            {'filters': {'min_rating': 5.1}},
            {'filters': {'min_review_count': -1}},
            {'filters': {'color': '   '}},
            {'filters': {'category': ''}},
            {'extra_field': 'not_allowed'},
            {'filters': {'extra_field': 'not_allowed'}},
            {'sort_by': 'invalid'},
            {'sort_order': 'diagonal'},
            {'include_recommendations': 'yes'},
            {'query': None},
            {'filters': None},
            {'filters': {'min_price': float('nan')}},
            {'filters': {'min_price': float('inf')}},
        ]
        for arg in invalid_inputs:
            with self.subTest(arg=arg), self.assertRaises(ValueError):
                self.catalog.search(arg)


class MixedChoiceAndRecommendationContractTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog([
            sample_product('001', price=500, rating=4.5, count=10, color='White', name='White Cotton T-Shirt', product_type='T-shirt'),
            sample_product('002', price=1200, rating=4.2, count=8, color='White', name='Classic White Tee', product_type='T-shirt'),
            sample_product('003', price=1500, rating=4.0, count=15, color='Black', name='Black Slim Jeans', product_type='Jeans'),
            sample_product('004', price=1800, rating=4.7, count=22, color='Black', name='Black Denim Jeans', product_type='Jeans'),
            sample_product('005', price=2200, rating=4.8, count=30, color='Black', name='Black Evening Dress', category='Garment Full body', product_type='Dress'),
            sample_product('006', price=800, rating=4.6, count=14, color='Black', name='Black Leather Clutch Bag', category='Accessories', product_type='Bag'),
            sample_product('007', price=3500, rating=4.9, count=25, color='Navy', name='Formal Wool Blazer', category='Garment Upper body', product_type='Blazer'),
            sample_product('008', price=900, rating=4.4, count=11, color='Navy', name='Navy Silk Tie', category='Accessories', product_type='Tie'),
        ])

    def test_mixed_choice_coverage_and_interleaving(self):
        """Mixed-choice query interleaves both product types to achieve full choice coverage."""
        res = self.catalog.search({'query': 'black jeans and white t-shirts', 'limit': 4})
        ids = [r['id'] for r in res['results']]
        self.assertEqual(len(ids), 4)
        # Should alternate: jeans, t-shirt, jeans, t-shirt
        self.assertEqual(ids[0], '003')  # Jeans
        self.assertEqual(ids[1], '001')  # T-shirt
        self.assertEqual(ids[2], '004')  # Jeans/Trousers
        self.assertEqual(ids[3], '002')  # T-shirt

    def test_mixed_choice_shared_budget(self):
        """Mixed-choice query with shared budget applies constraint to both sub-intents."""
        # Budget under 1600 excludes 004 (1800)
        res = self.catalog.search({'query': 'black jeans and white t-shirts under 1600', 'limit': 10})
        ids = [r['id'] for r in res['results']]
        self.assertIn('001', ids)  # T-shirt 500
        self.assertIn('002', ids)  # T-shirt 1200
        self.assertIn('003', ids)  # Jeans 1500
        self.assertNotIn('004', ids)  # Jeans 1800 exceeds 1600

    def test_recommendations_disabled_by_default(self):
        """When include_recommendations is false or omitted, recommendations must be empty."""
        res1 = self.catalog.search({'query': 'black dress'})
        self.assertEqual(res1['recommendations'], [])
        res2 = self.catalog.search({'query': 'black dress', 'include_recommendations': False})
        self.assertEqual(res2['recommendations'], [])

    def test_recommendations_empty_when_no_primary_results(self):
        """When primary results are empty, recommendations must be empty array."""
        res = self.catalog.search({'query': 'spaceship engine', 'include_recommendations': True})
        self.assertEqual(res['results'], [])
        self.assertEqual(res['recommendations'], [])

    def test_complementary_recommendations_with_explicit_clause(self):
        """Query with complement clause returns primary results and matching complement recommendations."""
        res = self.catalog.search({
            'query': 'black dress with a matching bag',
            'limit': 5,
            'include_recommendations': True
        })
        primary_ids = [r['id'] for r in res['results']]
        self.assertIn('005', primary_ids)  # Dress
        recs = res['recommendations']
        self.assertGreater(len(recs), 0)
        rec = recs[0]
        # Invariants
        self.assertEqual(rec['for_product_id'], '005')
        self.assertEqual(rec['id'], '006')  # Bag
        self.assertNotIn(rec['id'], primary_ids)
        self.assertTrue(1 <= len(rec['reason']) <= 300)

    def test_formal_suit_with_matching_tie(self):
        """Formal suit query with tie complement links blazer to tie recommendation."""
        res = self.catalog.search({
            'query': 'formal suit with a matching tie',
            'limit': 5,
            'include_recommendations': True
        })
        primary_ids = [r['id'] for r in res['results']]
        self.assertIn('007', primary_ids)  # Blazer
        recs = res['recommendations']
        self.assertGreater(len(recs), 0)
        self.assertEqual(recs[0]['for_product_id'], '007')
        self.assertEqual(recs[0]['id'], '008')  # Tie


if __name__ == '__main__':
    unittest.main()
