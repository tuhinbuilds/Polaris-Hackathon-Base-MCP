"""Indexed catalog search with natural-language query understanding."""
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from jsonschema import Draft202012Validator, ValidationError

from index import InvertedIndex

ROOT = Path(__file__).resolve().parent
INPUT_SCHEMA = json.loads((ROOT / 'schemas/input.schema.json').read_text())
OUTPUT_SCHEMA = json.loads((ROOT / 'schemas/output.schema.json').read_text())
VALIDATOR = Draft202012Validator(INPUT_SCHEMA)
STRING_FILTERS = ('category', 'product_type', 'color', 'audience', 'section', 'pattern')


def validate_arguments(arguments):
    # Reject non-JSON floating point values even when called directly from Python.
    json.dumps(arguments, allow_nan=False)
    try:
        VALIDATOR.validate(arguments)
    except ValidationError as exc:
        raise ValueError(exc.message) from exc
    filters = arguments.get('filters', {})
    if any(not filters[k].strip() for k in STRING_FILTERS if k in filters):
        raise ValueError('String filters cannot be blank')
    if filters.get('min_price', 0) > filters.get('max_price', math.inf):
        raise ValueError('min_price cannot exceed max_price')


@lru_cache(maxsize=2048)
def tokens(text):
    return tuple(re.findall(r'\w+', text.casefold()))


class Catalog:
    def __init__(self, products: List[Dict[str, Any]]):
        self.products = sorted(products, key=lambda p: p['id'])
        self.index = InvertedIndex(self.products)
        self.words = {
            p['id']: set(tokens(' '.join(str(p.get(k) or '') for k in
                ('name', 'description', *STRING_FILTERS)))) for p in self.products
        }

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def relevance(self, product, query_tokens):
        """Preserved extension point: lexical token relevance."""
        words = self.words[product['id']]
        if not all(word in words for word in query_tokens):
            return 0
        name_words = set(tokens(product['name']))
        return 1 + sum(word in name_words for word in query_tokens)

    def recommend(self, results, arguments):
        """Generates catalog-grounded complementary recommendations."""
        return self.index.recommend(results, arguments)

    @staticmethod
    def matches(product, filters):
        for key in STRING_FILTERS:
            if key in filters and str(product.get(key) or '').strip().casefold() != filters[key].strip().casefold():
                return False
        price = product['price']['amount']
        average = product['rating']['average']
        return (price >= filters.get('min_price', 0)
                and price <= filters.get('max_price', math.inf)
                and ('min_rating' not in filters or
                     (average is not None and average >= filters['min_rating']))
                and product['rating']['count'] >= filters.get('min_review_count', 0))

    def search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        validate_arguments(arguments)
        return self.index.search(arguments)
