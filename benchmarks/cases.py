"""Curated test cases and diagnostic query definitions for offline evaluation."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DiagnosticCase:
    case_id: str
    category: str
    arguments: Dict[str, Any]
    intent_description: str
    applicable_constraints: Dict[str, Any]
    catalog_justification: str
    diagnostic_type: str  # 'contract_test' or 'relevance_diagnostic'
    eval_type: str        # 'empty_response', 'mixed_choice', 'recommendation', 'constraint_check', 'manual_review'
    expected_product_types: List[str] = field(default_factory=list)
    expected_attributes: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None


DIAGNOSTIC_CASES: List[DiagnosticCase] = [
    # -------------------------------------------------------------
    # 1. Simple Product-Intent Searches
    # -------------------------------------------------------------
    DiagnosticCase(
        case_id="relevance_simple_01",
        category="single_intent",
        arguments={"query": "cotton t-shirt", "limit": 10},
        intent_description="Search for cotton t-shirts",
        applicable_constraints={},
        catalog_justification="Contains thousands of 'T-shirt' product types with cotton material in descriptions.",
        diagnostic_type="relevance_diagnostic",
        eval_type="constraint_check",
        expected_product_types=["T-shirt", "Top"],
        expected_attributes={},
    ),
    DiagnosticCase(
        case_id="relevance_simple_02",
        category="single_intent",
        arguments={"query": "denim jacket", "limit": 10},
        intent_description="Search for denim jackets",
        applicable_constraints={},
        catalog_justification="Contains Jacket product types with denim material in name/description.",
        diagnostic_type="relevance_diagnostic",
        eval_type="constraint_check",
        expected_product_types=["Jacket"],
        expected_attributes={},
    ),
    DiagnosticCase(
        case_id="relevance_simple_03",
        category="single_intent",
        arguments={"query": "running shorts", "limit": 10},
        intent_description="Athletic/running shorts",
        applicable_constraints={},
        catalog_justification="Contains Shorts product types across Ladieswear and Men audiences.",
        diagnostic_type="relevance_diagnostic",
        eval_type="constraint_check",
        expected_product_types=["Shorts"],
        expected_attributes={},
    ),

    # -------------------------------------------------------------
    # 2. Product Names and Synonyms
    # -------------------------------------------------------------
    DiagnosticCase(
        case_id="synonym_01",
        category="synonyms",
        arguments={"query": "sneakers", "limit": 10},
        intent_description="Casual athletic shoes / trainers",
        applicable_constraints={},
        catalog_justification="Catalog may use 'Trainers', 'Sneakers', or 'Sport shoes' in name or product_type.",
        diagnostic_type="relevance_diagnostic",
        eval_type="manual_review",
        expected_product_types=["Trainers", "Shoes", "Sneakers"],
        notes="Baseline may miss products labeled 'Trainers' if query uses 'sneakers'.",
    ),
    DiagnosticCase(
        case_id="synonym_02",
        category="synonyms",
        arguments={"query": "shades", "limit": 10},
        intent_description="Sunglasses",
        applicable_constraints={},
        catalog_justification="Catalog categorizes these under Accessories / Sunglasses.",
        diagnostic_type="relevance_diagnostic",
        eval_type="manual_review",
        expected_product_types=["Sunglasses"],
        notes="Slang synonym for sunglasses; strict token match is expected to fail in baseline.",
    ),
    DiagnosticCase(
        case_id="synonym_03",
        category="synonyms",
        arguments={"query": "trousers", "limit": 10},
        intent_description="Pants / trousers",
        applicable_constraints={},
        catalog_justification="Catalog frequently uses 'Trousers' as a product_type.",
        diagnostic_type="relevance_diagnostic",
        eval_type="constraint_check",
        expected_product_types=["Trousers"],
    ),

    # -------------------------------------------------------------
    # 3. Natural-Language Stopwords
    # -------------------------------------------------------------
    DiagnosticCase(
        case_id="stopwords_01",
        category="stopwords",
        arguments={"query": "a red dress for women", "limit": 10},
        intent_description="Red women's dress with conversational stopwords ('a', 'for')",
        applicable_constraints={"color": "Red", "audience": "Ladieswear"},
        catalog_justification="Contains Dress product types with Ladieswear audience and Red color.",
        diagnostic_type="relevance_diagnostic",
        eval_type="constraint_check",
        expected_product_types=["Dress"],
        expected_attributes={"color": "Red", "audience": "Ladieswear"},
        notes="Known baseline failure: 'for' and 'a' required by strict AND matching.",
    ),
    DiagnosticCase(
        case_id="stopwords_02",
        category="stopwords",
        arguments={"query": "comfortable clothes for sleeping", "limit": 10},
        intent_description="Sleepwear / pajamas with stopwords",
        applicable_constraints={},
        catalog_justification="Catalog contains Nightwear, Pyjamas, and Sleepwear sections.",
        diagnostic_type="relevance_diagnostic",
        eval_type="manual_review",
        expected_product_types=["Pyjamas", "Nightwear", "Robe"],
        notes="Baseline token set will fail on 'for' and 'sleeping' unless exact match exists.",
    ),

    # -------------------------------------------------------------
    # 4. Color and Product-Type Combinations
    # -------------------------------------------------------------
    DiagnosticCase(
        case_id="attribute_combo_01",
        category="attribute_combo",
        arguments={"query": "black jeans", "limit": 10},
        intent_description="Jeans with black color attribute",
        applicable_constraints={"color": "Black"},
        catalog_justification="Numerous Trousers/Jeans with color 'Black' in catalog.",
        diagnostic_type="relevance_diagnostic",
        eval_type="constraint_check",
        expected_product_types=["Trousers", "Jeans"],
        expected_attributes={"color": "Black"},
    ),
    DiagnosticCase(
        case_id="attribute_combo_02",
        category="attribute_combo",
        arguments={"query": "blue striped shirt", "limit": 10},
        intent_description="Shirt with Blue color and Stripe pattern",
        applicable_constraints={"color": "Blue", "pattern": "Stripe"},
        catalog_justification="Catalog contains 'Shirt' product_type with color 'Blue' and pattern 'Stripe'.",
        diagnostic_type="relevance_diagnostic",
        eval_type="constraint_check",
        expected_product_types=["Shirt"],
        expected_attributes={"color": "Blue"},
    ),

    # -------------------------------------------------------------
    # 5. Explicit Budget Constraints Embedded in Queries
    # -------------------------------------------------------------
    DiagnosticCase(
        case_id="budget_query_01",
        category="budget_in_query",
        arguments={"query": "cotton dress under 2000", "limit": 10},
        intent_description="Dress with cotton material costing at most INR 2000",
        applicable_constraints={"max_price": 2000},
        catalog_justification="Catalog contains Dress items under INR 2000.",
        diagnostic_type="relevance_diagnostic",
        eval_type="constraint_check",
        expected_product_types=["Dress"],
        expected_attributes={"max_price": 2000},
        notes="Baseline failure: searches for tokens 'under' and '2000' as product words.",
    ),
    DiagnosticCase(
        case_id="budget_query_02",
        category="budget_in_query",
        arguments={"query": "shoes below INR 1500", "limit": 10},
        intent_description="Footwear priced at or below INR 1500",
        applicable_constraints={"max_price": 1500},
        catalog_justification="Catalog has shoes/sneakers priced below INR 1500.",
        diagnostic_type="relevance_diagnostic",
        eval_type="constraint_check",
        expected_product_types=["Shoes", "Trainers", "Boots", "Sandals"],
        expected_attributes={"max_price": 1500},
        notes="Baseline failure: 'below' and '1500' will filter out valid items.",
    ),

    # -------------------------------------------------------------
    # 6. Typographical Errors
    # -------------------------------------------------------------
    DiagnosticCase(
        case_id="typo_01",
        category="typos",
        arguments={"query": "coton t-shirt", "limit": 10},
        intent_description="Typo in material: 'coton' instead of 'cotton'",
        applicable_constraints={},
        catalog_justification="Shoppers frequently mistype common fabric names.",
        diagnostic_type="relevance_diagnostic",
        eval_type="manual_review",
        expected_product_types=["T-shirt"],
        notes="Baseline has zero fuzzy matching; expected to return 0 results.",
    ),
    DiagnosticCase(
        case_id="typo_02",
        category="typos",
        arguments={"query": "denm trousers", "limit": 10},
        intent_description="Typo in fabric: 'denm' instead of 'denim'",
        applicable_constraints={},
        catalog_justification="Denim trousers exist in high volume.",
        diagnostic_type="relevance_diagnostic",
        eval_type="manual_review",
        expected_product_types=["Trousers"],
        notes="Baseline returns 0 results for 'denm'.",
    ),

    # -------------------------------------------------------------
    # 7. Meaningful Unusual Requests
    # -------------------------------------------------------------
    DiagnosticCase(
        case_id="unusual_01",
        category="meaningful_unusual",
        arguments={"query": "something comfortable for a lazy Sunday", "limit": 10},
        intent_description="Loungewear, sweatpants, soft hoodies, or casual basics",
        applicable_constraints={},
        catalog_justification="Catalog sections include 'Loungewear', 'Basics', 'Casual', soft jersey materials.",
        diagnostic_type="relevance_diagnostic",
        eval_type="manual_review",
        expected_product_types=["Hoodie", "Sweatshirt", "Trousers", "Pyjamas"],
        notes="Subjective semantic case judged by Bedrock in official competition.",
    ),
    DiagnosticCase(
        case_id="unusual_02",
        category="meaningful_unusual",
        arguments={"query": "dinner date outfit", "limit": 10},
        intent_description="Evening dresses, smart shirts, blazers, or formal wear",
        applicable_constraints={},
        catalog_justification="Catalog contains Dress, Blazer, Shirt in formal/evening sections.",
        diagnostic_type="relevance_diagnostic",
        eval_type="manual_review",
        expected_product_types=["Dress", "Blazer", "Shirt"],
        notes="Vague occasion query requiring semantic association.",
    ),

    # -------------------------------------------------------------
    # 8. Unrelated or Impossible Requests (Abstention)
    # -------------------------------------------------------------
    DiagnosticCase(
        case_id="irrelevant_01",
        category="irrelevant_impossible",
        arguments={"query": "spaceship engine", "limit": 10},
        intent_description="Completely out-of-catalog mechanical hardware",
        applicable_constraints={},
        catalog_justification="Authoritative catalog is fashion/apparel only. Must abstain.",
        diagnostic_type="contract_test",
        eval_type="empty_response",
        expected_product_types=[],
        notes="Must return {'results': [], 'recommendations': []}.",
    ),
    DiagnosticCase(
        case_id="irrelevant_02",
        category="irrelevant_impossible",
        arguments={"query": "diesel car tire", "limit": 10},
        intent_description="Automotive equipment not in fashion catalog",
        applicable_constraints={},
        catalog_justification="Catalog contains zero automotive products.",
        diagnostic_type="contract_test",
        eval_type="empty_response",
        expected_product_types=[],
    ),
    DiagnosticCase(
        case_id="irrelevant_03",
        category="irrelevant_impossible",
        arguments={"query": "quantum computer processor", "limit": 10},
        intent_description="Computing hardware not in catalog",
        applicable_constraints={},
        catalog_justification="Apparel-only catalog; impossible request.",
        diagnostic_type="contract_test",
        eval_type="empty_response",
        expected_product_types=[],
    ),
    DiagnosticCase(
        case_id="irrelevant_04",
        category="irrelevant_impossible",
        arguments={"query": "medical surgical scalpel", "limit": 10},
        intent_description="Medical device not in catalog",
        applicable_constraints={},
        catalog_justification="Apparel-only catalog; impossible request.",
        diagnostic_type="contract_test",
        eval_type="empty_response",
        expected_product_types=[],
    ),

    # -------------------------------------------------------------
    # 9. Mixed Choices (Shared Results Pool)
    # -------------------------------------------------------------
    DiagnosticCase(
        case_id="mixed_01",
        category="mixed_choices",
        arguments={"query": "black jeans and white T-shirts", "limit": 10},
        intent_description="Mixed shopping query requesting both black jeans and white t-shirts",
        applicable_constraints={},
        catalog_justification="Catalog contains ample black jeans and white t-shirts.",
        diagnostic_type="relevance_diagnostic",
        eval_type="mixed_choice",
        expected_product_types=["Trousers", "T-shirt"],
        notes="Baseline attempts to find single item with both 'jeans' and 't-shirts' words.",
    ),
    DiagnosticCase(
        case_id="mixed_02",
        category="mixed_choices",
        arguments={"query": "hoodie and sweatpants", "limit": 10},
        intent_description="Mixed query for both top and bottom casual wear",
        applicable_constraints={},
        catalog_justification="Catalog has both Hoodies and Sweatpants/Trousers.",
        diagnostic_type="relevance_diagnostic",
        eval_type="mixed_choice",
        expected_product_types=["Hoodie", "Trousers"],
    ),

    # -------------------------------------------------------------
    # 10. Complementary Requests
    # -------------------------------------------------------------
    DiagnosticCase(
        case_id="recommendation_01",
        category="complementary_recommendations",
        arguments={"query": "a dress for dinner with a matching bag", "limit": 10, "include_recommendations": True},
        intent_description="Primary dress with recommended complementary handbag",
        applicable_constraints={},
        catalog_justification="Catalog contains Dress and Bag/Handbag categories.",
        diagnostic_type="relevance_diagnostic",
        eval_type="recommendation",
        expected_product_types=["Dress"],
        notes="Recommendations must return 1-5 accessory items referencing primary results.",
    ),
    DiagnosticCase(
        case_id="recommendation_02",
        category="complementary_recommendations",
        arguments={"query": "formal suit with a matching tie", "limit": 10, "include_recommendations": True},
        intent_description="Primary formal wear with complementary tie/accessory",
        applicable_constraints={},
        catalog_justification="Catalog contains Suits/Blazers and Accessories/Ties.",
        diagnostic_type="relevance_diagnostic",
        eval_type="recommendation",
        expected_product_types=["Suit", "Blazer", "Jacket"],
    ),

    # -------------------------------------------------------------
    # 11. Audience Constraints
    # -------------------------------------------------------------
    DiagnosticCase(
        case_id="audience_01",
        category="audience_constraints",
        arguments={"query": "baby cotton romper", "limit": 10},
        intent_description="Baby clothing in Baby/Children audience",
        applicable_constraints={"audience": "Baby"},
        catalog_justification="Catalog has Baby audience with Bodysuits/Rompers.",
        diagnostic_type="relevance_diagnostic",
        eval_type="constraint_check",
        expected_product_types=["Bodysuit", "Romper", "Garment"],
        expected_attributes={"audience": "Baby"},
    ),
    DiagnosticCase(
        case_id="audience_02",
        category="audience_constraints",
        arguments={"query": "men formal trousers", "limit": 10},
        intent_description="Men's formal pants",
        applicable_constraints={"audience": "Men"},
        catalog_justification="Catalog has Men audience with Trousers product_type.",
        diagnostic_type="relevance_diagnostic",
        eval_type="constraint_check",
        expected_product_types=["Trousers"],
        expected_attributes={"audience": "Men"},
    ),

    # -------------------------------------------------------------
    # 12. Ambiguous but Defensible Queries
    # -------------------------------------------------------------
    DiagnosticCase(
        case_id="ambiguous_01",
        category="ambiguous",
        arguments={"query": "casual wear", "limit": 10},
        intent_description="Broad fashion request for everyday casual apparel",
        applicable_constraints={},
        catalog_justification="Sections and descriptions include 'Casual'.",
        diagnostic_type="relevance_diagnostic",
        eval_type="manual_review",
        expected_product_types=[],
        notes="High-level category; test should evaluate non-empty catalog grounding.",
    ),
]
