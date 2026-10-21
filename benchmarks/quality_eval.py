"""Semantic quality and retrieval diagnostic evaluation harness."""
import time
from typing import Any, Dict, List
from benchmarks.cases import DIAGNOSTIC_CASES, DiagnosticCase
from search import Catalog


class QualityEvaluator:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.product_map = {p['id']: p for p in catalog.products}

    def evaluate_case(self, case: DiagnosticCase) -> Dict[str, Any]:
        start = time.perf_counter()
        try:
            res = self.catalog.search(case.arguments)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            error = None
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            res = None
            error = str(e)

        results = res.get('results', []) if res else []
        recs = res.get('recommendations', []) if res else []
        result_count = len(results)
        rec_count = len(recs)

        # Inspect returned products
        returned_products = [self.product_map[r['id']] for r in results if r['id'] in self.product_map]

        # 1. Abstention check for irrelevant/impossible queries
        abstention_passed = None
        if case.eval_type == 'empty_response':
            abstention_passed = (result_count == 0 and rec_count == 0)

        # 2. Hard / applicable constraint compliance
        constraint_violations = 0
        total_constraint_checks = 0
        if case.applicable_constraints and returned_products:
            for p in returned_products:
                if 'max_price' in case.applicable_constraints:
                    total_constraint_checks += 1
                    if p['price']['amount'] > case.applicable_constraints['max_price']:
                        constraint_violations += 1
                if 'color' in case.applicable_constraints:
                    total_constraint_checks += 1
                    expected_color = case.applicable_constraints['color'].strip().casefold()
                    if str(p.get('color') or '').strip().casefold() != expected_color:
                        constraint_violations += 1
                if 'audience' in case.applicable_constraints:
                    total_constraint_checks += 1
                    expected_aud = case.applicable_constraints['audience'].strip().casefold()
                    actual_aud = str(p.get('audience') or '').strip().casefold()
                    if expected_aud not in actual_aud and actual_aud not in expected_aud:
                        constraint_violations += 1

        constraint_compliance_rate = (
            (total_constraint_checks - constraint_violations) / total_constraint_checks
            if total_constraint_checks > 0 else (1.0 if not case.applicable_constraints else 0.0)
        )

        # 3. Mixed choice coverage
        choice_coverage = None
        if case.eval_type == 'mixed_choice':
            if not returned_products:
                choice_coverage = 0.0
            else:
                found_types = set()
                for p in returned_products:
                    pt = str(p.get('product_type') or '')
                    for target_type in case.expected_product_types:
                        if target_type.casefold() in pt.casefold() or pt.casefold() in target_type.casefold():
                            found_types.add(target_type)
                choice_coverage = len(found_types) / len(case.expected_product_types) if case.expected_product_types else 0.0

        # 4. Recommendation check
        rec_generated = rec_count > 0

        # Determine overall diagnostic status
        if error:
            status = "ERROR"
        elif case.eval_type == 'empty_response':
            status = "PASS" if abstention_passed else "FAIL_NON_EMPTY"
        elif case.eval_type == 'mixed_choice':
            status = "PASS" if (choice_coverage and choice_coverage >= 1.0) else "PARTIAL_OR_ZERO_COVERAGE"
        elif case.eval_type == 'recommendation':
            status = "RECS_GENERATED" if rec_generated else "NO_RECS_GENERATED"
        elif result_count == 0:
            status = "NO_RESULTS"
        elif constraint_violations > 0:
            status = "CONSTRAINT_VIOLATION"
        else:
            status = "RESULTS_FOUND"

        return {
            "case_id": case.case_id,
            "category": case.category,
            "query": case.arguments.get('query', ''),
            "eval_type": case.eval_type,
            "diagnostic_type": case.diagnostic_type,
            "status": status,
            "result_count": result_count,
            "recommendation_count": rec_count,
            "latency_ms": elapsed_ms,
            "abstention_passed": abstention_passed,
            "constraint_checks": total_constraint_checks,
            "constraint_violations": constraint_violations,
            "constraint_compliance_rate": constraint_compliance_rate,
            "choice_coverage": choice_coverage,
            "sample_result_ids": [r['id'] for r in results[:3]],
            "sample_product_names": [p.get('name', '') for p in returned_products[:3]],
            "notes": case.notes,
            "requires_manual_review": case.eval_type == 'manual_review',
            "error": error,
        }

    def run_all_quality_diagnostics(self) -> Dict[str, Any]:
        results = [self.evaluate_case(case) for case in DIAGNOSTIC_CASES]

        total_cases = len(results)
        non_empty_results = sum(1 for r in results if r['result_count'] > 0)
        empty_results = sum(1 for r in results if r['result_count'] == 0)

        # Irrelevant cases accuracy
        irrelevant_cases = [r for r in results if r['eval_type'] == 'empty_response']
        irrelevant_correct = sum(1 for r in irrelevant_cases if r['abstention_passed'])
        abstention_rate = irrelevant_correct / len(irrelevant_cases) if irrelevant_cases else 1.0

        # Mixed choice cases
        mixed_cases = [r for r in results if r['eval_type'] == 'mixed_choice']
        mean_choice_coverage = (
            sum(r['choice_coverage'] for r in mixed_cases if r['choice_coverage'] is not None) / len(mixed_cases)
            if mixed_cases else 0.0
        )

        # Recommendation generation rate
        rec_cases = [r for r in results if r['eval_type'] == 'recommendation']
        rec_rate = sum(1 for r in rec_cases if r['recommendation_count'] > 0) / len(rec_cases) if rec_cases else 0.0

        # Latency statistics across diagnostic queries
        latencies = [r['latency_ms'] for r in results if r['latency_ms'] is not None]
        mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

        return {
            "total_cases": total_cases,
            "non_empty_results_count": non_empty_results,
            "empty_results_count": empty_results,
            "irrelevant_abstention_accuracy": abstention_rate,
            "irrelevant_cases_count": len(irrelevant_cases),
            "mixed_choice_mean_coverage": mean_choice_coverage,
            "mixed_choice_cases_count": len(mixed_cases),
            "recommendation_generation_rate": rec_rate,
            "recommendation_cases_count": len(rec_cases),
            "manual_review_cases_count": sum(1 for r in results if r['requires_manual_review']),
            "mean_diagnostic_latency_ms": mean_latency,
            "case_results": results,
        }
