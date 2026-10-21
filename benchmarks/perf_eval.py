"""Performance benchmark runner: Concurrency-1 & Concurrency-5 workloads aligned with official brief."""
import concurrent.futures
import math
import time
from typing import Any, Dict, List, Tuple
from search import Catalog

OFFICIAL_POOL_QUERIES = [
    {"limit": 10, "offset": 0},
    {"filters": {"color": "Black"}, "limit": 10, "offset": 0},
    {"filters": {"category": "Garment Upper body"}, "sort_by": "price", "sort_order": "asc", "limit": 10, "offset": 0},
    {"filters": {"audience": "Ladieswear", "max_price": 2500}, "limit": 10, "offset": 0},
    {"filters": {"product_type": "Vest top"}, "sort_by": "rating", "sort_order": "desc", "limit": 10, "offset": 0},
    {"filters": {"min_price": 1000, "max_price": 3000}, "sort_by": "price", "sort_order": "desc", "limit": 10, "offset": 0},
    {"filters": {"section": "Womens Everyday Basics"}, "limit": 10, "offset": 0},
    {"filters": {"pattern": "Solid"}, "sort_by": "id", "sort_order": "asc", "limit": 10, "offset": 0},
    {"filters": {"color": "White", "min_rating": 4.0}, "limit": 10, "offset": 0},
    {"filters": {"min_review_count": 3}, "sort_by": "review_count", "sort_order": "desc", "limit": 10, "offset": 0},
]


def nearest_rank_percentile(data: List[float], percentile: float) -> float:
    """Calculates nearest-rank percentile."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = math.ceil((percentile / 100.0) * len(sorted_data)) - 1
    index = max(0, min(len(sorted_data) - 1, index))
    return sorted_data[index]


class PerformanceBenchmark:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog

    def _execute_query(self, query_args: Dict[str, Any]) -> Tuple[float, bool]:
        start = time.perf_counter()
        try:
            res = self.catalog.search(query_args)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            is_valid = isinstance(res, dict) and 'results' in res and len(res['results']) <= query_args.get('limit', 10)
            return elapsed_ms, is_valid
        except Exception:
            return 10000.0, False  # Penalty for timeout/error per brief

    def run_benchmark(
        self,
        warmup_count: int = 20,
        requests_c1: int = 50,
        requests_c5: int = 50,
        repetitions: int = 3,
    ) -> Dict[str, Any]:
        # 1. Warm-up (20 unscored calls per official brief)
        for i in range(warmup_count):
            q = OFFICIAL_POOL_QUERIES[i % len(OFFICIAL_POOL_QUERIES)]
            self._execute_query(q)

        all_c1_latencies: List[float] = []
        all_c5_latencies: List[float] = []
        total_errors = 0
        total_duration = 0.0

        # 2. Three repetitions of [Concurrency 1 then Concurrency 5]
        for rep in range(repetitions):
            # Concurrency 1
            t_start = time.perf_counter()
            for i in range(requests_c1):
                q = OFFICIAL_POOL_QUERIES[i % len(OFFICIAL_POOL_QUERIES)]
                lat, ok = self._execute_query(q)
                all_c1_latencies.append(lat)
                if not ok:
                    total_errors += 1
            total_duration += (time.perf_counter() - t_start)

            # Concurrency 5
            t_start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(self._execute_query, OFFICIAL_POOL_QUERIES[i % len(OFFICIAL_POOL_QUERIES)])
                    for i in range(requests_c5)
                ]
                for f in concurrent.futures.as_completed(futures):
                    lat, ok = f.result()
                    all_c5_latencies.append(lat)
                    if not ok:
                        total_errors += 1
            total_duration += (time.perf_counter() - t_start)

        # Combined Metrics across all pooled calls (3 * (50 + 50) = 300 calls)
        all_latencies = all_c1_latencies + all_c5_latencies
        total_calls = len(all_latencies)

        p50 = nearest_rank_percentile(all_latencies, 50.0)
        p95 = nearest_rank_percentile(all_latencies, 95.0)
        p99 = nearest_rank_percentile(all_latencies, 99.0)
        mean_lat = sum(all_latencies) / total_calls if total_calls else 0.0

        c1_mean = sum(all_c1_latencies) / len(all_c1_latencies) if all_c1_latencies else 0.0
        c1_p95 = nearest_rank_percentile(all_c1_latencies, 95.0)

        c5_mean = sum(all_c5_latencies) / len(all_c5_latencies) if all_c5_latencies else 0.0
        c5_p95 = nearest_rank_percentile(all_c5_latencies, 95.0)

        throughput = total_calls / total_duration if total_duration > 0 else 0.0
        correct_rate = (total_calls - total_errors) / total_calls if total_calls else 0.0

        # Official formula: 5 * correct_rate * max(0, min(1, (2000 - p95) / 1800))
        potential_perf_points = 5.0 * correct_rate * max(0.0, min(1.0, (2000.0 - p95) / 1800.0))

        return {
            "repetitions": repetitions,
            "warmup_requests": warmup_count,
            "c1_requests_per_rep": requests_c1,
            "c5_requests_per_rep": requests_c5,
            "total_measured_requests": total_calls,
            "c1_mean_ms": c1_mean,
            "c1_p95_ms": c1_p95,
            "c5_mean_ms": c5_mean,
            "c5_p95_ms": c5_p95,
            "errors_or_timeouts": total_errors,
            "correct_response_rate": correct_rate,
            "mean_latency_ms": mean_lat,
            "min_latency_ms": min(all_latencies) if all_latencies else 0.0,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "p99_latency_ms": p99,
            "max_latency_ms": max(all_latencies) if all_latencies else 0.0,
            "total_duration_seconds": total_duration,
            "throughput_rps": throughput,
            "potential_performance_points": round(potential_perf_points, 2),
            "meets_full_performance_credit_goal": (p95 <= 200.0),
        }
