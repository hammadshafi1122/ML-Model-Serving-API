
"""Benchmark the latency and throughput of a local prediction API.
Start the FastAPI server before running this script.
"""

import json
import math
import statistics
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8000/predict"
PAYLOAD = json.dumps({
    "features": [5.1, 3.5, 1.4, 0.2]
}).encode("utf-8")

REQUESTS = 100
TIMEOUT = 10


def percentile(values, percent):
    """Calculate a nearest-rank percentile."""
    if not values:
        return None

    ordered = sorted(values)
    index = max(
        0,
        math.ceil((percent / 100) * len(ordered)) - 1
    )
    return ordered[index]


def main():
    latencies = []
    successful_latencies = []
    successes = 0
    failures = 0

    print(f"Benchmarking: {URL}")
    print(f"Total requests: {REQUESTS}\n")

    started = time.perf_counter()

    for number in range(1, REQUESTS + 1):
        request = urllib.request.Request(
            URL,
            data=PAYLOAD,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        begin = time.perf_counter()

        try:
            with urllib.request.urlopen(
                request, timeout=TIMEOUT
            ) as response:
                response.read()
                latency = (
                    time.perf_counter() - begin
                ) * 1000
                latencies.append(latency)

                if response.status == 200:
                    successes += 1
                    successful_latencies.append(latency)
                else:
                    failures += 1
                    print(
                        f"Request {number}: unexpected HTTP "
                        f"status {response.status}"
                    )

        except urllib.error.HTTPError as exc:
            latency = (
                time.perf_counter() - begin
            ) * 1000
            latencies.append(latency)
            failures += 1
            print(
                f"Request {number}: HTTP "
                f"{exc.code} - {exc.reason}"
            )

        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
        ) as exc:
            latency = (
                time.perf_counter() - begin
            ) * 1000
            latencies.append(latency)
            failures += 1
            print(
                f"Request {number}: "
                f"connection/timeout error - {exc}"
            )

    elapsed = time.perf_counter() - started

    print("\n--- Benchmark Results ---")
    print(f"Total requests: {REQUESTS}")
    print(f"Successful HTTP 200 responses: {successes}")
    print(f"Failed requests: {failures}")
    print(f"Total elapsed time: {elapsed:.3f} seconds")

    if elapsed > 0:
        print(
            f"Successful request rate: "
            f"{successes / elapsed:.2f} requests/sec"
        )

    if latencies:
        print(
            f"Mean latency (all responses): "
            f"{statistics.mean(latencies):.2f} ms"
        )
        print(
            f"Median latency (all responses): "
            f"{statistics.median(latencies):.2f} ms"
        )
        print(
            f"P95 latency (all responses): "
            f"{percentile(latencies, 95):.2f} ms"
        )

    if successful_latencies:
        print(
            f"Mean successful latency: "
            f"{statistics.mean(successful_latencies):.2f} ms"
        )
        print(
            f"P95 successful latency: "
            f"{percentile(successful_latencies, 95):.2f} ms"
        )
    else:
        print(
            "No successful predictions; "
            "check the API server and model."
        )

    print(
        "\nNote: This is a sequential benchmark. Results "
        "depend on your machine, server configuration, "
        "and model loading."
    )


if __name__ == "__main__":
    main()
