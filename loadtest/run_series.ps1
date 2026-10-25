# Protocol: Locust load-testing series
# Interleaved warm-up (15 s) and measured run (70 s) across 5, 20, and 50 users for 3 repeats.
# For each run, reconciles Locust request counts against Prometheus server metrics.
# Before running: docker compose restart api, wait until docker compose ps shows healthy, keep the machine idle and plugged in (any other CPU load changes the results).

$out = "docs\results\loadtest"
$py = ".venv\Scripts\python.exe"
$base = "http://127.0.0.1:8000"

foreach ($r in 1..3) {
    foreach ($u in 5, 20, 50) {
        $tag = "u{0:D2}_r{1}" -f $u, $r

        # Warm-up
        & $py -m locust -f loadtest/locustfile.py --headless -u $u -r $u -t 15s --host $base --only-summary

        # Pre-run metrics snapshot
        & $py scripts\read_metrics.py --url "$base/metrics" --out "$out\${tag}_before.txt"

        # Measured run
        & $py -m locust -f loadtest/locustfile.py --headless -u $u -r $u -t 70s --reset-stats --host $base --json-file "$out\$tag" --only-summary

        # Post-run metrics snapshot
        & $py scripts\read_metrics.py --url "$base/metrics" --out "$out\${tag}_after.txt"

        # Summarize & reconcile
        & $py -m scripts.summarize_loadtest --json "$out\${tag}.json" --users $u --metrics-before "$out\${tag}_before.txt" --metrics-after "$out\${tag}_after.txt" --out "$out\${tag}.md"

        if ($LASTEXITCODE -ne 0) {
            Write-Warning "$tag FAILED summarize/reconcile"
        }
    }
}
