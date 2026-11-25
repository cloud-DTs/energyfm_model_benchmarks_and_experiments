EnergyFM V2 benchmark results

The archive contains CSV predictions and metrics for E1, and JSON run
records for E2 and E3. The results/ directory is ready to copy to the
repository root. See the repository README and V2.docx for the procedure.

E1 · Weather context

- comparison.csv is the archived side-by-side comparison for experiments
  01, 02, 03, 08 and 09.
- case-errors/ contains the original matched per-case MAEs. The archived
  Chronicle averages for 03 and 08 use float32 accumulation. Recomputing
  directly from forecasts in float64 shifts those averages by up to 1.45 kWh.
- chronos-metrics.csv and chronos-run.json record the Chronos rerun,
  including the 2025 and 2026 results for 08-single.
- predictions/chronos/ contains the rerun's forecasts. predictions/chronicle/
  contains the original Chronicle forecast exports. Both contain actuals
  and forecasts with and without context. For experiment 01, Chronicle's
  origin_utc labels the first target hour end. Chronos's issue_time_utc labels
  the end of the preceding observed hour. For experiment 09, lead hours 16–39
  select the next calendar day. The comparison table gives each target unit.
- input-sources.json records the data sources, attribution and preparation.

E2 · Distillation

- objective-comparison-vea.json records the full fair comparison from V2.
- The four label-budget files cover both students on VEA and UCI.
- inference-efficiency-matched.json contains the timings and equivalence checks.
- Checkpoint paths were removed from the original objective comparison JSON.

E3 · Retrospective replay

- The Toto and Chronos 2 Small files contain trigger days, candidate metrics
  and acceptance decisions. Toto used seed 42 and Small used seeds 42–44.
  Filter the Small candidates to seed 42 for the two-model comparison.
- Each model selected its own triggers. These results describe a retrospective
  replay with a quality gate and without an independent post-gate test.

run-sources.json records the source run IDs, the repository commit for the
reproduction procedure, the model revisions and the VEA and UCI dataset IDs.
