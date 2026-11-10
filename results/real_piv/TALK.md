# Speaker notes — real PIV result

Drop-in figure: `figures/summary_panel.png`.

## The one-liner
> We ran our **first real experimental PIV data** end-to-end through the FSI-SLM
> pipeline. It correctly reads the flow as **separated (deep stall)** — and, more
> importantly, our physics validators **caught the language model over-reaching**
> on the one thing the data couldn't support.

## 60-second story
1. **Data.** 200 DaVis PIV frames of a NACA0009 at AoA 17.2°, in water. Grid
   152×127, 15 Hz, 13.3 s.
2. **One subtlety we handled.** The exports are *mean-subtracted fluctuation
   fields* ("Vector Arithmetic") — the time-mean is zero everywhere. That's
   perfect for POD, but it means freestream speed and chord are rig inputs, not
   measurements. We say so; we don't fake them.
3. **Result.** f_shed 1.65 Hz, St 0.105, wake 0.65 c, flat POD (mode-1 8.5%) →
   **separated flow**. All four figures tell the same story.
4. **The headline.** The LoRA model claimed "VIV lock-in, confidence 1.00." That's
   wrong — it was fooled by the missing structural channel. **Both physics
   validators rejected it**, and the rule-based baseline independently said
   "separated." That's the trust-layer thesis, proven on real data.
5. **Same code path.** The real data flowed through the *exact* feature-extraction
   code the synthetic training data uses — no special-casing.

## Numbers to have ready
| | value |
|---|---|
| AoA | 17.2° (NACA0009 static stall ≈ 9–10°) |
| f_shed / St | 1.65 Hz / 0.105 |
| wake width / length | 0.65 c / 1.4 c |
| POD mode-1 | 8.5% (flat; a clean street would be a ~30–40% pair) |
| Re / U / c | 1.57×10⁵ / 1.569 m/s / 0.10 m |
| regime (baseline) | separated, conf 0.88 |

## Likely questions
- **"Why is St low / no clean tone?"** Deep stall is broadband, not a tidy von
  Kármán street — that's exactly why POD is flat. The 1.65 Hz peak is weak
  (peak/median 3.0) and there's a comparable peak near 2.3 Hz.
- **"Where's U from?"** Rig pump constant (same one the synthetic generator uses);
  a mean-subtracted field can't give it. St and Re scale with it —
  `--freestream`/`--chord` show the sensitivity.
- **"Why did the model say VIV?"** The structural displacement / force channels
  aren't in a PIV-only export, so natural frequency read as 0 and the model's
  lock-in logic misfired. The validators exist precisely for this.
- **"What's next?"** Ingest the synchronized load-cell (Cl/Cd/Cm) and video-strain
  (displacement, f_n) channels into the same RawExperiment — the pipeline already
  supports them, no code change — then lock-in can actually be assessed.

## Reproduce live if asked
```bash
python cli.py ingest-piv        # record + figures + validation + baseline
```
