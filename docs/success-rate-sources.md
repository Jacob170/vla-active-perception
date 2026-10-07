# π0.5 success rates on LIBERO: who measured what, on which model

Checked 2026-10-07 from the live sources. Every π0.5 number below comes from a **LIBERO-fine-tuned** model. No source reports a base (not fine-tuned) π0.5 on LIBERO, apart from the 0% bug report for `pi05_libero_base`.

## 1. Classic LIBERO (no perturbation)

| Source | Model / checkpoint | Who trained it | Spatial | Object | Goal | 10 | Avg |
|---|---|---|---|---|---|---|---|
| openpi `examples/libero/README.md` (Physical Intelligence) | `pi05_libero` (openpi, gs://openpi-assets/checkpoints/pi05_libero), "π0.5 @ 30k (finetuned)" | Physical Intelligence | 98.8 | 98.2 | 98.0 | 92.4 | 96.85 |
| LeRobot docs, huggingface.co/docs/lerobot/pi05 | `lerobot/pi05_libero_finetuned_v044` (= our `pi05_libero_finetuned`, byte-identical) | LeRobot (HF) | 97.0 | 99.0 | 98.0 | 96.0 | 97.5 |
| `lerobot/pi05_libero_finetuned` model card | same checkpoint | LeRobot | no success rate on the card | | | | |
| LIBERO-PRO repo README, "original" | "single official checkpoint" (not named) | not stated | see §2 | | | | |

## 2. LIBERO-PRO (perturbed)

| Source | Model | Suite | Obj | **Pos** | Sem | Task | Env |
|---|---|---|---|---|---|---|---|
| LIBERO-PRO GitHub README (Zhou et al.) | π0.5, "single official checkpoint" | Spatial | 0.97 | **0.20** | 0.97 | 0.01 | 0.46 |
| same | same | Goal | 0.97 | 0.38 | 0.97 | 0.00 | 0.46 |
| same | same | 10 | 0.92 | 0.08 | 0.93 | 0.01 | 0.46 |
| same | same | Object | 0.98 | 0.17 | 0.96 | 0.01 | 0.73 |
| same | same | overall | | | | | 0.53 |
| LIBERO-PRO arXiv 2510.03827 **v1**, Table 2 | same | Goal | 0.96 | **0.00** | 0.94 | 0.04 | — |
| Lee & Har, arXiv 2606.20754 (current HTML), Table I | π0.5, source not stated | Spatial / Object / Goal / 10 | — | **43.8 / 21.0 / 32.8 / 11.0 %** | — | — | Env: 42.2 / 75.8 / 51.6 / 70.8 % |
| same | same | Spatial / Object / Goal / 10, "Obj" shift | 97.8 / 93.6 / 84.8 / 66.6 % | — | — | — | — |

Column order of the LIBERO-PRO README row was parsed from a flattened table; the text confirms Spatial-Pos 0.20 and Goal-Pos 0.38. Treat the rest as [Likely] until checked on the page.

## 3. What this means

- **Classic LIBERO baseline for our checkpoint exists:** LeRobot measured 97.0% on Spatial with exactly our model. Physical Intelligence measured 98.8% with their own openpi checkpoint. Episode counts behind the LeRobot numbers are not stated.
- **No LIBERO-PRO number exists for our checkpoint.** LIBERO-PRO says "official checkpoint" ([Likely] openpi `pi05_libero`, since it is a single multi-suite checkpoint). Lee & Har don't say what they used or whether they fine-tuned it.
- **Spatial-Pos disagrees 2×:** 0.20 (LIBERO-PRO) vs 43.8% (Lee & Har). Same model family, unknown checkpoints, unknown Pos level (LIBERO-PRO offers x/y shifts 0.1–0.5 via `use_swap`). The numbers are not comparable until the Pos setting is known.
- **LIBERO-PRO changed between versions:** Goal-Pos was 0.00 in v1 and 0.38 in the current README.
- **The spec's 45.0% / 0.935 / 0.963:** cited from Lee & Har Table I by the Aug 27 chat, not invented. The current paper says 43.8% / 0.943 / 0.961. [Guessing] earlier paper version or a misread; the export can't tell which.
- **Our success baseline must be measured:** A.1 (classic Spatial, expect about 97%, as LeRobot measured) and A.2.2 (PRO Spatial-Pos at a recorded Pos level).
