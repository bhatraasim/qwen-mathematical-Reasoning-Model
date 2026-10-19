# Evaluation summary

## Accuracy by run

| Run | N | Strict acc | Strict 95% CI | Flexible acc | Flexible 95% CI | Well-formed | Hit token limit | no_match | Mean tokens |
|---|---|---|---|---|---|---|---|---|---|
| base_legacy | 1319 | 39.95% | 37.3-42.6% | 43.06% | 40.4-45.8% | 0.0% | 3.3% | 159 | 332 |
| sft_distilled_tagged_tagged | 1319 | 42.15% | 39.5-44.8% | 42.23% | 39.6-44.9% | 96.7% | 1.4% | 26 | 314 |
| checkpoint-100_tagged | 1319 | 42.84% | 40.2-45.5% | 43.06% | 40.4-45.8% | 97.6% | 1.4% | 26 | 318 |
| checkpoint-200_tagged | 1319 | 44.96% | 42.3-47.7% | 45.11% | 42.4-47.8% | 97.6% | 1.8% | 29 | 311 |

## Paired comparison vs `sft_distilled_tagged_tagged`

| Run | Change (strict) | p (strict) | Change (flexible) | p (flexible) | Only reference right | Only this run right |
|---|---|---|---|---|---|---|
| base_legacy | -2.20 pts | 0.1214 | +0.83 pts | 0.5768 | 155 | 166 |
| checkpoint-100_tagged | +0.68 pts | 0.5522 | +0.83 pts | 0.4599 | 86 | 97 |
| checkpoint-200_tagged | +2.81 pts | 0.0055 | +2.88 pts | 0.0044 | 66 | 104 |

Notes: *strict* uses the extractor stored in each JSON; *flexible* re-scores the saved responses with a lenient extractor (units, dollar signs, last-number fallback). The last two columns use flexible scoring. p-values are exact McNemar tests on the questions where the two runs disagree; they are not corrected for multiple comparisons, so treat p just under 0.05 with caution.
