| Check | Status | Value | Target |
|---|---|---|---|
| Format 1080x1920 @30fps | PASS | 1080x1920 @ 30.00 | 1080x1920 @ 30 |
| Codecs h264/aac | PASS | h264/aac | h264/aac |
| Loudness I | PASS | -14.0 | -14 ±1 LUFS (ref -14.3) |
| True peak | PASS | -3.9 | ≤ -1 dBTP (ref +0.2, clipped) |
| Loudness range | PASS | 1.2 | ≤ 6 LU (ref 3.4) |
| Mean shot length | PASS | 3.4 | 3.34 s ±30% |
| Max shot length | PASS | 7.48 | ≤ 8.6 s |
| Pacing by third (fast→slow) | PASS | [2.52, 4.53, 4.53] | [2.1, 4.7, 4.7] s ±35% |
| Visual change rate vs reference (same detector) | WARN | 37 | ≈ 26 (reference: 27 in 70s) |
| Zoom range | PASS | [0.627, 1.196] | 0.68-1.19 (transitions) / 1.00-1.13 (shots) |
| Punch-in shots | PASS | 5 | levels [1.09, 1.05, 1.12, 1.03] |
| Animated zooms only in first 23% | PASS | 5 (last at 14.1s) | ≤ 15.6s |
| Zoom-blur transitions | PASS | 3 | 3 (3 per 70 s) |
| Hook has hero text + push-in | PASS | 2 hero(es), hook 3.17s | ≥1 hero + push 1.00→1.075 |
| Base caption y-centre | PASS | 62/62 in range | [0.64, 0.68] |
| Max caption width | PASS | 0.768 | ≤ 0.86 |
| Words per base group | PASS | median 2.0, max 3 | 1-3 (median 2) |
| Hero word count | PASS | 13 | ≈ 13 (13 per 70 s, front-loaded) |
| Gold share of heroes | PASS | 0.69 | ≈ 0.58 |
| Speech rate | PASS | 3.43 | ≈ 3.34 words/s (depends on speaker) |
| Max pause | PASS | 0.321 | ≤ 0.45 s |
| B-roll share | PASS | 0.17 | ≈ 0.169 |
| Info cards | PASS | 3 | 3 (3 per 70 s) |
| Cards above bottom safe zone | PASS | 0.841 | ≤ 0.85 |
| Shimmer SFX on gold heroes/cards | PASS | 13 | ≥ gold heroes (9) |
| Whoosh on zoom transitions | PASS | 3 | 3 |
| Outro calm (1.00x, no heroes) | PASS | max dev 0.000, heroes 0 | 1.00x, 0 |
| Grade deltas (luma×, contrast×, sat×, warmth+) | PASS | [1.08, 1.1, 1.25, 11.89] | [1.13, 1.19, 1.36, 8.1] |
