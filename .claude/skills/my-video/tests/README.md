# my-video test harness (not used by the skill at run time)

The reference file is an already-edited before/after showcase, so there is no raw footage of it. To test the
pipeline end to end, a raw-like clip is reconstructed from the reference's "Depois" card:

1. `prep_test.py REF_CARD.mp4 OUT.mp4 ../style.json ../assets/models/face_detection_yunet_2023mar.onnx`
   inpaints burned-in captions (faces protected) and inverts the measured grade.
2. Run the skill (`make_video.py --stage plan`), then `review_test.py WORK/plan.json` (the Claude review step for this
   Portuguese clip: caption fixes, B-roll queries, 3 cards), `broll.py --refill`, `make_video.py --stage render`.
3. `matched_compare.py WORK/plan.json OUT.mp4 REF_CARD.mp4 compare.png` compares the same spoken moment in output
   and reference (time-mapped through the plan) and prints colour statistics.

`report/` holds the results of the last run (verify table, side-by-side sheets, Hindi/Malayalam text-path renders).
Known test-input artifacts: caption remnants near the face (inpainting is disabled there), the reference's own
B-roll clips carry their burned-in captions and halo, and the reference's 13 jump cuts are baked into the input.
