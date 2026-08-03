"""Config for the head/body/tail long-tail analysis
(src/eval/longtail_strata.py). Same knob-file convention as the other
paper-figure/table configs.

STRATA are (name, minimum training-admission count), ordered high to low; a
code falls in the first stratum whose threshold it clears. The 100 / 10 cut
is the usual many- / medium- / few-shot convention from the long-tail
recognition literature, not tuned on our results -- worth keeping that way,
since a threshold chosen to flatter the numbers would be the obvious thing
for a reviewer to poke at.
"""

STRATA = [
    ("head", 100),
    ("body", 10),
    ("tail", 0),
]

# Instruction dataset whose train split defines "seen during training".
# The headline recipe; every fine-tuned model in CHECKPOINTS was trained on
# it, so one frequency table applies to all of them.
TRAIN_INSTRUCTIONS = "merlin_thrfull_icd2x.pq"

# (display size, base collection, full-FT collection) -- dev-selected best
# epoch, canonical seed 42, held-out test split.
CHECKPOINTS = [
    ("0.6B", "test-06b-base", "test-06b-full-e4"),
    ("8B", "test-8b-base", "test-8b-full-e3"),
    ("14B", "test-14b-base", "test-14b-full-e3"),
    ("32B", "test-32b-base", "test-32b-full-e3"),
]
