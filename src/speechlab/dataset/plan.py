"""Dataset plan: WHICH speakers, transcripts, and train/test split to build (Session 6a).

This is a PURE definition — no audio, no SAPI, no model. It names the voices and
transcripts and encodes the held-out split so both the generator
(scripts/make_dataset.py) and the split test (tests/test_dataset_split.py) read the
SAME source of truth. Keeping it importable means the split can be tested without
synthesizing a single WAV.

The split is speaker- AND transcript-disjoint on purpose. The leakage validator
(dataset.manifest.validate_manifest) refuses any speaker or transcript that appears in
two splits, so we only generate the two CONSISTENT blocks of the speaker x transcript
grid — the cells where "split by speaker" and "split by transcript" agree:

    train = TRAIN_SPEAKERS x TRAIN_TRANSCRIPTS
    test  = TEST_SPEAKERS  x TEST_TRANSCRIPTS

A test speaker (Zira) is a voice the detector thresholds will NEVER be tuned on, and a
test transcript is text never seen in training — so Session 6b measures generalization
to an unseen speaker on unseen words, not memorization.
"""
from __future__ import annotations

# Installed Windows SAPI voices on this machine (verified: GetInstalledVoices()).
# speaker_id -> the exact voice name SelectVoice() expects.
VOICES: dict[str, str] = {
    "david": "Microsoft David Desktop",   # en-US, male
    "hazel": "Microsoft Hazel Desktop",   # en-GB, female
    "zira": "Microsoft Zira Desktop",     # en-US, female
}

# transcript_id -> text. Apostrophe-free so the PowerShell here-string in the
# generator stays simple (same constraint as scripts/make_sample.py). Each is long
# enough (>7 s synthesized) to fit a 2 s transform segment inside the safe region
# (config variants.region_min_s + seg_len_s + region_pad_end_s).
TRANSCRIPTS: dict[str, str] = {
    # tx001 == the original Session 1 transcript (unchanged, so it is the SAME text).
    "tx001": (
        "Good evening everyone. "
        "Today I want to talk about the power of clear and confident communication. "
        "A great speech is not about big words. "
        "It is about rhythm, pauses, and the energy you bring to every sentence. "
        "Thank you for listening."
    ),
    "tx002": (
        "Welcome to this session on effective presentation skills. "
        "The way you deliver a message matters as much as the message itself. "
        "Strong speakers vary their pace and let key ideas breathe. "
        "They pause with intent and never rush an important point. "
        "Practice turns these habits into instinct."
    ),
    "tx003": (
        "Let us consider how data shapes modern decisions. "
        "Every day teams gather numbers and search for patterns inside them. "
        "A clear chart can reveal a story that raw tables hide. "
        "But honest analysis also admits what the data cannot prove. "
        "Good judgement means knowing the limits of your evidence."
    ),
    "tx004": (
        "Imagine standing before a large audience for the first time. "
        "Your heart races and the room feels impossibly quiet. "
        "Take a slow breath and remember why your topic matters. "
        "Speak to one friendly face and let your natural voice carry. "
        "Confidence grows each time you choose to begin."
    ),
}

# The held-out split. Disjoint by construction (sets below share no members).
TRAIN_SPEAKERS = ("david", "hazel")
TEST_SPEAKERS = ("zira",)
TRAIN_TRANSCRIPTS = ("tx001", "tx002")
TEST_TRANSCRIPTS = ("tx003", "tx004")


def planned_pairs() -> list[tuple[str, str, str]]:
    """Every (speaker_id, transcript_id, split) baseline we will synthesize.

    Only the two consistent blocks of the grid, so no speaker or transcript ever
    straddles a split. Returns, e.g. ("david", "tx001", "train").
    """
    pairs: list[tuple[str, str, str]] = []
    for spk in TRAIN_SPEAKERS:
        for tx in TRAIN_TRANSCRIPTS:
            pairs.append((spk, tx, "train"))
    for spk in TEST_SPEAKERS:
        for tx in TEST_TRANSCRIPTS:
            pairs.append((spk, tx, "test"))
    return pairs


if __name__ == "__main__":
    # Self-check: the split is genuinely disjoint on both axes.
    assert set(TRAIN_SPEAKERS).isdisjoint(TEST_SPEAKERS), "speaker leakage in plan"
    assert set(TRAIN_TRANSCRIPTS).isdisjoint(TEST_TRANSCRIPTS), "transcript leakage in plan"
    pairs = planned_pairs()
    assert len(pairs) == 6, len(pairs)
    assert all(VOICES[s] for s, _, _ in pairs)
    assert all(TRANSCRIPTS[t] for _, t, _ in pairs)
    print(f"plan OK: {len(pairs)} pairs "
          f"({len(TRAIN_SPEAKERS)}x{len(TRAIN_TRANSCRIPTS)} train + "
          f"{len(TEST_SPEAKERS)}x{len(TEST_TRANSCRIPTS)} test)")
