"""Interactive dashboard (Session 5). Thin UI over src/speechlab logic.

Run:  streamlit run src/speechlab/app/main.py

This script only *arranges* and *displays*. Every number it shows comes from
speechlab.app.analysis (which wraps the Session 1-4 measurement + detection code).
It adds no measurement of its own — the honesty contract (numbers come from code,
never prose) holds right up to the screen, and a trust badge shows the guard result.
"""
from __future__ import annotations

import pathlib
import sys

import altair as alt
import pandas as pd
import streamlit as st

# Streamlit runs this file by path, so make the package importable.
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from speechlab.config import load_config  # noqa: E402
from speechlab.dataset.manifest import read_manifest  # noqa: E402
from speechlab.app.analysis import analyze, load_audio  # noqa: E402

FLAW_COLORS = {"long_pause": "#e4572e", "rate_deviation": "#f3a712",
               "monotone": "#4c6ef5"}

st.set_page_config(page_title="Speech flaw grounding", layout="wide")


@st.cache_data(show_spinner=False)
def _config() -> dict:
    return load_config()


@st.cache_data(show_spinner=False)
def _manifest(path: str):
    return read_manifest(path)


@st.cache_data(show_spinner="Analyzing clip…")
def _run(baseline_path: str, cand_path: str | None, cand_bytes: bytes | None) -> dict:
    """Cached end-to-end analysis. Keyed by baseline path + candidate (path or bytes)."""
    cfg = _config()
    by, bsr = load_audio(baseline_path)
    cy, csr = load_audio(cand_bytes) if cand_bytes is not None else load_audio(cand_path)
    return analyze(cfg, by, bsr, cy, csr)


def timeline_chart(rms_df, evidence_df, duration_s, gt=None):
    """Energy envelope with detected flaw intervals shaded; optional ground-truth band.

    Layered Altair chart (bundled, no extra dep). The shaded rectangles are the
    detectors' predicted intervals — the thing the project claims to localize.
    """
    x = alt.X("time_s:Q", title="Time (seconds)",
              scale=alt.Scale(domain=[0, duration_s], nice=False))
    line = alt.Chart(rms_df).mark_line(color="#495057").encode(
        x=x, y=alt.Y("rms_norm:Q", title="Loudness (0–1)"))
    layers = [line]
    if not evidence_df.empty:
        bands = alt.Chart(evidence_df).mark_rect(opacity=0.28).encode(
            x="start_s:Q", x2="end_s:Q",
            color=alt.Color("flaw_type:N", title="Detected flaw",
                            scale=alt.Scale(domain=list(FLAW_COLORS),
                                            range=list(FLAW_COLORS.values()))))
        layers.insert(0, bands)
    if gt is not None:
        gt_df = alt.Chart(
            pd.DataFrame({"s": [gt[0]], "e": [gt[1]]})
        ).mark_rect(opacity=0.0, stroke="#000", strokeWidth=2, strokeDash=[6, 3]).encode(
            x="s:Q", x2="e:Q")
        layers.append(gt_df)
    return alt.layer(*layers).properties(height=240)


# ---------------------------------------------------------------- sidebar
manifest = _manifest(str(ROOT / _config()["paths"]["manifest"]))
baselines = [r for r in manifest if r.baseline_or_variant == "baseline"]
variants = [r for r in manifest if r.baseline_or_variant == "variant"]

with st.sidebar:
    st.header("Source")
    if not baselines:
        st.error("No baseline in the manifest. Run scripts/make_dataset.py first.")
        st.stop()
    mode = st.segmented_control("Candidate", ["From dataset", "Upload"],
                                default="From dataset", label_visibility="collapsed")
    cand_row, cand_path, cand_bytes, reveal = None, None, None, False
    if mode == "Upload":
        up = st.file_uploader("Candidate WAV", type=["wav"])
        if up is None:
            st.info("Upload a WAV to analyze, or switch to a dataset clip.")
            st.stop()
        cand_bytes, cand_name = up.getvalue(), up.name
        # An upload has no known speaker/pair; use the first baseline as the reference.
        base_row = baselines[0]
    else:
        ids = [r.sample_id for r in variants]
        default = next((i for i, s in enumerate(ids) if s.endswith("long_pause_s1")), 0)
        pick = st.selectbox("Candidate clip", ids, index=default)
        cand_row = next(r for r in variants if r.sample_id == pick)
        cand_path, cand_name = str(ROOT / cand_row.audio_path), cand_row.sample_id
        # Speaker normalization only means something against the SAME speaker's
        # baseline, so pick the candidate's OWN pair baseline, not just baselines[0].
        base_row = next((b for b in baselines if b.pair_id == cand_row.pair_id),
                        baselines[0])
        reveal = st.toggle("Reveal ground truth", value=False,
                           help="Overlay the known injected interval (dashed box).")

    st.caption(f"Baseline (reference): `{base_row.sample_id}` — "
               f"speaker `{base_row.speaker_id}`, {base_row.sample_rate} Hz")


# ---------------------------------------------------------------- main
st.title("Temporal speech-flaw grounding")
st.caption("Evidence-grounded coaching, not a verdict on talent. Every flaw is a "
           "measured *deviation from this speaker's own baseline* — not a universal "
           "rule of 'good speaking'. Numbers come from measurement code, never prose.")

res = _run(str(ROOT / base_row.audio_path), cand_path, cand_bytes)
ev_df = res["evidence_df"]
flagged_s = float((ev_df["end_s"] - ev_df["start_s"]).sum()) if not ev_df.empty else 0.0

with st.container(horizontal=True):
    st.metric("Candidate length", f"{res['cand_duration_s']:.1f} s", border=True)
    st.metric("Flaw intervals", f"{len(res['evidence'])}", border=True)
    st.metric("Time flagged", f"{flagged_s:.1f} s", border=True)
    st.metric("Delivery-concern score", f"{res['score']}/100", border=True)
st.caption(f"Score is a transparent rule of thumb (not a validated metric): "
           f"{res['score_formula']}. It only ranks this clip's detected deviations.")

guard_ok = not res["guard_fabricated"] and not res["guard_schema"]
if guard_ok:
    st.success("Evidence verified: 0 fabricated numbers, all evidence objects valid.",
               icon=":material/verified:")
else:
    st.error(f"Guard FAILED — fabricated: {res['guard_fabricated']}, "
             f"schema: {res['guard_schema']}", icon=":material/error:")

cols = st.columns(2)
with cols[0]:
    with st.container(border=True):
        st.markdown("**Baseline (reference)**")
        st.audio(str(ROOT / base_row.audio_path), alt="Baseline reference recording")
with cols[1]:
    with st.container(border=True):
        st.markdown("**Candidate (analyzed)**")
        st.audio(cand_bytes if cand_bytes is not None else cand_path,
                 alt="Candidate recording under analysis")


gt = None
if reveal and cand_row is not None and cand_row.gt_start_s is not None:
    gt = (cand_row.gt_start_s, cand_row.gt_end_s)

with st.container(border=True):
    st.subheader("Timeline — loudness with detected flaw intervals")
    st.altair_chart(
        timeline_chart(res["cand_rms"], ev_df, res["cand_duration_s"], gt),
        alt="Candidate loudness over time with detected flaw intervals shaded")
    if gt is not None:
        st.caption("Dashed box = known ground-truth interval (revealed for comparison; "
                   "the detector never saw it).")

with st.container(border=True):
    st.subheader("Pitch contour (voiced frames)")
    if res["cand_f0"].empty:
        st.caption("No voiced pitch detected.")
    else:
        st.altair_chart(
            alt.Chart(res["cand_f0"]).mark_line(color="#4c6ef5").encode(
                x=alt.X("time_s:Q", title="Time (seconds)",
                        scale=alt.Scale(domain=[0, res["cand_duration_s"]], nice=False)),
                y=alt.Y("f0_hz:Q", title="Pitch (Hz)")).properties(height=200),
            alt="Candidate pitch in Hz over time")

st.subheader("Evidence")
if ev_df.empty:
    st.success("No deviations flagged against this speaker's baseline.",
               icon=":material/check_circle:")
else:
    for _, r in ev_df.iterrows():
        with st.container(border=True):
            st.markdown(f":{'red' if r.flaw_type=='long_pause' else 'orange' if r.flaw_type=='rate_deviation' else 'blue'}"
                        f"[**{r.flaw_type.replace('_',' ')}**]  "
                        f"`{r.start_s:.2f}s → {r.end_s:.2f}s`")
            st.write(r.sentence)
            st.table({
                ":material/straighten: Feature": r.feature,
                ":material/person: This speaker's baseline": f"{r.reference_value:g}",
                ":material/graphic_eq: Candidate measured": f"{r.candidate_value:g}",
                ":material/rule: Rule that fired": r.rule,
            })

with st.expander("Limitations & honest caveats"):
    st.markdown(
        "- **One speaker, one transcript** so far — these are computer-injected flaws, "
        "ground truth for *that manipulation only*, not proof the detector handles "
        "naturally-occurring human delivery flaws.\n"
        "- **Thresholds are untuned placeholders** (one speaker, all `train`); tuning + "
        "held-out metrics are Session 6.\n"
        "- **Monotone** can over-fire on naturally low-variability moments; **rate** uses "
        "an energy-peak syllable proxy, not alignment. Fixes depend on the Session 2 "
        "aligner (Session 6).\n"
        "- The **delivery-concern score is illustrative**, not a validated measure of "
        "speaking quality.")



