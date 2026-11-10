import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="NIDS Drift Monitor", layout="wide")

DATA = Path(__file__).parent / "data"
REQUIRED = ["preprocess.joblib", "meta.json", "stream.parquet", "window_metrics.csv",
            "strategy_summary.csv", "drift_events.json", "model_initial.json",
            "model_adapted.json"]
missing = [f for f in REQUIRED if not (DATA / f).exists()]
if missing:
    st.error("These files are missing from the data folder: " + ", ".join(missing))
    st.stop()

GROUP_COLORS = {"Normal": "#8fbf9f", "Old attacks": "#f2a65a", "New attacks": "#c1121f"}
STRAT_COLORS = {"Static (no adaptation)": "#8d99ae",
                "Drift-triggered retrain": "#2a9d8f",
                "Online ARF": "#e76f51"}
PRED_COL = {"Static (no adaptation)": "pred_static",
            "Drift-triggered retrain": "pred_triggered",
            "Online ARF": "pred_arf"}


@st.cache_data
def load_data():
    stream = pd.read_parquet(DATA / "stream.parquet")
    metrics = pd.read_csv(DATA / "window_metrics.csv")
    summary = pd.read_csv(DATA / "strategy_summary.csv", index_col=0)
    summary.index.name = "Strategy"
    with open(DATA / "drift_events.json") as f:
        events = json.load(f)
    with open(DATA / "meta.json") as f:
        meta = json.load(f)
    return stream, metrics, summary, events, meta


def load_booster(path):
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


@st.cache_resource
def load_models():
    pre = joblib.load(DATA / "preprocess.joblib")
    initial = load_booster(DATA / "model_initial.json")
    adapted = load_booster(DATA / "model_adapted.json")
    return pre, initial, adapted


stream, metrics, summary, events, meta = load_data()
cfg = events["config"]
N_WIN, WIN = cfg["N_WIN"], cfg["WIN"]
D1, D2S, D2E = cfg["DRIFT1_WIN"], cfg["DRIFT2_START"], cfg["DRIFT2_END"]
KS_FRAC = cfg["KS_FRAC"]
retrains = events["retrains"]
ks_frac = events["ks_frac_by_window"]
adwin_windows = sorted({r // WIN for r in events["adwin_alarm_rows"]})

PHASES = {
    f"Pre-drift ({0}-{D1 - 1})": (0, D1 - 1),
    f"New attack families ({D1}-{D2S - 1})": (D1, D2S - 1),
    f"Gradual volume drift ({D2S}-{D2E - 1})": (D2S, D2E - 1),
    f"Volume at maximum ({D2E}-{N_WIN - 1})": (D2E, N_WIN - 1),
}


def phase_text(w):
    if w < D1:
        return "Stable period: traffic looks like the data the model was trained on."
    if w < D2S:
        return "Abrupt drift: new attack families have appeared that the initial model never saw."
    if w < D2E:
        return "Gradual drift: traffic volume is growing window by window."
    return "Volume drift complete: traffic volume now sits at its new, higher level."


# ---------------------------------------------------------------- page 1
def page_overview():
    st.title("Network Intrusion Detection with Concept Drift")
    st.write(
        "This dashboard shows what happens to an intrusion detection model on UNSW-NB15 when "
        "network traffic changes over time, and compares three ways of reacting: leaving the "
        "model alone, retraining it when drift is detected, and using an online learner "
        "that updates continuously."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stream windows", N_WIN)
    c2.metric("Rows per window", f"{WIN:,}")
    c3.metric("Abrupt drift at window", D1)
    c4.metric("Gradual drift", f"{D2S} to {D2E}")

    st.subheader("Strategy comparison")
    phase_cols = list(summary.columns[:4])
    st.dataframe(summary.round(3))
    long = summary[phase_cols].reset_index().melt(
        id_vars="Strategy", var_name="Phase", value_name="Mean F1")
    fig = px.bar(long, x="Phase", y="Mean F1", color="Strategy", barmode="group",
                 color_discrete_map=STRAT_COLORS)
    fig.update_layout(yaxis=dict(range=[0, 1.02]), legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig)

    st.subheader("Baseline models on the official test set")
    st.dataframe(pd.DataFrame(meta["baseline_metrics"]).T.round(4))


# ---------------------------------------------------------------- page 2
def page_data():
    st.title("Data explorer")
    tab1, tab2, tab3 = st.tabs(["Stream composition", "Attack families", "Feature distributions"])

    with tab1:
        comp = stream.groupby(["window", "group"]).size().reset_index(name="rows")
        fig = px.bar(comp, x="window", y="rows", color="group",
                     color_discrete_map=GROUP_COLORS,
                     category_orders={"group": ["Normal", "Old attacks", "New attacks"]})
        fig.add_vline(x=D1 - 0.5, line_dash="dash")
        fig.add_vrect(x0=D2S - 0.5, x1=D2E - 0.5, fillcolor="grey", opacity=0.15, line_width=0)
        st.plotly_chart(fig)
        st.caption("Dashed line: new attack families start. Grey band: traffic volume grows.")

    with tab2:
        att = stream[stream["label"] == 1]
        fam = att.groupby(["attack_cat", "group"]).size().reset_index(name="rows")
        fam = fam.sort_values("rows")
        fig = px.bar(fam, x="rows", y="attack_cat", color="group", orientation="h",
                     color_discrete_map=GROUP_COLORS)
        st.plotly_chart(fig)

        post = att[att["window"] >= D1]
        hits = pd.DataFrame({s: (post[c] == 1) for s, c in PRED_COL.items()})
        hits["attack_cat"] = post["attack_cat"]
        st.caption(f"Share of attacks detected per family after window {D1} (recall)")
        st.dataframe(hits.groupby("attack_cat").mean().round(3))

    with tab3:
        feats = meta["skewed_cols"] + meta["regular_cols"]
        default = feats.index("sbytes") if "sbytes" in feats else 0
        f = st.selectbox("Feature", feats, index=default)
        chosen = st.multiselect("Phases to compare", list(PHASES),
                                default=[list(PHASES)[0], list(PHASES)[-1]])
        use_log = st.checkbox("Log scale (log1p)", value=f in meta["skewed_cols"])
        frames = []
        for name in chosen:
            a, b = PHASES[name]
            part = stream.loc[stream["window"].between(a, b), [f]].copy()
            part["phase"] = name
            frames.append(part)
        if frames:
            d = pd.concat(frames)
            if use_log:
                d[f] = np.log1p(d[f].clip(lower=0))
            fig = px.histogram(d, x=f, color="phase", barmode="overlay",
                               histnorm="probability density", nbins=60, opacity=0.55)
            st.plotly_chart(fig)
        else:
            st.info("Pick at least one phase.")


# ---------------------------------------------------------------- page 3
def render_window(w, tag):
    cur = metrics[metrics["window"] == w].set_index("strategy")
    prev = metrics[metrics["window"] == w - 1].set_index("strategy") if w > 0 else None

    st.caption(f"Window {w} of {N_WIN - 1}")
    st.info(phase_text(w))

    cols = st.columns(len(STRAT_COLORS))
    for col, s in zip(cols, STRAT_COLORS):
        f1 = cur.loc[s, "f1"]
        delta = None if prev is None else f"{f1 - prev.loc[s, 'f1']:+.3f}"
        col.metric(f"F1: {s}", f"{f1:.3f}", delta)

    notes = []
    if w in adwin_windows:
        notes.append("ADWIN (watching the static model's errors) raised a drift alarm.")
    if ks_frac[w] >= KS_FRAC:
        notes.append(f"KS test: {ks_frac[w]:.0%} of features differ from the reference "
                     f"(alarm threshold {KS_FRAC:.0%}).")
    for r in retrains:
        if r["window"] == w:
            notes.append(f"Model retrained ({r['reason']}) on {r['train_rows']:,} labelled rows.")
    if notes:
        for n in notes:
            st.warning(n)
    else:
        st.success("No drift alarms in this window.")

    fig = go.Figure()
    for s, color in STRAT_COLORS.items():
        d = metrics[(metrics["strategy"] == s) & (metrics["window"] <= w)]
        fig.add_trace(go.Scatter(x=d["window"], y=d["f1"], mode="lines", name=s,
                                 line=dict(color=color, width=3)))
    seen_retrains = [r["window"] for r in retrains if r["window"] <= w]
    if seen_retrains:
        fig.add_trace(go.Scatter(x=seen_retrains, y=[0.03] * len(seen_retrains),
                                 mode="markers", name="Retrain",
                                 marker=dict(symbol="triangle-up", size=12, color="#2a9d8f")))
    seen_alarms = [a for a in adwin_windows if a <= w]
    if seen_alarms:
        fig.add_trace(go.Scatter(x=seen_alarms, y=[0.97] * len(seen_alarms),
                                 mode="markers", name="ADWIN alarm",
                                 marker=dict(symbol="x", size=11, color="crimson")))
    fig.add_vline(x=D1, line_dash="dash", line_color="black",
                  annotation_text="new attack families")
    fig.add_vrect(x0=D2S, x1=D2E, fillcolor="grey", opacity=0.15, line_width=0,
                  annotation_text="volume drift")
    fig.update_layout(xaxis=dict(range=[0, N_WIN - 1], title="window"),
                      yaxis=dict(range=[0, 1.02], title="F1"),
                      height=380, margin=dict(t=30, b=10),
                      legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, key=f"{tag}_f1_{w}")

    k = pd.DataFrame({"window": range(w + 1), "share": ks_frac[: w + 1]})
    fig2 = px.bar(k, x="window", y="share")
    fig2.update_traces(marker_color="#EE6C4D")
    fig2.add_hline(y=KS_FRAC, line_dash="dash", annotation_text="alarm threshold")
    fig2.update_layout(xaxis=dict(range=[-0.5, N_WIN - 0.5], title="window"),
                       yaxis=dict(range=[0, max(max(ks_frac), KS_FRAC) * 1.15],
                                  title="share of features shifted"),
                       height=260, margin=dict(t=30, b=10))
    st.plotly_chart(fig2, key=f"{tag}_ks_{w}")

    rows = stream[stream["window"] == w]
    table = []
    for g in ["Normal", "Old attacks", "New attacks"]:
        sub = rows[rows["group"] == g]
        if len(sub) == 0:
            continue
        rec = {"Traffic type": g, "Rows": len(sub)}
        for s, c in PRED_COL.items():
            rec[s] = f"{(sub[c] == sub['label']).mean():.0%}"
        table.append(rec)
    st.caption("Share of rows classified correctly in this window")
    st.dataframe(pd.DataFrame(table), hide_index=True)


def page_stream():
    st.title("Stream and drift monitor")
    st.write("Move the slider to inspect one window, or press Play to replay the whole stream.")
    c1, c2 = st.columns([3, 1])
    w = c1.slider("Window", 0, N_WIN - 1, 0)
    speed = c2.select_slider("Replay speed", ["Slow", "Medium", "Fast"], value="Medium")
    delay = {"Slow": 0.8, "Medium": 0.4, "Fast": 0.15}[speed]
    play = st.button("Play from the start")

    area = st.empty()
    if play:
        for i in range(N_WIN):
            with area.container():
                render_window(i, "play")
            time.sleep(delay)
    else:
        with area.container():
            render_window(w, "manual")


# ---------------------------------------------------------------- page 4
SOURCES = {
    "Default values (no real record)": None,
    "Random normal record": "Normal",
    "Random known attack (family seen in training)": "Old attacks",
    "Random new-family attack (unseen in training)": "New attacks",
}


def page_predict():
    st.title("Try a prediction")
    pre, m_initial, m_adapted = load_models()
    order = meta["raw_feature_order"]
    cat_cols = meta["cat_cols"]
    editable = meta["top_raw_features"][:8]

    st.write("Start from a record, edit the key fields, and see how both models classify it.")
    left, right = st.columns([1, 2])

    with left:
        source = st.selectbox("Starting point", list(SOURCES))
        if st.button("Draw a random record"):
            st.session_state["draw"] = int(np.random.randint(1_000_000))
        threshold = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05)

    seed = st.session_state.get("draw", 0)
    group = SOURCES[source]
    if group is None:
        base, truth, tag = dict(meta["defaults"]), None, "default"
    else:
        row = stream[stream["group"] == group].sample(1, random_state=seed).iloc[0]
        base = {c: row[c] for c in order}
        truth = f"{row['attack_cat']} (label {int(row['label'])})"
        tag = f"{group}-{seed}"

    values = dict(base)
    with right:
        st.subheader("Key fields")
        cols = st.columns(2)
        for i, f in enumerate(editable):
            col = cols[i % 2]
            key = f"{f}-{tag}"
            if f in cat_cols:
                options = sorted(stream[f].astype(str).unique())
                current = str(base[f])
                idx = options.index(current) if current in options else 0
                values[f] = col.selectbox(f, options, index=idx, key=key)
            else:
                start = float(base[f])
                values[f] = col.number_input(f, value=start, min_value=min(0.0, start),
                                             format="%.4f", key=key)

    X = pd.DataFrame([values])[order]
    for c in order:
        X[c] = X[c].astype(str) if c in cat_cols else X[c].astype(float)
    Xt = pre.transform(X)
    dm = xgb.DMatrix(Xt)
    p_initial = float(m_initial.predict(dm)[0])
    p_adapted = float(m_adapted.predict(dm)[0])

    st.divider()
    res1, res2 = st.columns(2)
    for col, name, p in [(res1, "Initial model (trained before the drift)", p_initial),
                         (res2, "Adapted model (after drift-triggered retraining)", p_adapted)]:
        with col:
            st.markdown(f"**{name}**")
            msg = f"{'ATTACK' if p >= threshold else 'NORMAL'}: attack probability {p:.1%}"
            (st.error if p >= threshold else st.success)(msg)
            st.progress(min(max(p, 0.0), 1.0))

    if truth:
        st.caption(f"Ground truth for the starting record: {truth}. "
                   "Your edits are not re-labelled, so treat this as a reference only.")
    st.caption(f"The other {len(order) - len(editable)} features keep the starting record's "
               "values. Default values are medians, so they may not form a realistic flow.")
    with st.expander("Show all values sent to the model"):
        st.dataframe(X.T.rename(columns={0: "value"}))


# ---------------------------------------------------------------- navigation
st.sidebar.title("NIDS Drift Monitor")
page = st.sidebar.radio("Go to", ["Overview", "Data explorer",
                                  "Stream and drift monitor", "Try a prediction"])
{"Overview": page_overview,
 "Data explorer": page_data,
 "Stream and drift monitor": page_stream,
 "Try a prediction": page_predict}[page]()