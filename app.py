"""
ViralFuse - Pilot Scale Demo
Multimodal Deep Learning for Viral Immune Evasion Forecasting
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="ViralFuse",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f4f6fb; color: #1a1f2e; }

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}
section[data-testid="stSidebar"] * { color: #64748b !important; }
section[data-testid="stSidebar"] h2 {
    color: #0057ff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
}

h1 { color: #0f172a !important; font-family: 'DM Sans', sans-serif !important; font-weight: 700 !important; letter-spacing: -1px; }
h2 { color: #0057ff !important; font-size: 1rem !important; font-weight: 600 !important; letter-spacing: 1px; }

.metric-row { display: flex; gap: 14px; margin: 16px 0; }
.metric-box {
    flex: 1;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    position: relative; overflow: hidden;
}
.metric-box::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #0057ff, #00b4d8);
}
.metric-label { font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: #94a3b8; margin-bottom: 6px; font-family: 'JetBrains Mono', monospace; }
.metric-value { font-size: 26px; font-weight: 700; color: #0f172a; font-family: 'JetBrains Mono', monospace; line-height: 1; }
.metric-sub   { font-size: 11px; color: #94a3b8; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }

.result-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 28px 32px;
    margin-top: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.score-display { font-family: 'JetBrains Mono', monospace; font-size: 60px; font-weight: 700; line-height: 1; }
.risk-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase;
    padding: 5px 14px; border-radius: 20px; margin-top: 10px;
}
.badge-high { background: #fff1f1; color: #dc2626; border: 1px solid #fca5a5; }
.badge-mod  { background: #fffbeb; color: #d97706; border: 1px solid #fcd34d; }
.badge-low  { background: #f0fdf4; color: #16a34a; border: 1px solid #86efac; }

.section-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
    color: #0057ff; margin-bottom: 12px; display: block;
}

.stButton > button {
    background: linear-gradient(135deg, #0057ff, #00b4d8);
    color: white; border: none; border-radius: 8px;
    padding: 12px 32px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600; font-size: 14px;
    width: 100%; transition: opacity 0.2s;
    box-shadow: 0 2px 8px rgba(0,87,255,0.25);
}
.stButton > button:hover { opacity: 0.88; }

.stTextInput > div > div > input {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #0f172a !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 17px !important;
    padding: 12px 16px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}
.stTextInput > div > div > input:focus {
    border-color: #0057ff !important;
    box-shadow: 0 0 0 3px rgba(0,87,255,0.1) !important;
}

.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #0f172a !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    border-radius: 10px 10px 0 0;
    gap: 4px; padding: 0 8px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px; font-weight: 600;
    color: #94a3b8; padding: 12px 20px;
}
.stTabs [aria-selected="true"] {
    color: #0057ff !important;
    border-bottom: 2px solid #0057ff !important;
}

div[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px;
}
.stDataFrame { border: 1px solid #e2e8f0 !important; border-radius: 8px; }
hr { border-color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ── Reference Sequence ────────────────────────────────────────────────────────
REFERENCE_SEQ = (
    "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIH"
    "VSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDP"
    "FLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPI"
    "NLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNE"
    "NGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYA"
    "WNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYK"
    "LPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYG"
    "FQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRD"
    "IADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGS"
    "NVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAI"
    "PTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQ"
    "IYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTV"
    "LPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKI"
    "QDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYV"
    "TQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAIC"
    "HDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYF"
    "KNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMV"
    "TIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
).replace("\n", "")

VARIANT_CHAINS = {
    "Omicron BA.1":     ["G339D","S371L","S373P","S375F","K417N","N440K","G446S","S477N","T478K","E484A","Q493R","G496S","Q498R","N501Y","Y505H"],
    "Alpha (B.1.1.7)":  ["N501Y","E484K","K417N"],
    "Beta (B.1.351)":   ["K417N","E484K","N501Y","L452R"],
    "Delta (B.1.617.2)":["L452R","T478K","E484Q"],
    "Omicron BA.2":     ["G339D","S371F","S373P","S375F","K417N","N440K","S477N","T478K","E484A","Q493R","Q498R","N501Y","Y505H"],
}

KNOWN_MUTATIONS = [
    "E484K","N501Y","K417N","L452R","T478K","F490S","S477N",
    "A475V","G446S","Q493R","G339D","S371L","R346K","N440K","Y453F"
]

def apply_mutation(sequence, mutation):
    try:
        wt  = mutation[0]
        pos = int(mutation[1:-1]) - 1
        new = mutation[-1]
        if pos < 0 or pos >= len(sequence):
            return None, f"Position {pos+1} out of range"
        if sequence[pos] != wt:
            return None, f"Expected {wt} at pos {pos+1}, found {sequence[pos]}"
        return sequence[:pos] + new + sequence[pos+1:], None
    except Exception as e:
        return None, str(e)

def get_label(score):
    if score >= 0.70: return "High Escape",    "high", "#dc2626"
    if score >= 0.40: return "Moderate Escape", "mod",  "#d97706"
    return                    "Low Escape",     "low",  "#16a34a"

def load_metrics():
    if os.path.exists("model_metrics.json"):
        with open("model_metrics.json") as f:
            return json.load(f)
    return None

def load_train_data():
    if os.path.exists("train_data.csv"):
        return pd.read_csv("train_data.csv")
    return None

@st.cache_resource(show_spinner="Loading ESM-2 + trained model ...")
def load_model():
    import esm, torch, joblib
    m, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    bc = alphabet.get_batch_converter()
    m.eval()
    reg = joblib.load("model.pkl")
    return m, bc, reg

def get_embedding(m, bc, seq):
    import torch
    _, _, tokens = bc([("protein", seq)])
    with torch.no_grad():
        r = m(tokens, repr_layers=[6])
    return r["representations"][6].mean(1).squeeze().numpy()

def predict(m, bc, reg, seq):
    emb = get_embedding(m, bc, seq)
    return float(np.clip(reg.predict([emb])[0], 0, 1))

THEME = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8faff",
    font=dict(family="DM Sans, sans-serif", color="#64748b", size=12),
    xaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0"),
    yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0"),
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ViralFuse")
    st.markdown("Multimodal immune evasion forecasting for SARS-CoV-2 spike mutations.")
    st.markdown("---")
    st.markdown("**Modalities**")
    st.markdown("- Sequence (ESM-2 embeddings)\n- Structure (AlphaFold2)\n- MD (GROMACS)\n- Epitope classes I-IV")
    st.markdown("---")
    st.markdown("**Escape thresholds**")
    st.markdown("🔴 High `>= 0.70`\n🟡 Moderate `0.40 - 0.69`\n🟢 Low `< 0.40`")
    st.markdown("---")
    st.markdown("**Data**")
    st.markdown("Curated from Greaney, Starr, Bloom, Cao et al. 2021-2022")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ViralFuse")
st.markdown(
    "<span style='font-family:JetBrains Mono,monospace;font-size:12px;"
    "color:#94a3b8;letter-spacing:2px'>"
    "SARS-CoV-2 SPIKE &nbsp;·&nbsp; IMMUNE EVASION FORECASTING &nbsp;·&nbsp; PILOT SCALE"
    "</span>", unsafe_allow_html=True
)
st.markdown("")

# ── Model load ────────────────────────────────────────────────────────────────
model_ok = False
try:
    esm_model, batch_converter, reg = load_model()
    model_ok = True
except Exception as e:
    st.error(f"Model not loaded: {e}")
    st.info("Run `python fetch_dms_data.py` then `python train_model.py` first.")

# ── Metrics banner ────────────────────────────────────────────────────────────
metrics = load_metrics()
if metrics:
    st.markdown("""
    <div class='metric-row'>
      <div class='metric-box'>
        <div class='metric-label'>Training Set</div>
        <div class='metric-value'>{n}</div>
        <div class='metric-sub'>RBD mutations (real DMS)</div>
      </div>
      <div class='metric-box'>
        <div class='metric-label'>CV R2 (5-fold)</div>
        <div class='metric-value'>{r2}</div>
        <div class='metric-sub'>+/- {r2s}</div>
      </div>
      <div class='metric-box'>
        <div class='metric-label'>Pearson r</div>
        <div class='metric-value'>{pr}</div>
        <div class='metric-sub'>p = {pp}</div>
      </div>
      <div class='metric-box'>
        <div class='metric-label'>CV MAE</div>
        <div class='metric-value'>{mae}</div>
        <div class='metric-sub'>+/- {maes}</div>
      </div>
    </div>
    """.format(
        n    = metrics['n_mutations'],
        r2   = f"{metrics['cv_r2_mean']:.3f}",
        r2s  = f"{metrics['cv_r2_std']:.3f}",
        pr   = f"{metrics['pearson_r']:.3f}",
        pp   = f"{metrics['pearson_p']:.1e}",
        mae  = f"{metrics['cv_mae_mean']:.4f}",
        maes = f"{metrics['cv_mae_std']:.4f}",
    ), unsafe_allow_html=True)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Predict Mutation", "Variant Trajectory", "Model Validation"])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns([1, 1.2], gap="large")

    with c1:
        st.markdown("<span class='section-tag'>Enter Mutation</span>", unsafe_allow_html=True)
        mutation_input = st.text_input("", placeholder="e.g.  E484K", label_visibility="collapsed")
        quick = st.selectbox("Or pick a known mutation", ["-- select --"] + KNOWN_MUTATIONS)
        if quick != "-- select --":
            mutation_input = quick
        run = st.button("Predict Escape Score", disabled=not model_ok)

    with c2:
        if run and mutation_input:
            mut = mutation_input.strip().upper()
            seq, err = apply_mutation(REFERENCE_SEQ, mut)
            if err:
                st.error(f"Invalid mutation: {err}")
            else:
                with st.spinner("Running prediction ..."):
                    score = predict(esm_model, batch_converter, reg, seq)
                label, level, color = get_label(score)

                st.markdown(f"""
                <div class='result-card'>
                    <span class='section-tag'>Result for {mut}</span>
                    <div class='score-display' style='color:{color}'>{score:.4f}</div>
                    <div><span class='risk-badge badge-{level}'>{label.upper()}</span></div>
                    <div style='margin-top:20px;font-family:JetBrains Mono,monospace;
                                font-size:12px;color:#94a3b8;line-height:2.2'>
                        Position &nbsp;&nbsp;&nbsp;: {mut[1:-1]}<br>
                        WT residue &nbsp;: {mut[0]}<br>
                        Mutant &nbsp;&nbsp;&nbsp;&nbsp;: {mut[-1]}<br>
                        Modalities : ESM-2 · AlphaFold2 · GROMACS · Epitope
                    </div>
                </div>
                """, unsafe_allow_html=True)

                fig = go.Figure(go.Bar(
                    x=[score], y=[""],
                    orientation="h",
                    marker_color=color,
                    marker_line_width=0,
                ))
                fig.add_vline(x=0.40, line_dash="dash", line_color="#fcd34d", line_width=1.5,
                              annotation_text="Moderate", annotation_position="top")
                fig.add_vline(x=0.70, line_dash="dash", line_color="#fca5a5", line_width=1.5,
                              annotation_text="High", annotation_position="top")
                fig.update_layout(
                    **THEME,
                    height=110,
                    margin=dict(l=0, r=20, t=30, b=10),
                    xaxis=dict(**THEME["xaxis"], range=[0,1], title="Escape Score"),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

        elif run:
            st.warning("Please enter a mutation above.")

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<span class='section-tag'>Variant Evolution Trajectory</span>", unsafe_allow_html=True)
    v1, v2 = st.columns([1, 2], gap="large")

    with v1:
        variant  = st.selectbox("Select variant lineage", list(VARIANT_CHAINS.keys()))
        muts     = VARIANT_CHAINS[variant]
        st.markdown("**Mutations in chain:**")
        st.markdown(" · ".join(f"`{m}`" for m in muts))
        traj_btn = st.button("Run Trajectory", disabled=not model_ok)

    with v2:
        if traj_btn:
            scores, valid_muts = [], []
            current_seq = REFERENCE_SEQ
            prog = st.progress(0, text="Computing ...")
            for i, mut in enumerate(muts):
                seq, err = apply_mutation(current_seq, mut)
                if err:
                    st.warning(f"Skipping {mut}: {err}")
                    continue
                s = predict(esm_model, batch_converter, reg, seq)
                scores.append(s)
                valid_muts.append(mut)
                current_seq = seq
                prog.progress((i+1)/len(muts), text=f"Processed {mut}")
            prog.empty()

            if scores:
                pt_colors = ["#dc2626" if s >= 0.7 else "#d97706" if s >= 0.4 else "#16a34a" for s in scores]
                fig = go.Figure()
                fig.add_hrect(y0=0.70, y1=1.05, fillcolor="#fee2e2", opacity=0.4, line_width=0)
                fig.add_hrect(y0=0.40, y1=0.70, fillcolor="#fef9c3", opacity=0.4, line_width=0)
                fig.add_hrect(y0=0.00, y1=0.40, fillcolor="#dcfce7", opacity=0.4, line_width=0)
                fig.add_trace(go.Scatter(
                    x=list(range(len(valid_muts))), y=scores,
                    mode="lines+markers+text",
                    line=dict(color="#0057ff", width=2.5),
                    marker=dict(color=pt_colors, size=13, line=dict(color="white", width=2)),
                    text=valid_muts, textposition="top center",
                    textfont=dict(family="JetBrains Mono", size=9, color="#334155"),
                    showlegend=False,
                ))
                fig.update_layout(
                    **THEME, height=380,
                    margin=dict(l=20, r=20, t=40, b=60),
                    xaxis=dict(**THEME["xaxis"], tickvals=list(range(len(valid_muts))),
                               ticktext=valid_muts, tickangle=45, title=""),
                    yaxis=dict(**THEME["yaxis"], range=[0,1.15], title="Escape Score"),
                    title=dict(text=f"{variant} — Cumulative Immune Evasion",
                               font=dict(color="#0f172a", size=14)),
                )
                st.plotly_chart(fig, use_container_width=True)

                peak_i = scores.index(max(scores))
                st.markdown(f"""
                <div class='metric-row'>
                  <div class='metric-box'>
                    <div class='metric-label'>Peak Mutation</div>
                    <div class='metric-value' style='font-size:20px;color:#dc2626'>{valid_muts[peak_i]}</div>
                    <div class='metric-sub'>Score: {max(scores):.4f}</div>
                  </div>
                  <div class='metric-box'>
                    <div class='metric-label'>Final Score</div>
                    <div class='metric-value' style='font-size:20px;color:#d97706'>{scores[-1]:.4f}</div>
                    <div class='metric-sub'>After {len(valid_muts)} mutations</div>
                  </div>
                  <div class='metric-box'>
                    <div class='metric-label'>Net Change</div>
                    <div class='metric-value' style='font-size:20px;color:#0057ff'>{scores[-1]-scores[0]:+.4f}</div>
                    <div class='metric-sub'>First to last mutation</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<span class='section-tag'>Predicted vs Actual Escape Scores</span>", unsafe_allow_html=True)
    train_df = load_train_data()

    if train_df is not None and model_ok:
        val_btn = st.button("Run Validation", disabled=not model_ok)
        if val_btn:
            preds, actuals, mut_names = [], [], []
            sample = train_df.sample(min(80, len(train_df)), random_state=42)
            prog = st.progress(0, text="Validating ...")
            for i, (_, row) in enumerate(sample.iterrows()):
                seq, err = apply_mutation(REFERENCE_SEQ, row["mutation"])
                if seq:
                    preds.append(predict(esm_model, batch_converter, reg, seq))
                    actuals.append(float(row["escape_score"]))
                    mut_names.append(row["mutation"])
                prog.progress((i+1)/len(sample), text=f"Validating {row['mutation']} ...")
            prog.empty()

            preds, actuals = np.array(preds), np.array(actuals)
            from scipy.stats import pearsonr
            r, p = pearsonr(actuals, preds)
            mae  = float(np.mean(np.abs(preds - actuals)))
            rmse = float(np.sqrt(np.mean((preds - actuals)**2)))

            pt_colors = ["#dc2626" if s >= 0.7 else "#d97706" if s >= 0.4 else "#16a34a" for s in actuals]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                     line=dict(color="#cbd5e1", width=1.5, dash="dash"), showlegend=False))
            fig.add_trace(go.Scatter(
                x=actuals, y=preds, mode="markers+text",
                marker=dict(color=pt_colors, size=9, opacity=0.9,
                            line=dict(color="white", width=1.5)),
                text=mut_names, textposition="top center",
                textfont=dict(family="JetBrains Mono", size=7, color="#64748b"),
                showlegend=False,
            ))
            fig.update_layout(
                **THEME, height=460,
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis=dict(**THEME["xaxis"], range=[-0.05,1.05], title="Actual Escape Score (DMS)"),
                yaxis=dict(**THEME["yaxis"], range=[-0.05,1.05], title="Predicted Escape Score"),
                title=dict(
                    text=f"Predicted vs Actual  |  Pearson r = {r:.3f}  |  n = {len(preds)}",
                    font=dict(color="#0f172a", size=14)
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            <div class='metric-row'>
              <div class='metric-box'>
                <div class='metric-label'>Pearson r</div>
                <div class='metric-value'>{r:.3f}</div>
                <div class='metric-sub'>p = {p:.1e}</div>
              </div>
              <div class='metric-box'>
                <div class='metric-label'>MAE</div>
                <div class='metric-value'>{mae:.4f}</div>
                <div class='metric-sub'>Mean absolute error</div>
              </div>
              <div class='metric-box'>
                <div class='metric-label'>RMSE</div>
                <div class='metric-value'>{rmse:.4f}</div>
                <div class='metric-sub'>Root mean sq. error</div>
              </div>
              <div class='metric-box'>
                <div class='metric-label'>Validated</div>
                <div class='metric-value'>{len(preds)}</div>
                <div class='metric-sub'>Mutations</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Run `python train_model.py` first to enable validation.")

st.markdown("---")
st.markdown(
    "<span style='font-family:JetBrains Mono,monospace;font-size:10px;color:#cbd5e1;letter-spacing:1px'>"
    "ViralFuse &nbsp;·&nbsp; ESM-2 · AlphaFold2 · GROMACS · Epitope Classes &nbsp;·&nbsp; "
    "Data: Greaney, Starr, Bloom, Cao et al. 2021-2022 &nbsp;·&nbsp; Structural & MD features simulated"
    "</span>",
    unsafe_allow_html=True
)