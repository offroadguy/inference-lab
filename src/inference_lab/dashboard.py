"""Read-only portfolio dashboard. All displayed results come from the local journal."""
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from inference_lab.config import Plan
from inference_lab.report import compare, summarize
from inference_lab.store import read_snapshot

st.set_page_config(page_title="Inference Lab · Engineering Observatory", page_icon="◈", layout="wide")
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {color-scheme:dark;}
.stApp {background:#0b1018;color:#e8edf5;font-family:'DM Sans',sans-serif;}
header[data-testid="stHeader"] {background:#0b1018dd;}
.block-container {padding-top:4rem;max-width:1500px;padding-bottom:3rem;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.035em;}
h1 {font-size:2.6rem!important;line-height:1.15!important;font-weight:600!important;}
h3 {font-size:1.15rem!important;}
[data-testid="stSidebar"] {background:#0f1621;border-right:1px solid #253042;}
[data-testid="stSidebar"] .block-container {padding-top:2rem;}
[data-testid="stMetric"] {background:#121c29;border:1px solid #29374b;border-radius:14px;padding:20px 22px;min-height:128px;}
[data-testid="stMetricLabel"] {color:#a6b5ca;font-size:.83rem;}
[data-testid="stMetricValue"] {font-family:'Space Grotesk',sans-serif;font-size:2rem;color:#f3f6fc;}
[data-testid="stCaptionContainer"] {color:#a0afc4;}
[data-baseweb="select"]>div {background:#152131;color:#e8edf5;border-color:#33455c;}
button[data-baseweb="tab"] {color:#aab8ca;padding:12px 18px;font-size:.9rem;}
button[data-baseweb="tab"][aria-selected="true"] {color:#62e0c2;}
[data-baseweb="tab-highlight"] {background:#62e0c2;}
[data-baseweb="tab-border"] {background:#263347;}
[data-testid="stExpander"] {background:#101a27;border-color:#29374b;border-radius:12px;}
[data-testid="stDownloadButton"] button {background:#152b30;color:#9cf0dc;border:1px solid #2c615b;border-radius:9px;}
[data-testid="stAlert"] {background:#182431;border:1px solid #344e65;color:#e0ebf8;}
.brand {font-family:'Space Grotesk',sans-serif;font-size:1.25rem;font-weight:600;letter-spacing:-.04em;margin:8px 0 5px;}
.brand span {color:#63dfc0;font-size:1.55rem;margin-right:10px;}
.overline {color:#63dfc0;font-size:.68rem;font-weight:700;letter-spacing:.18em;text-transform:uppercase;margin-bottom:14px;}
.subtitle {color:#9eafc5;font-size:1rem;max-width:740px;line-height:1.65;margin:0 0 24px;}
.pill {display:inline-block;font-size:.68rem;letter-spacing:.07em;font-weight:600;border:1px solid #3c5364;
 border-radius:6px;padding:5px 9px;color:#b9ccdc;background:#152331;margin:0 6px 7px 0;}
.pill.green {border-color:#2a6758;color:#83e8cc;background:#102e29;}
.pill.amber {border-color:#705c36;color:#efce87;background:#2a241a;}
.note {background:linear-gradient(110deg,#172536,#101b28);border:1px solid #30465e;border-left:3px solid #69b5f8;
 border-radius:10px;padding:16px 20px;font-size:.86rem;color:#bdccdf;line-height:1.6;margin:14px 0 22px;}
.panel {border:1px solid #2c3a4e;background:#111b28;border-radius:13px;padding:22px;margin-bottom:16px;min-height:130px;}
.panel h4 {font-family:'Space Grotesk',sans-serif;font-size:1.05rem;color:#ecf3fc;margin:0 0 9px;}
.panel p {font-size:.85rem;color:#aabbd0;line-height:1.65;margin:0;}
.step {display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #263449;padding:15px 0;font-size:.87rem;}
.step span {color:#93a6bd;}.step b {color:#9fe6d2;font-weight:500;}
@media (max-width:1100px) {
 [data-testid="stHorizontalBlock"] {flex-wrap:wrap;}
 [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {flex:1 1 calc(50% - 1rem)!important;min-width:210px;}
 .block-container {padding-left:1.5rem;padding-right:1.5rem;}
}
@media (max-width:600px) {
 [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {flex-basis:100%!important;}
 h1 {font-size:2rem!important;}
}
.footer {border-top:1px solid #253247;padding-top:18px;margin-top:28px;color:#8597af;font-size:.74rem;line-height:1.7;}
</style>
""")

COLORS = {"success": "#61dfbe", "rejected": "#edbd70", "error": "#f08391", "incomplete": "#9c9beb"}


def label(run):
    p = run["plan"]
    if p["simulated_fault"] != "none":
        name = "Controlled failure"
    elif "recovery" in p["hypothesis"].lower():
        name = "Recovery validation"
    elif p["concurrency"] > 1:
        name = "Concurrency experiment"
    else:
        name = "Baseline experiment"
    return f"{name} · {p['rate']:g} req/s · C{p['concurrency']} · {run['id'][:6]}"


def chart(fig, height=300):
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="#111b28", plot_bgcolor="#111b28",
                      margin=dict(l=20, r=20, t=25, b=35), font=dict(family="DM Sans, sans-serif", color="#aabbd0"),
                      legend=dict(orientation="h", y=1.15), hoverlabel=dict(bgcolor="#24364c"))
    fig.update_xaxes(gridcolor="#243144", zeroline=False)
    fig.update_yaxes(gridcolor="#243144", zeroline=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def ms(value):
    return "Unavailable" if value is None else f"{value * 1000:,.1f} ms"


def number(value, unit=""):
    return "Unavailable" if value is None else f"{value:,.2f}{unit}"


def panel(title, text):
    st.html(f'<div class="panel"><h4>{escape(title)}</h4><p>{escape(text)}</p></div>')


with st.sidebar:
    st.html('<div class="brand"><span>◈</span> INFERENCE LAB</div><div class="overline">Engineering observatory</div>')
    st.caption("Venkata Ganji · Independent portfolio")
    st.divider()
    st.markdown("**Experiment workspace**")
    source = st.selectbox("Evidence source", ["All runs", "Simulated", "Modal"])
    st.caption("Simulation validates the harness. Remote runs establish inference evidence.")
    st.divider()
    st.markdown("**Built for explainability**")
    st.caption("01 / Explicit spending approval\n\n02 / Durable request evidence\n\n03 / Bounded resource lifetime\n\n04 / Verifiable cleanup")
    st.divider()
    st.link_button("View source ↗", "https://github.com/offroadguy/inference-lab", width="stretch")
    st.caption("Python · vLLM · Modal\n\nSQLite · Streamlit · Plotly")


@st.fragment(run_every="5s")
def render():
    runs, requests = read_snapshot()
    if source != "All runs":
        runs = [r for r in runs if r["plan"]["provider"] == source.lower()]
    st.html('<div class="overline">Measure / Understand / Improve</div>')
    st.title("Inference, made observable.")
    st.html('<p class="subtitle">From the first streamed response to the last resource shutdown. '
            'An engineering notebook for performance, capacity, and recovery.</p>')
    if not runs:
        st.info("No experiments in this view yet. Create a plan in the CLI or select another evidence source.")
        st.code("uv run python -m inference_lab.demo", language="bash")
        return
    by_id = {r["id"]: r for r in runs}
    chosen = st.selectbox("Inspect an experiment", list(by_id), format_func=lambda rid: label(by_id[rid]))
    run = by_id[chosen]
    p = run["plan"]
    rows = requests[chosen]
    summary = summarize(run, rows)
    simulated = p["provider"] == "simulated"
    observed = datetime.fromtimestamp(run["created_at"], timezone.utc).strftime("%b %d, %Y · %H:%M UTC")
    st.html(f'<span class="pill {"amber" if simulated else "green"}">'
            f'{"SIMULATED EVIDENCE" if simulated else "REMOTE EXPERIMENT"}</span>'
            f'<span class="pill">{escape(p["workload"].upper())}</span>'
            f'<span class="pill">{escape(run["status"].upper())}</span>'
            f'<span class="pill">{escape(observed)}</span>')
    if simulated:
        st.html('<div class="note"><strong>Software demonstration · no GPU was used.</strong> '
                'These are real observations of a synthetic streaming fixture. They demonstrate measurement, '
                'admission control, and recovery—not model speed, GPU capacity, or cost savings.</div>')
    else:
        st.caption("Client measurements from a remote run. Billing and unexposed device metrics remain unverified.")

    cols = st.columns(4)
    cols[0].metric("REQUESTS COMPLETED", f"{summary['successful']} / {summary['offered']}",
                   help="Successful steady-state requests / offered requests, including client rejections.")
    cols[1].metric("P95 RESPONSE LATENCY", ms(summary["latency_p95_s"]), help="Successful requests only.")
    cols[2].metric("ACHIEVED THROUGHPUT", number(summary["successful_rps"], " req/s"))
    cols[3].metric("P95 FIRST RESPONSE", ms(summary["ttft_p95_s"]), help="First nonempty content chunk, not server TTFT.")
    st.write("")
    overview, comparison, reliability, evidence = st.tabs(
        ["Performance", "Compare experiments", "Reliability & cost", "Evidence & methodology"])

    with overview:
        left, right = st.columns([1.65, 1])
        successes = [r for r in rows if r["phase"] == "steady" and r["status"] == "success"]
        with left:
            st.subheader("Response latency")
            st.caption(f"{len(successes)} successful samples · milliseconds · client end-to-end")
            if successes:
                latencies = [r["latency_s"] * 1000 for r in successes]
                fig = go.Figure(go.Histogram(x=latencies, nbinsx=14, marker_color="#60dcbf", opacity=.85))
                fig.add_vline(x=summary["latency_p95_s"] * 1000, line_dash="dot", line_color="#b6a4ff",
                              annotation_text="p95", annotation_font_color="#d4c9ff")
                fig.update_layout(xaxis_title="Response latency (ms)", yaxis_title="Requests", bargap=.12)
                chart(fig)
            else:
                panel("No successful latency samples", "Errors and incomplete requests remain available in the evidence table.")
        with right:
            st.subheader("Request outcomes")
            st.caption("Client admission rejections are included")
            counts = Counter(r["status"] for r in rows if r["phase"] == "steady")
            if counts:
                fig = go.Figure(go.Pie(labels=[s.title() for s in counts], values=list(counts.values()), hole=.73,
                                      marker_colors=[COLORS.get(s, "#7296c0") for s in counts],
                                      textinfo="none", sort=False))
                fig.update_layout(annotations=[dict(text=f"<b>{sum(counts.values())}</b><br>offered",
                                                    x=.5, y=.5, showarrow=False, font_size=20)])
                chart(fig)
            else:
                panel("Awaiting requests", "The experiment has not recorded steady-state arrivals yet.")
        st.subheader("The experiment")
        a, b, c = st.columns(3)
        with a:
            panel("Hypothesis", p["hypothesis"])
        with b:
            panel("Load envelope", f"{p['rate']:g} requests/sec · {p['concurrency']} concurrent · "
                  f"{p['max_tokens']} maximum output tokens. Regular arrivals with bounded admission.")
        with c:
            outcome = "Within configured targets" if summary["within_targets"] else "Targets not demonstrated"
            panel(outcome, f"p95 target ≤ {p['target_p95_s']:g}s; error/rejection target ≤ "
                  f"{p['target_error_rate']:.0%}. Small samples do not establish production SLOs.")
        if successes:
            st.subheader("Response timeline")
            base = min(r["started_at"] for r in successes)
            fig = go.Figure()
            for field, name, color in [("latency_s", "End-to-end", "#62dfc1"),
                                       ("ttft_s", "First content", "#9c9bef")]:
                valid = [r for r in successes if r.get(field) is not None]
                fig.add_trace(go.Scatter(x=[r["started_at"] - base for r in valid],
                                        y=[r[field] * 1000 for r in valid], mode="lines+markers",
                                        name=name, line=dict(color=color, width=2), marker_size=6))
            fig.update_layout(xaxis_title="Seconds since first successful request started", yaxis_title="Latency (ms)")
            chart(fig, 270)

    with comparison:
        st.subheader("One change. Inspect the tradeoff.")
        st.caption("Compare exact configurations before attributing an improvement.")
        candidates = [r for r in runs if r["id"] != chosen]
        if not candidates:
            st.info("Record a second experiment to enable comparison.")
        else:
            baseline_id = st.selectbox("Baseline experiment", [r["id"] for r in candidates],
                                       format_func=lambda rid: label(by_id[rid]))
            baseline = by_id[baseline_id]
            baseline_summary = summarize(baseline, requests[baseline_id])
            delta = compare({"run": baseline, "summary": baseline_summary}, {"run": run, "summary": summary})
            changes = delta["controlled_changes"]
            if delta["single_variable"]:
                st.success("One configuration variable changed. Review repetitions and quality before drawing conclusions.")
            elif not changes:
                st.info("Identical recorded configuration: useful as a repetition, not a controlled change.")
            else:
                st.warning(f"{len(changes)} configuration variables differ. This is a confounded comparison.")
            if changes:
                st.dataframe(pd.DataFrame([{"Setting": k, "Baseline": str(a), "Selected": str(b)}
                                           for k, (a, b) in changes.items()]), hide_index=True, width="stretch")
            a, b = st.columns(2)
            with a:
                st.subheader("Successful throughput")
                fig = go.Figure(go.Bar(x=["Baseline", "Selected"],
                                      y=[baseline_summary["successful_rps"], summary["successful_rps"]],
                                      marker_color=["#8294b5", "#61dfbe"]))
                fig.update_layout(yaxis_title="Successful requests / second")
                chart(fig)
            with b:
                st.subheader("Tail latency")
                fig = go.Figure(go.Bar(x=["Baseline", "Selected"],
                                      y=[s["latency_p95_s"] * 1000 if s["latency_p95_s"] is not None else None
                                         for s in [baseline_summary, summary]], marker_color=["#8294b5", "#b5a5fb"]))
                fig.update_layout(yaxis_title="p95 response latency (ms)")
                chart(fig)
            st.caption("No automatic causal or significance claim. Synthetic throughput cannot size GPU infrastructure.")
        st.subheader("Experiment inventory")
        inventory = []
        for item in runs:
            s = summarize(item, requests[item["id"]])
            inventory.append({"Experiment": label(item), "Evidence": item["plan"]["provider"],
                              "Offered rate (req/s)": item["plan"]["rate"],
                              "Successes": s["successful"], "Offered": s["offered"],
                              "Throughput (req/s)": s["successful_rps"], "Cleanup": item["shutdown"]})
        st.dataframe(pd.DataFrame(inventory), hide_index=True, width="stretch")
        st.subheader("Offered load vs. achieved throughput")
        fig = go.Figure()
        for provider, color in [("simulated", "#edbd70"), ("modal", "#61dfbe")]:
            group = [item for item in inventory if item["Evidence"] == provider]
            if group:
                fig.add_trace(go.Scatter(x=[item["Offered rate (req/s)"] for item in group],
                                        y=[item["Throughput (req/s)"] for item in group],
                                        text=[item["Experiment"] for item in group], mode="markers", name=provider,
                                        marker=dict(color=color, size=11, opacity=.8)))
        fig.update_layout(xaxis_title="Configured offered requests / second",
                          yaxis_title="Successful requests / second")
        chart(fig)
        st.caption("Different workloads and concurrency bounds are separate experiments, not a single capacity curve.")

    with reliability:
        a, b = st.columns([1, 1.4])
        with a:
            st.subheader("Lifecycle checkpoints")
            stages = [("Plan persisted", True), ("Spending approved", bool(run["approval"])),
                      ("Endpoint validated", "validated" in run["stages"]),
                      ("Warm-up complete", "warmup" in run["stages"]),
                      ("Benchmark complete", "benchmark" in run["stages"])]
            for title, done in stages:
                status = "Not required · simulation" if title == "Spending approved" and simulated else (
                    "Recorded" if done else "Not recorded")
                st.html(f'<div class="step"><span>{title}</span><b>{status}</b></div>')
            st.html(f'<div class="step"><span>Resource shutdown</span><b>{escape(run["shutdown"].title())}</b></div>')
        with b:
            st.subheader("Failure evidence")
            failures = Counter(r.get("error") or r["status"] for r in rows
                               if r["status"] in {"error", "rejected", "incomplete"})
            if failures:
                fig = go.Figure(go.Bar(x=list(failures.values()), y=list(failures.keys()), orientation="h",
                                      marker_color="#efbd76"))
                fig.update_layout(xaxis_title="Requests", yaxis_title=None)
                chart(fig, 260)
            else:
                panel("No recorded failures", "This is an observation of the selected run, not a reliability guarantee.")
            st.caption("Error codes are evidence, not root cause. A timeout alone cannot prove OOM or exhausted credits.")
        st.write("")
        a, b, c = st.columns(3)
        estimate = Plan(**p).estimate()
        a.metric("RESOURCE COST ESTIMATE", number(summary["resource_cost_estimate_usd"], " USD"))
        b.metric("PLANNED UPPER ESTIMATE", number(estimate["upper_usd"], " USD"))
        c.metric("PROVIDER BILLING", "Not verified")
        st.caption("Simulation costs $0 in cloud resources. Remote estimates exclude uncertain charges and are not billing caps.")
        with st.expander("Resource limits & spending assumptions"):
            st.json({"maximum_authorized_usd": p["max_spend_usd"], "runtime_seconds": p["runtime_s"], **estimate})
        st.subheader("Server telemetry")
        telemetry_rows = [{"at": datetime.fromtimestamp(t["at"], timezone.utc), **t["values"]}
                          for t in run["telemetry"] if t["values"]]
        if telemetry_rows:
            st.line_chart(pd.DataFrame(telemetry_rows).set_index("at"))
        else:
            panel("Device telemetry unavailable", "No server queue or KV-cache samples were exposed in this run. "
                  "GPU utilization and allocated memory are not inferred from client timing.")

    with evidence:
        st.subheader("Trace every claim to a request.")
        st.caption(f"Run {chosen} · raw records include warm-up, errors and incomplete responses.")
        fields = ["id", "phase", "status", "error", "prompt_tokens", "output_tokens", "ttft_s", "latency_s"]
        if rows:
            st.dataframe(pd.DataFrame(rows).reindex(columns=fields), hide_index=True, width="stretch")
        a, b = st.columns(2)
        a.download_button("↓ Export full evidence · JSON", json.dumps({"plan": p, "summary": summary, "requests": rows},
                           indent=2), file_name=f"{chosen}.json", mime="application/json", width="stretch")
        b.download_button("↓ Export request table · CSV", pd.DataFrame(rows).to_csv(index=False),
                           file_name=f"{chosen}.csv", mime="text/csv", width="stretch")
        with st.expander("Measurement definitions & limitations", expanded=True):
            for limitation in summary["limitations"]:
                st.markdown(f"- {limitation}")
        with st.expander("Pinned configuration & complete summary"):
            st.json({"plan": p, "summary": summary})
        panel("Engineering ownership", "Independent portfolio project by Venkata Ganji, developed with Codex assistance. "
              "Project code owns orchestration, approval, measurements, persistence and recovery. "
              "Modal supplies compute; vLLM supplies inference; Qwen supplies the model.")
    st.html('<div class="footer">INFERENCE LAB / LOCAL OBSERVATORY<br>'
            'Evidence before claims. No spending controls in the dashboard. Refreshes every 5 seconds.</div>')


render()
