"""Read-only portfolio dashboard. All displayed results come from the local journal."""
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from inference_lab.config import Plan
from inference_lab.report import compare, summarize
from inference_lab.models import PROFILES
from inference_lab.reference import crosscheck
from inference_lab.store import read_snapshot

st.set_page_config(page_title="Inference Lab · Benchmark Console", page_icon="◈", layout="wide")
st.html("""
<style>
:root {color-scheme:light;}
.stApp {background:#f5f6f8;color:#192330;font-family:Arial,sans-serif;}
header[data-testid="stHeader"] {background:#f5f6f8ee;}
.block-container {padding-top:4rem;max-width:1560px;padding-bottom:2rem;}
h1,h2,h3 {color:#192330;letter-spacing:-.035em;}
h1 {font-size:1.75rem!important;line-height:1.2!important;font-weight:700!important;}
h3 {font-size:1rem!important;}
[data-testid="stSidebar"] {background:#eef0f4;border-right:1px solid #d7dce5;}
[data-testid="stSidebar"] .block-container {padding-top:1.5rem;}
[data-testid="stVerticalBlock"] {gap:.75rem;}
[data-testid="stMetric"] {background:#fff;border:1px solid #d9dfe8;border-top:3px solid #0d9488;border-radius:6px;padding:12px 16px;min-height:94px;}
[data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"] {border-top-color:#7c3aed;}
[data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"] {border-top-color:#2563eb;}
[data-testid="stColumn"]:nth-child(4) [data-testid="stMetric"] {border-top-color:#7c3aed;}
[data-testid="stMetricLabel"] {color:#556174;font-size:.72rem;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}
[data-testid="stMetricValue"] {font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:1.65rem;color:#182537;}
[data-testid="stCaptionContainer"] {color:#5b687a;}
[data-baseweb="select"]>div {background:#fff;color:#192330;border-color:#cbd3df;border-radius:5px;}
button[data-baseweb="tab"] {color:#526075;padding:10px 12px;font:12px ui-monospace,SFMono-Regular,Menlo,monospace;}
button[data-baseweb="tab"][aria-selected="true"] {color:#1d4ed8;background:#e8efff;border-radius:5px 5px 0 0;}
[data-baseweb="tab-highlight"] {background:#2563eb;}
[data-baseweb="tab-border"] {background:#d7dce5;}
[data-testid="stExpander"] {background:#fff;border-color:#d9dfe8;border-radius:6px;}
[data-testid="stDownloadButton"] button {background:#fff;color:#1d4ed8;border:1px solid #c3d2f3;border-radius:5px;}
.brand {font:700 1.05rem ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:-.05em;margin:4px 0 7px;color:#17243b;}
.brand span {color:#2563eb;margin-right:8px;}
.overline {color:#526075;font:11px ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.08em;margin-bottom:8px;}
.subtitle {color:#647084;font-size:.85rem;line-height:1.5;margin:0 0 12px;}
.pill {display:inline-block;font:10px ui-monospace,SFMono-Regular,Menlo,monospace;border:1px solid #d4dce8;
 border-radius:4px;padding:4px 7px;color:#526075;background:#fff;margin:0 5px 3px 0;}
.pill.green {border-color:#a0d9c8;color:#07644d;background:#e8f7f1;}
.pill.amber {border-color:#ebcb8c;color:#85520b;background:#fff7e6;}
.note {background:#fff8e9;border:1px solid #eddbb3;border-left:3px solid #d97706;
 border-radius:5px;padding:9px 12px;font-size:.78rem;color:#75501a;line-height:1.5;margin:8px 0 12px;}
.panel {border:1px solid #d9dfe8;background:#fff;border-radius:6px;padding:16px;margin-bottom:12px;min-height:100px;}
.panel h4 {font-size:.9rem;color:#24324a;margin:0 0 7px;}
.panel p {font-size:.8rem;color:#5b687a;line-height:1.6;margin:0;}
.step {display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #d9dfe8;padding:12px 0;font-size:.8rem;gap:12px;}
.step span {color:#526075;}.step b {color:#087961;font-weight:500;}
.console-status {display:flex;gap:16px;flex-wrap:wrap;font:11px ui-monospace,SFMono-Regular,Menlo,monospace;color:#526075;border-bottom:1px solid #d7dce5;padding:0 0 10px;margin-bottom:12px;}
.console-status b {color:#2563eb;font-weight:600;}
.key {font:11px ui-monospace,SFMono-Regular,Menlo,monospace;line-height:2.3;color:#526075;}
.key span {display:inline-block;width:7px;height:7px;margin-right:8px;border-radius:2px;}
.footer {border-top:1px solid #d7dce5;padding-top:12px;margin-top:20px;color:#647084;font:10px ui-monospace,SFMono-Regular,Menlo,monospace;line-height:1.7;}
@media (max-width:1100px) {
 [data-testid="stHorizontalBlock"] {flex-wrap:wrap;}
 [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {flex:1 1 calc(50% - 1rem)!important;min-width:210px;}
 .block-container {padding-left:1.25rem;padding-right:1.25rem;}
}
@media (max-width:600px) {
 [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {flex-basis:100%!important;}
}
</style>
""")

COLORS = {"success": "#0d9488", "rejected": "#d97706", "error": "#e11d48", "incomplete": "#7c3aed"}


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
    return f"{p['model'].split('/')[-1]} · {name} · {p['rate']:g} req/s · {run['id'][:6]}"


def chart(fig, height=300):
    fig.update_layout(template="plotly_white", height=height, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                      margin=dict(l=20, r=20, t=25, b=35), font=dict(family="Menlo, monospace", size=11, color="#526075"),
                      legend=dict(orientation="h", y=1.15), hoverlabel=dict(bgcolor="#ffffff", font_color="#192330"))
    fig.update_xaxes(gridcolor="#e9edf3", zeroline=False)
    fig.update_yaxes(gridcolor="#e9edf3", zeroline=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def ms(value):
    return "Unavailable" if value is None else f"{value * 1000:,.1f} ms"


def number(value, unit=""):
    return "Unavailable" if value is None else f"{value:,.2f}{unit}"


def panel(title, text):
    st.html(f'<div class="panel"><h4>{escape(title)}</h4><p>{escape(text)}</p></div>')


with st.sidebar:
    st.html('<div class="brand"><span>&gt;_</span> inference.lab</div><div class="overline">BENCHMARK CONSOLE / v0.2</div>')
    st.caption("Venkata Ganji · Independent portfolio")
    st.divider()
    st.markdown("**Workspace**")
    source = st.selectbox("Evidence source", ["All runs", "Simulated", "Modal"])
    st.caption("Simulation validates the harness. Remote runs establish inference evidence.")
    st.divider()
    st.markdown("**Signal legend**")
    st.html('<div class="key"><span style="background:#2563eb"></span>Requests / throughput<br>'
            '<span style="background:#7c3aed"></span>Latency / timing<br>'
            '<span style="background:#0d9488"></span>Success / verified<br>'
            '<span style="background:#d97706"></span>Simulation / pending<br>'
            '<span style="background:#e11d48"></span>Errors / failures</div>')
    st.divider()
    st.link_button("View source ↗", "https://github.com/offroadguy/inference-lab", width="stretch")
    st.caption("Python · vLLM · Modal\n\nSQLite · Streamlit · Plotly")


@st.fragment(run_every="5s")
def render():
    runs, requests = read_snapshot()
    if source != "All runs":
        runs = [r for r in runs if r["plan"]["provider"] == source.lower()]
    st.html('<div class="console-status"><span><b>LAB</b> / inference</span>'
            '<span>READ ONLY</span><span>REFRESH / 5s</span>'
            f'<span>RUNS / {len(runs):02d}</span></div>')
    st.title("Inference benchmark console")
    st.html('<p class="subtitle">Latency, throughput, quality, cost. Every number backed by a run.</p>')
    if not runs:
        st.info("No experiments in this view yet. Create a plan in the CLI or select another evidence source.")
        st.code("uv run python -m inference_lab.demo", language="bash")
        return
    by_id = {r["id"]: r for r in runs}
    chosen = st.selectbox("Inspect an experiment", list(by_id),
                          index=next((i for i, r in enumerate(runs) if r["status"] == "complete"), 0),
                          format_func=lambda rid: label(by_id[rid]))
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
                'Fixture timings validate the harness only; model performance and GPU capacity remain unverified.</div>')
    else:
        st.caption("Client measurements from a remote run. Billing and unexposed device metrics remain unverified.")

    cols = st.columns(4)
    cols[0].metric("COMPLETED / OFFERED", f"{summary['successful']} / {summary['offered']}",
                   help="Successful steady-state requests / offered requests, including client rejections.")
    cols[1].metric("LATENCY / P95", ms(summary["latency_p95_s"]), help="Successful requests only.")
    cols[2].metric("THROUGHPUT / REQ·S⁻¹", number(summary["successful_rps"], " req/s"))
    cols[3].metric("FIRST CONTENT / P95", ms(summary["ttft_p95_s"]), help="First nonempty content chunk, not server TTFT.")
    st.write("")
    overview, comparison, reliability, quality, models, campaigns, evidence = st.tabs(
        ["01 Performance", "02 Compare", "03 Reliability", "04 Quality",
         "05 Models", "06 Campaigns", "07 Evidence"])

    with overview:
        left, right = st.columns([1.65, 1])
        successes = [r for r in rows if r["phase"] == "steady" and r["status"] == "success"]
        with left:
            st.subheader("Response latency")
            st.caption(f"{len(successes)} successful samples · milliseconds · client end-to-end")
            if successes:
                latencies = [r["latency_s"] * 1000 for r in successes]
                fig = go.Figure(go.Histogram(x=latencies, nbinsx=14, marker_color="#7c3aed", opacity=.75))
                fig.add_vline(x=summary["latency_p95_s"] * 1000, line_dash="dot", line_color="#d97706",
                              annotation_text="p95", annotation_font_color="#a65e08")
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
        st.subheader("Measurement confidence")
        a, b, c = st.columns(3)
        a.metric("LATENCY-QUALIFIED GOODPUT", number(summary["goodput_rps"], " req/s"))
        b.metric("CLIENT SCHEDULE LAG · P95", ms(summary["schedule_lag_p95_s"]))
        bounds = summary.get("error_rate_wilson_95")
        c.metric("ERROR RATE · 95% UPPER BOUND", f"{bounds[1]:.1%}" if bounds else "Unavailable")
        ci = summary.get("latency_p95_bootstrap_95_s")
        if ci:
            st.caption(f"p95 latency bootstrap interval: {ms(ci[0])}–{ms(ci[1])}. "
                       "Conditional on observed requests; repeated runs assess temporal variation.")
        with st.expander("Capacity evidence checks"):
            if summary["capacity_exclusions"]:
                for reason in summary["capacity_exclusions"]:
                    st.markdown(f"- {reason}")
            else:
                st.success("This run meets the evidence checks. Use at least three repetitions for a campaign operating point.")
        st.subheader("Run configuration")
        a, b, c = st.columns(3)
        with a:
            panel("Hypothesis", p["hypothesis"])
        with b:
            panel("Load envelope", f"{p['rate']:g} requests/sec · {p['concurrency']} concurrent · "
                  f"{p['max_tokens']} maximum output tokens. {p.get('arrival', 'regular').title()} arrivals with bounded admission.")
        with c:
            outcome = "Within configured targets" if summary["within_targets"] else "Targets not demonstrated"
            panel(outcome, f"p95 target ≤ {p['target_p95_s']:g}s; error/rejection target ≤ "
                  f"{p['target_error_rate']:.0%}. Small samples do not establish production SLOs.")
        if successes:
            st.subheader("Response timeline")
            base = min(r["started_at"] for r in successes)
            fig = go.Figure()
            for field, name, color in [("latency_s", "End-to-end", "#2563eb"),
                                       ("ttft_s", "First content", "#7c3aed")]:
                valid = [r for r in successes if r.get(field) is not None]
                fig.add_trace(go.Scatter(x=[r["started_at"] - base for r in valid],
                                        y=[r[field] * 1000 for r in valid], mode="lines+markers",
                                        name=name, line=dict(color=color, width=2), marker_size=6))
            fig.update_layout(xaxis_title="Seconds since first successful request started", yaxis_title="Latency (ms)")
            chart(fig, 270)

    with comparison:
        st.subheader("Experiment comparison")
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
                                      marker_color=["#94a3b8", "#0d9488"]))
                fig.update_layout(yaxis_title="Successful requests / second")
                chart(fig)
            with b:
                st.subheader("Tail latency")
                fig = go.Figure(go.Bar(x=["Baseline", "Selected"],
                                      y=[s["latency_p95_s"] * 1000 if s["latency_p95_s"] is not None else None
                                         for s in [baseline_summary, summary]], marker_color=["#94a3b8", "#7c3aed"]))
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
        for provider, color in [("simulated", "#d97706"), ("modal", "#0d9488")]:
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
                                      marker_color="#e11d48"))
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
        c.metric("IMPORTED BILLING STATEMENT", number(summary["provider_reported_cost_usd"], " USD"))
        st.caption("Billing provenance: " + summary["billing_verification"].replace("_", " "))
        st.caption("Simulation costs $0 in cloud resources. Remote estimates exclude uncertain charges and are not billing caps.")
        with st.expander("Resource limits & spending assumptions"):
            st.json({"maximum_authorized_usd": p["max_spend_usd"], "runtime_seconds": p["runtime_s"], **estimate})
        st.subheader("Server telemetry")
        telemetry_rows = [{"at": datetime.fromtimestamp(t["at"], timezone.utc), **t["values"]}
                          for t in run["telemetry"] if t["values"]]
        if telemetry_rows:
            st.line_chart(pd.DataFrame(telemetry_rows).set_index("at"))
        devices = [{"at": datetime.fromtimestamp(t["at"], timezone.utc),
                    "Memory used (GiB)": t["gpu_memory_bytes"] / 2**30,
                    "GPU utilization (%)": t["gpu_utilization"]}
                   for t in run["telemetry"] if t.get("gpu_memory_bytes") is not None]
        if devices:
            device_frame = pd.DataFrame(devices).set_index("at")
            st.caption("NVIDIA device observations: aggregate memory and mean device utilization; distinct units.")
            st.line_chart(device_frame[["Memory used (GiB)"]])
            st.line_chart(device_frame[["GPU utilization (%)"]])
        if not devices and not telemetry_rows:
            panel("Device telemetry unavailable", "No server queue or KV-cache samples were exposed in this run. "
                  "GPU utilization and allocated memory are not inferred from client timing.")

    with quality:
        st.subheader("Quality evaluation")
        st.caption("Versioned task scoring accompanies performance. Synthetic fixture outputs are not model-quality evidence.")
        a, b, c = st.columns(3)
        a.metric("UNIQUE COMPLETED CASES", str(summary["quality_unique_cases"]))
        coverage = summary.get("quality_coverage")
        b.metric("EVALUATION COVERAGE", f"{coverage:.0%}" if coverage is not None else "Not evaluated")
        c.metric("TOKEN LENGTHS", "Conformant" if summary["length_conformant"] is True else
                 "Mismatched" if summary["length_conformant"] is False else "Not controlled")
        if summary["quality_categories"]:
            st.dataframe(pd.DataFrame([{"Category": name, **values}
                for name, values in summary["quality_categories"].items()]), hide_index=True, width="stretch")
        else:
            panel("Run a dedicated quality workload", "The built-in suite covers arithmetic, structured extraction, "
                  "code reasoning and grounded reading across 40 distinct cases. Custom JSONL suites are supported. "
                  "This is regression coverage, not a general intelligence ranking.")
        st.subheader("Upstream benchmark cross-check")
        reference = run.get("reference", {})
        if reference.get("metrics"):
            check = crosscheck(summary, reference["metrics"])
            st.dataframe(pd.DataFrame([{"Metric": name, **values} for name, values in check["metrics"].items()]),
                         hide_index=True, width="stretch")
            for limitation in check["limitations"]:
                st.caption(limitation)
        else:
            panel("No upstream result attached", "Enable reference_s in a reviewed single-length, Poisson token plan "
                  "to run vLLM's benchmark from a local Docker client within the same approved resource lifetime. "
                  "Local unit tests alone do not establish measurement accuracy.")
        server = summary["server_measurements"]
        if server["histograms"]:
            st.subheader("Server request-phase measurements")
            st.dataframe(pd.DataFrame([{"Metric": name, **values} for name, values in server["histograms"].items()]),
                         hide_index=True, width="stretch")
            st.caption(server.get("scope", ""))

    with models:
        st.subheader("Model registry")
        st.caption("Qwen2.5 0.5B / 3B / 7B / 32B. All below 100B; this is not a claim to support every model under 100B.")
        inventory = []
        for key, profile in PROFILES.items():
            matching = [r for r in runs if r["plan"]["model"] == profile.model]
            tokenizer_checked = any(r.get("manifest", {}).get("version") == "engineering-passages-v1"
                                    for r in matching)
            remote = [r for r in matching if r["plan"]["provider"] == "modal" and r["status"] == "complete"
                      and r["shutdown"] == "verified" and r.get("server_identity", {}).get("verified")
                      and summarize(r, requests[r["id"]])["successful"] > 0]
            inventory.append({"Profile": key, "Parameters (B)": profile.parameters_b,
                              "Hardware": f"{profile.gpu_count} × {profile.gpu}",
                              "Pinned tokenizer prepared": tokenizer_checked,
                              "Verified remote smoke runs": len(remote),
                              "State": "Remote smoke passed" if remote else "GPU validation pending"})
        st.dataframe(pd.DataFrame(inventory), hide_index=True, width="stretch")
        st.caption("Readiness counts reflect the current evidence-source filter. Simulation never validates model execution.")
        with st.expander("Analytical memory budgets and model revisions"):
            st.json([profile.to_dict() for profile in PROFILES.values()])
        panel("What larger-model support means", "Profiles specify pinned weights, FP16 serving, context limits, "
              "tensor parallelism and resource budgets. The 32B profile uses two 80GB A100s. "
              "Weights, activations and KV cache must fit together; parameter count alone is not a compatibility test.")

    with campaigns:
        st.subheader("Repeated load sweeps")
        st.caption("Sequential experiments, aggregate spending approval, durable progress and explicit capacity checks.")
        campaign_files = sorted(Path(os.getenv("LAB_DATA") or "data").glob("campaign-*-report.json"), reverse=True)
        if campaign_files:
            file = st.selectbox("Campaign report", campaign_files, format_func=lambda path: path.stem)
            try:
                report = json.loads(file.read_text())
                st.dataframe(pd.DataFrame(report["points"]), hide_index=True, width="stretch")
                st.json({"highest_tested_qualified_offered_rps": report["highest_tested_qualified_offered_rps"],
                         "status": report["campaign"]["status"],
                         "approved_maximum_usd": report["campaign"]["max_spend_usd"]})
                for limitation in report["limitations"]:
                    st.caption(limitation)
            except (ValueError, KeyError, OSError):
                st.warning("Campaign report is not readable yet; retry after the next export.")
        else:
            panel("No campaign results yet", "Use sweep-plan, sweep-approve and sweep-run in the CLI. "
                  "A capacity point needs three qualifying remote repetitions; synthetic runs remain demonstrations.")

    with evidence:
        st.subheader("Request journal")
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
    st.html('<div class="footer">inference.lab / local journal<br>'
            'read-only · refresh 5s · spending approval via CLI</div>')


render()
