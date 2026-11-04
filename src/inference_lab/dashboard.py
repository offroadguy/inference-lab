"""Read-only dashboard: spending approval stays in the CLI."""
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from inference_lab.store import read_snapshot
from inference_lab.report import summarize

st.set_page_config(page_title="Inference Lab", page_icon="🔬", layout="wide")
st.title("Inference Lab")
st.caption("Local experiment notebook · approval in CLI · client measurements and server telemetry kept separate")


@st.fragment(run_every="3s")
def render():
    runs, requests = read_snapshot()
    if not runs:
        st.info("Create a plan and run it with the CLI. See README for a free simulated demo.")
        return
    chosen = st.selectbox("Run", [r["id"] for r in runs])
    run = next(r for r in runs if r["id"] == chosen)
    rows = requests[chosen]
    summary = summarize(run, rows)
    if run["plan"]["provider"] == "simulated":
        st.warning("SIMULATED DATA — verifies software behavior, not GPU/model performance")
    st.write(f"**{run['status']}** · shutdown: **{run['shutdown']}** · stages: {', '.join(run['stages']) or 'planned'}")
    cols = st.columns(4)
    for col, name in zip(cols, ["successful", "latency_p95_s", "successful_rps", "output_tokens_s"]):
        value = summary[name]
        col.metric(name.replace("_", " "), "unavailable" if value is None else f"{value:.3f}")
    steady = [r for r in rows if r["phase"] == "steady"]
    successes = [r for r in steady if r["status"] == "success"]
    if successes:
        frame = pd.DataFrame(successes)
        st.plotly_chart(px.histogram(frame, x="latency_s", nbins=20, title="Client end-to-end latency"))
        st.plotly_chart(px.scatter(frame, x="started_at", y="ttft_s", title="First content chunk latency"))
    if steady:
        st.plotly_chart(px.histogram(pd.DataFrame(steady), x="status", color="error", title="Request outcomes"))
    comparisons = [{"id": r["id"], "provider": r["plan"]["provider"], "rate": r["plan"]["rate"],
                    **summarize(r, requests[r["id"]])} for r in runs]
    st.plotly_chart(px.scatter(pd.DataFrame(comparisons), x="rate", y="successful_rps", color="provider",
                              hover_name="id", title="Achieved throughput versus offered load (check configuration)"))
    telemetry_rows = [{"at": t["at"], **t["values"]} for t in run["telemetry"] if t["values"]]
    if telemetry_rows:
        st.line_chart(pd.DataFrame(telemetry_rows).set_index("at"))
    st.caption("GPU allocated-memory/utilization trends unavailable. KV-cache occupancy is shown only when exposed.")
    st.dataframe(pd.DataFrame(steady), column_config={"output": None})
    with st.expander("Configuration, cost, methodology and limitations"):
        st.json({"plan": run["plan"], "summary": summary})
    st.download_button("Download results JSON", json.dumps({"plan": run["plan"], "summary": summary,
                        "requests": rows}, indent=2), file_name=f"{chosen}.json")
    if steady:
        st.download_button("Download requests CSV", pd.DataFrame(steady).to_csv(index=False),
                           file_name=f"{chosen}.csv")


render()
