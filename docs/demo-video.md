# Inference Lab demo video

[Watch on YouTube](https://youtu.be/eu_YeSP6nyQ) ·
[Download the MP4](video/inference-lab-demo.mp4).

A 73-second captioned walkthrough recorded from the actual local Streamlit app.
The footage is trimmed and reordered for clarity; it is not an animation or a live
GPU execution recording. There is no narration or music.

| Time | Chapter |
|---|---|
| 0:00 | Local experiment workspace and overview |
| 0:10 | Model selection and JSON configuration |
| 0:24 | Serving and load controls |
| 0:38 | Preflight estimates and configuration downloads |
| 0:50 | Previously completed GPU run results |
| 1:03 | Model readiness and next steps |

The recorded result is a Qwen2.5-0.5B-Instruct A100-40GB smoke run: 60/60 requests,
752.5 ms client p95 latency. It does not establish sustained capacity or general
model quality. The preflight configuration shown is a separate L4 plan; its $0.71
upper estimate is not the A100 run's actual bill. See the
[validation report](engineering-report.md) for evidence scope and limitations.

Built by Venkata Ganji with OpenAI Codex assistance.
Source: https://github.com/offroadguy/inference-lab
Portfolio: https://offroadguy.github.io/
LinkedIn: https://www.linkedin.com/in/venkata-sai-ganji-3a30a2a4/
