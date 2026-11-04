QUALITY = [
    {"id": "q1", "prompt": "What is 17 plus 25? Reply with only the number.", "expected": "42"},
    {"id": "q2", "prompt": "Sort these numbers ascending: 9, 2, 5. Reply only with comma-separated numbers.",
     "expected": "2,5,9"},
    {"id": "q3", "prompt": "A service returns HTTP 429. Is it rate limiting or a successful response? "
                            "Reply with only 'rate limiting' or 'success'.", "expected": "rate limiting"},
    {"id": "q4", "prompt": "Given x = [1, 2, 3], what does len(x) return in Python? Reply only with the number.",
     "expected": "3"},
]


def workload(kind, index):
    if kind == "quality":
        return QUALITY[index % len(QUALITY)]["prompt"]
    if kind == "chat":
        return ["Explain continuous batching in three sentences.",
                "Why does an inference server keep a KV cache?",
                "Explain the difference between latency and throughput."][index % 3] + f" Example {index}."
    passage = ("The service accepts requests through a bounded queue. Workers batch compatible requests. "
               "The client records latency and retries only when authorized. A memory budget reserves room "
               "for weights, activations and KV cache. Overload should produce observable rejection. ")
    return "Summarize the engineering decisions in five bullets:\n" + passage * 30 + f"\nDocument {index}."


def quality_score(prompt, output):
    item = next((x for x in QUALITY if x["prompt"] == prompt), None)
    if item is None:
        return None
    return output.strip().lower().replace(" ", "") == item["expected"].replace(" ", "")
