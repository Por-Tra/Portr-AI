"""
████████████   ██ ██████              ██                            ████      ██
██        ██ ████     ████            ███                          ██ ███     ██
██        ██████        ███ ██████████████████ ████████           ███  ███    ██
███████████████         ███ ██        ███      ██                ██     ███   ██
██          ███         ███ ██        ███      ██     █████████ ████████████  ██
██           █████   █████  ██        ████     ██              ███        ███ ██
██             █████████    ██         ███████ ██             ███          ██ ██

auteur : ChatGPT(LLM)

File name : audit_hf_sft.py


Read-only Hugging Face SFT survey using public viewer samples.

No model training or active dataset changes. Small, reproducible samples and
normalized candidate pairs are cached under artifacts/sft/hf_survey/20261117.
The measured fit rate is descriptive, not a factual-quality certification or an
unbiased estimate of the entire dataset.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokenizer.tokenizer import Tokenizer

OUT = ROOT / "artifacts/sft/hf_survey/20261117"
REPOS = ["sammybow/luth-sft", "legmlai/openhermes-fr", "CATIE-AQ/CFP"]


def get(url):
    cache = OUT / "raw" / (hashlib.sha256(url.encode()).hexdigest() + ".json")
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    request = urllib.request.Request(url, headers={"User-Agent": "Portr-AI-dataset-survey/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 502, 503, 504) or attempt == 2:
                raise
            time.sleep(.8 * (attempt + 1))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"url": url, "data": data}, ensure_ascii=False), encoding="utf-8")
    return {"url": url, "data": data}


def endpoint(name, **params):
    return "https://datasets-server.huggingface.co/" + name + "?" + urllib.parse.urlencode(params)


def pair(row):
    """Extract an independent first user/assistant turn, preserving its system text."""
    if any(row.get(field) is True for field in ("bad_entry", "bad_prompt_detected", "bad_response_detected")):
        return None, "upstream_bad_flag"
    if "prompt" in row and "accepted_completion" in row:
        prompt, answer = row["prompt"], row["accepted_completion"]
    elif "inputs" in row and "targets" in row:
        prompt, answer = row["inputs"], row["targets"]
    elif isinstance(row.get("messages"), list):
        messages = row["messages"]
        prompt = None
        system = []
        for message in messages:
            if message.get("role") == "system" and prompt is None:
                system.append(message.get("content", ""))
            elif message.get("role") == "user" and prompt is None:
                prompt = message.get("content")
            elif message.get("role") == "assistant" and prompt is not None:
                answer = message.get("content")
                prompt = "\n\n".join([*system, prompt])
                break
            else:
                return None, "unsupported_conversation"
        else:
            return None, "missing_pair"
    else:
        return None, "unsupported_schema"
    if not isinstance(prompt, str) or not isinstance(answer, str) or not prompt.strip() or not answer.strip():
        return None, "empty_pair"
    prompt, answer = prompt.strip(), answer.strip()
    if " ".join(prompt.split()).casefold() == " ".join(answer.split()).casefold():
        return None, "identical_prompt_answer"
    return (prompt, answer), None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.load(ROOT / "tokenizer/token/bpe_model.json")
    tasks, cards = [], {}
    for repo in REPOS:
        metadata = get("https://huggingface.co/api/datasets/" + repo)["data"]
        cards[repo] = {"sha": metadata.get("sha"), "license": metadata.get("cardData", {}).get("license"),
                       "url": "https://huggingface.co/datasets/" + repo}
        splits = get(endpoint("splits", dataset=repo))["data"]["splits"]
        for split in splits:
            initial = get(endpoint("rows", dataset=repo, config=split["config"], split=split["split"], offset=0, length=1))["data"]
            total = initial["num_rows_total"]
            rng = random.Random(repo + split["split"] + "20261117")
            blocks = 5 if repo == "sammybow/luth-sft" else 10
            for block in range(blocks):
                low, high = int(total*block/blocks), max(int(total*(block+1)/blocks)-40, int(total*block/blocks))
                offset = min(rng.randint(low,high), max(0,total-40))
                tasks.append({"repo": repo, "config":split["config"],"split":split["split"],"total":total,"offset":offset})
    def fetch(task):
        try:
            data = get(endpoint("rows",dataset=task["repo"],config=task["config"],split=task["split"],offset=task["offset"],length=40))["data"]
            return task, data, None
        except (urllib.error.URLError, TimeoutError) as exc:
            return task, None, str(exc)
    summaries, examples, errors = {}, {}, []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for task,data,error in executor.map(fetch,tasks):
            key = task["repo"] + "/" + task["split"]
            if error:
                errors.append({**task,"error":error})
                continue
            summary = summaries.setdefault(key,{"repo":task["repo"],"split":task["split"],"total_rows":task["total"],
                "sampled":0,"valid_pairs":0,"fits_context":0,"fits_current_limits":0,"candidate_count":0,
                "rejections":Counter(),"prompt_tokens":[],"answer_tokens":[],"total_tokens":[],"sources":Counter(),"multiturn":0})
            candidates = examples.setdefault(key,[])
            for entry in data["rows"]:
                summary["sampled"] += 1
                if entry.get("truncated_cells"):
                    summary["rejections"]["truncated_viewer_row"]+=1
                    continue
                row = entry["row"]
                summary["sources"][str(row.get("dataset",row.get("task","unknown")))] += 1
                messages = row.get("messages",[])
                summary["multiturn"] += int(sum(m.get("role")=="assistant" for m in messages)>1)
                result,reason = pair(row)
                if reason:
                    summary["rejections"][reason] += 1
                    continue
                prompt,answer=result
                p,a=tok.encode(prompt),tok.encode(answer)
                total=len(p)+len(a)+3
                summary["valid_pairs"]+=1
                summary["prompt_tokens"].append(len(p));summary["answer_tokens"].append(len(a));summary["total_tokens"].append(total)
                summary["fits_context"]+=int(total<=256)
                summary["fits_current_limits"]+=int(total<=256 and 1<=len(a)<=160)
                if total>256:
                    summary["rejections"]["context_over_256"]+=1
                    continue
                if len(a)>160:
                    summary["rejections"]["answer_over_160"]+=1
                    continue
                if "```" in prompt or "```" in answer:
                    summary["rejections"]["fenced_code"]+=1
                    continue
                if "<think>" in answer or "<tool_call>" in answer:
                    summary["rejections"]["reasoning_or_tool_markup"]+=1
                    continue
                summary["candidate_count"]+=1
                candidates.append({"prompt":prompt,"answer":answer,"source":task["repo"],"source_split":task["split"],
                    "source_component":row.get("dataset"),"task":row.get("task"),"subject":row.get("subject"),
                    "source_revision":cards[task["repo"]]["sha"],"source_row":entry["row_idx"],
                    "source_license":cards[task["repo"]]["license"],
                    "group":"prompt:"+hashlib.sha256(json.dumps(p).encode()).hexdigest()})
    for key,s in summaries.items():
        for field in ("prompt_tokens","answer_tokens","total_tokens"):
            lengths=s.pop(field)
            s[field+"_quantiles"] = np.quantile(lengths,[0,.25,.5,.75,.95,1]).tolist() if lengths else []
        s["candidate_rate"]=s["candidate_count"]/s["sampled"]
        print(key, "rows=",s["total_rows"],"sample=",s["sampled"],"candidates=",s["candidate_count"],
              "rate=",round(s["candidate_rate"],3),flush=True)
        filename = key.replace("/","__")+".jsonl"
        (OUT/filename).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in examples[key]),encoding="utf-8")
    output={"date":"2026-11-17","tokenizer_fingerprint":tok.fingerprint(),"context_tokens":256,"max_answer_tokens":160,
        "cards":cards,"subsets":summaries,"errors":errors,
        "limitations":["Stratified windows, not an unbiased independent random sample.",
        "First-turn extraction only; fit rate is not a factual correctness assessment.",
        "No models trained and no active datasets changed."]}
    (OUT/"survey.json").write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"review_samples.json").write_text(json.dumps({key:random.Random(key).sample(rows,min(5,len(rows)))
        for key,rows in examples.items()},ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
