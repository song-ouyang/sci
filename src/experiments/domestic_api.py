"""DeepSeek/Qwen research adapter, Python 3.9+, standard library only."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from datetime import datetime, timezone
from urllib import error, parse, request
import uuid


PROVIDERS = {
    "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "deepseek-v4-pro",
                 "DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    "qwen": ("DASHSCOPE_API_KEY", "QWEN_MODEL", "qwen3.8-max",
             "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_env(path):
    """Literal KEY=value only; never execute shell code or expand substitutions."""
    if not path.exists():
        return
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError("Invalid environment file line %d" % number)
        if value.startswith(("'", '"')):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError("Unclosed quote at environment line %d" % number)
            value = value[1:-1]
        if not os.environ.get(key):
            os.environ[key] = value


def provider_config(name):
    key_env, model_env, model_default, url_env, url_default = PROVIDERS[name]
    base_url = os.environ.get(url_env, url_default).rstrip("/")
    url = parse.urlsplit(base_url)
    host = url.hostname or ""
    valid_host = (host == "api.deepseek.com" if name == "deepseek" else
                  host in {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com",
                           "dashscope-us.aliyuncs.com"} or host.endswith(".maas.aliyuncs.com"))
    if (url.scheme != "https" or not valid_host or url.username or url.password
            or url.query or url.fragment or url.port not in (None, 443)):
        raise ValueError("%s must be an official HTTPS provider endpoint" % url_env)
    model = os.environ.get(model_env, model_default).strip()
    if not model:
        raise ValueError("%s cannot be empty" % model_env)
    return {"provider": name, "model": model, "base_url": base_url,
            "key_env": key_env, "key_configured": bool(os.environ.get(key_env, "").strip())}


def validate_batch(batch):
    if not isinstance(batch, dict) or set(batch) != {"messages", "expected_ids", "allowed_labels"}:
        raise ValueError("Batch requires exactly messages, expected_ids, allowed_labels (no answer key)")
    messages = batch["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    for i, message in enumerate(messages):
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or message["role"] not in {"system", "user"}
                or not isinstance(message["content"], str) or not message["content"].strip()
                or (message["role"] == "system" and i != 0)):
            raise ValueError("Only an optional initial system message and fresh user input are allowed")
    if messages[-1]["role"] != "user":
        raise ValueError("Last message must be user input")
    for key in ("expected_ids", "allowed_labels"):
        values = batch[key]
        if (not isinstance(values, list) or not values
                or any(not isinstance(v, str) or not v for v in values)
                or len(set(values)) != len(values)):
            raise ValueError("%s must contain unique nonempty strings" % key)


def build_payload(config, batch, thinking=True, max_tokens=16384):
    validate_batch(batch)
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    payload = {"model": config["model"], "messages": batch["messages"],
               "stream": True, "stream_options": {"include_usage": True},
               "max_tokens": max_tokens}
    if config["provider"] == "deepseek":
        payload["thinking"] = {"type": "enabled" if thinking else "disabled"}
        if thinking:
            payload["reasoning_effort"] = "high"
    else:
        payload["enable_thinking"] = thinking
    return payload


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("Nonstandard JSON constant")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)


def validate_predictions(content, batch):
    # Match revision-v3's allowance for one enclosing JSON code fence.
    content = content.strip()
    if content.startswith("```") and content.endswith("```"):
        lines = content.splitlines()
        if len(lines) < 3 or lines[0].strip() not in {"```", "```json"} or lines[-1].strip() != "```":
            raise ValueError("Malformed JSON code fence")
        content = "\n".join(lines[1:-1]).strip()
    predictions = strict_json(content)
    if not isinstance(predictions, list) or len(predictions) != len(batch["expected_ids"]):
        raise ValueError("Prediction count does not match complete batch")
    for prediction, expected_id in zip(predictions, batch["expected_ids"]):
        if not isinstance(prediction, dict) or set(prediction) != {"id", "predicted"}:
            raise ValueError("Each prediction must contain exactly id and predicted")
        if prediction["id"] != expected_id:
            raise ValueError("Prediction ID/order mismatch")
        if not isinstance(prediction["predicted"], str) or prediction["predicted"] not in batch["allowed_labels"]:
            raise ValueError("Invalid prediction label")
    return predictions


def parse_stream(raw):
    """Accept both providers' SSE usage shapes, require terminal marker and stop."""
    content, reasoning, done, finish = [], [], False, None
    observed_model, response_id, usage = None, None, None
    events = re.split(r"\r?\n\r?\n", raw.decode("utf-8"))
    for event in events:
        data = "\n".join(line[5:].lstrip(" ") for line in event.splitlines()
                         if line.startswith("data:"))
        if not data:
            continue
        if done:
            raise ValueError("Unexpected event after DONE")
        if data.strip() == "[DONE]":
            done = True
            continue
        chunk = strict_json(data)
        if not isinstance(chunk, dict) or chunk.get("error"):
            raise ValueError("Provider returned an invalid/error stream event")
        if chunk.get("model"):
            if observed_model and observed_model != chunk["model"]:
                raise ValueError("Model changed within stream")
            observed_model = chunk["model"]
        response_id = chunk.get("id") or response_id
        usage = chunk.get("usage") or usage
        choices = chunk.get("choices", [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise ValueError("Expected one completion")
        for choice in choices:
            if choice.get("index", 0) != 0:
                raise ValueError("Unexpected completion index")
            delta = choice.get("delta", {})
            if finish is not None and any(delta.get(k) for k in ("content", "reasoning_content")):
                raise ValueError("Content after completion finish")
            if delta.get("tool_calls") or delta.get("function_call"):
                raise ValueError("Tool calls are forbidden")
            for key, target in (("content", content), ("reasoning_content", reasoning)):
                value = delta.get(key)
                if value is not None:
                    if not isinstance(value, str):
                        raise ValueError("Unexpected nontext response")
                    target.append(value)
            if choice.get("finish_reason") is not None:
                if finish is not None:
                    raise ValueError("Multiple finish events")
                finish = choice["finish_reason"]
    if not done or finish != "stop":
        raise ValueError("Incomplete response, truncation, or non-stop finish")
    return {"content": "".join(content), "reasoning_content": "".join(reasoning),
            "model": observed_model, "id": response_id, "usage": usage, "finish_reason": finish}


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def run_batch(config, batch, output_root, thinking=True, max_tokens=16384,
              timeout=120, opener=None):
    payload = build_payload(config, batch, thinking, max_tokens)
    key = os.environ.get(config["key_env"], "").strip()
    if not key:
        raise ValueError("Missing %s; fill local .env first" % config["key_env"])
    folder = output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                            + "-" + config["provider"] + "-" + uuid.uuid4().hex[:12])
    folder.mkdir(parents=True, mode=0o700)
    write_json(folder / "batch.json", batch)
    write_json(folder / "request.json", payload)
    metadata = {**config, "started_at": utc_now(), "status": "started",
                "timeout_seconds": timeout, "automatic_retries": 0,
                "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(folder / "metadata.json", metadata)
    started = time.monotonic()
    opener = opener or request.build_opener(NoRedirect())
    req = request.Request(config["base_url"] + "/chat/completions",
                          data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                          headers={"Authorization": "Bearer " + key,
                                   "Content-Type": "application/json", "Accept": "text/event-stream"})
    try:
        with opener.open(req, timeout=timeout) as response:
            metadata["http_status"] = response.status
            metadata["request_id"] = response.headers.get("x-request-id")
            with (folder / "response.sse").open("wb") as archive:
                for line in response:
                    archive.write(line)
                    archive.flush()
        metadata["status"] = "invalid_response"
        parsed = parse_stream((folder / "response.sse").read_bytes())
        write_json(folder / "response.json", parsed)
        predictions = validate_predictions(parsed["content"], batch)
        write_json(folder / "predictions.json", predictions)
        metadata.update(status="valid", observed_model=parsed["model"], usage=parsed["usage"])
    except error.HTTPError as exc:
        metadata.update(status="http_error", http_status=exc.code,
                        error="HTTP %s; see http_error.txt" % exc.code)
        # Provider errors can echo credentials; redact those before persisting.
        (folder / "http_error.txt").write_text(
            exc.read().decode("utf-8", errors="replace").replace(key, "[REDACTED]"), encoding="utf-8")
    except (OSError, error.URLError) as exc:
        metadata.update(status="transport_error", error=type(exc).__name__)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        metadata.update(status="invalid_response", error=str(exc).replace(key, "[REDACTED]"))
    finally:
        metadata.update(finished_at=utc_now(), elapsed_seconds=round(time.monotonic() - started, 3))
        write_json(folder / "metadata.json", metadata)
        write_json(folder / "sha256.json", {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.iterdir()) if path.is_file() and path.name != "sha256.json"})
    return folder, metadata


def smoke_batch():
    return {"messages": [{"role": "user", "content":
                          'Return exactly this JSON array, with no other text: '
                          '[{"id":"smoke-1","predicted":"OK"}]'}],
            "expected_ids": ["smoke-1"], "allowed_labels": ["OK"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "smoke", "run"))
    parser.add_argument("--provider", choices=("deepseek", "qwen", "all"), default="all")
    parser.add_argument("--env-file", type=Path, default=Path(__file__).resolve().parents[2] / ".env")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "results/domestic_api_checks")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--thinking", choices=("on", "off"), default="on")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args(argv)
    load_env(args.env_file)
    configs = [provider_config(name) for name in
               (PROVIDERS if args.provider == "all" else [args.provider])]
    if args.command == "check":
        print(json.dumps(configs, ensure_ascii=False, indent=2))
        return 0 if all(c["key_configured"] for c in configs) else 1
    if args.timeout < 1:
        raise ValueError("timeout must be positive")
    if args.command == "run" and not args.input:
        parser.error("run requires --input path/to/batch.json")
    batch = (strict_json(args.input.read_text(encoding="utf-8"))
             if args.command == "run" else smoke_batch())
    thinking = args.thinking == "on" if args.command == "run" else False
    max_tokens = args.max_tokens if args.command == "run" else 128
    # Validate all providers before any network request.
    prepared = [(c, build_payload(c, batch, thinking, max_tokens)) for c in configs]
    if args.dry_run:
        print(json.dumps([{**c, "request": payload} for c, payload in prepared],
                         ensure_ascii=False, indent=2))
        return 0
    missing = [c["key_env"] for c in configs if not c["key_configured"]]
    if missing:
        raise ValueError("Missing " + ", ".join(missing) + "; fill local .env first")
    failed = False
    for config in configs:
        folder, metadata = run_batch(config, batch, args.output / args.command,
                                     thinking, max_tokens, args.timeout)
        print(json.dumps({"provider": config["provider"], "status": metadata["status"],
                          "output": str(folder.resolve())}, ensure_ascii=False), flush=True)
        failed |= metadata["status"] != "valid"
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as exc:
        print("Error: " + str(exc), file=sys.stderr)
        sys.exit(2)
