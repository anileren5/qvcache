#!/usr/bin/env python3
"""Build TriviaQA simZipf+ streams.

simZipf draws each request from a frozen pool of 51 paraphrases, so popular
questions repeat the same vector. simZipf+ keeps the Zipf draw over questions
but emits a different paraphrase every time a question is chosen. Released
paraphrases are used first. Further visits are new Mistral-7B rephrases.
Vectors are embedded with the encoder that reproduces Aker's released queries.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from generate_simzipf_plus import choose_topics  # noqa: E402

GROUP_LINES = 51
SKEWS = (0.3, 0.6, 0.99)


def read_question_lines(path: str) -> list[str]:
    # A few questions contain a bare CR. Keep them on one line; split only on LF.
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"")
    lines = raw.decode("utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def load_groups(path: str) -> list[list[str]]:
    lines = read_question_lines(path)
    if len(lines) % GROUP_LINES != 0:
        raise SystemExit(f"{path} has {len(lines)} lines, not a multiple of {GROUP_LINES}")
    groups: list[list[str]] = []
    for start in range(0, len(lines), GROUP_LINES):
        seen: set[str] = set()
        unique: list[str] = []
        for line in lines[start : start + GROUP_LINES]:
            text = line.strip()
            if text and text not in seen:
                seen.add(text)
                unique.append(text)
        if not unique:
            raise SystemExit(f"empty question group at line {start + 1}")
        groups.append(unique)
    return groups


def plan_streams(
    groups: list[list[str]],
    n_stream: int,
    seed: int,
    skews: tuple[float, ...],
    distribution: str = "zipfian",
) -> dict:
    """Zipf or uniform over questions. Each visit takes the next unused paraphrase slot."""
    if distribution == "uniform":
        specs = [("uniform", "uniform")]
    elif distribution == "zipfian":
        specs = [(str(skew), "zipfian") for skew in skews]
    else:
        raise SystemExit(f"unknown distribution {distribution}")
    streams = {}
    for label, dist in specs:
        rng = np.random.default_rng(seed)
        topics = choose_topics(len(groups), n_stream, rng, dist, float(label) if dist == "zipfian" else 0.0)
        orders = [rng.permutation(len(group)) for group in groups]
        cursor = np.zeros(len(groups), dtype=np.int64)
        keys: list[str] = []
        texts: dict[str, str] = {}
        n_pool = 0
        for topic in topics.tolist():
            slot = int(cursor[topic])
            cursor[topic] = slot + 1
            key = f"{label}:{topic}:{slot}"
            keys.append(key)
            pool = groups[topic]
            if slot < len(pool):
                texts[key] = pool[int(orders[topic][slot])]
                n_pool += 1
        pending = [key for key in keys if key not in texts]
        streams[label] = {
            "topics": topics.astype(np.int32),
            "keys": keys,
            "texts": texts,
            "n_pool": n_pool,
            "n_pending": len(pending),
            "pending": pending,
            "max_count": int(cursor.max()) if len(cursor) else 0,
            "topics_overflowing": int(np.sum(cursor > np.array([len(g) for g in groups]))),
        }
    return streams


def load_raw_groups(path: str) -> list[list[str]]:
    """Frozen 51-line groups, including repeated lines. simZipf samples these with replacement."""
    lines = read_question_lines(path)
    if len(lines) % GROUP_LINES != 0:
        raise SystemExit(f"{path} has {len(lines)} lines, not a multiple of {GROUP_LINES}")
    return [lines[start : start + GROUP_LINES] for start in range(0, len(lines), GROUP_LINES)]


def plan_frozen_uniform(groups: list[list[str]], n_stream: int, seed: int) -> dict:
    """simZipf uniform: i.i.d. group, then a uniform line from that group's frozen pool."""
    rng = np.random.default_rng(seed)
    topics = choose_topics(len(groups), n_stream, rng, "uniform", 0.0)
    members = rng.integers(0, GROUP_LINES, size=n_stream, dtype=np.int64)
    keys: list[str] = []
    texts: dict[str, str] = {}
    for index, (topic, member) in enumerate(zip(topics.tolist(), members.tolist())):
        key = f"frozen:{index}"
        keys.append(key)
        texts[key] = groups[topic][member].strip()
    return {
        "topics": topics.astype(np.int32),
        "keys": keys,
        "texts": texts,
        "n_pool": n_stream,
        "n_pending": 0,
    }


def write_query_bin(vectors: np.ndarray, path: str) -> None:
    arr = np.ascontiguousarray(vectors, dtype=np.float32)
    npts, dim = arr.shape
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("II", int(npts), int(dim)))
        arr.tofile(f)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-8, None)


def probe_encoder(questions: list[str], official_path: str, out_path: str) -> dict:
    import torch
    from transformers import AutoModel, AutoTokenizer

    official = np.load(official_path).astype(np.float32)
    official_norms = np.linalg.norm(official, axis=1)
    official_n = _normalize(official)
    print(
        f"official queries {official.shape} norm mean={official_norms.mean():.4f} "
        f"std={official_norms.std():.4f}",
        flush=True,
    )

    candidates = [
        ("BAAI/bge-base-en-v1.5", "Represent this sentence for searching relevant passages: ", "cls"),
        ("BAAI/bge-base-en-v1.5", "", "cls"),
        ("BAAI/bge-base-en", "Represent this sentence for searching relevant passages: ", "cls"),
        ("BAAI/bge-base-en", "", "cls"),
        ("intfloat/e5-base-v2", "query: ", "mean"),
        ("facebook/contriever", "", "mean"),
    ]
    # A spread of groups, not only the first question, so unpopular groups
    # are not the only ones compared with the released stream.
    sample = questions[: 20 * 51]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    best = None
    for name, prefix, pool in candidates:
        started = time.time()
        print(f"probing {name} prefix={prefix!r} pool={pool}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        model.to(device)
        model.eval()
        raw = _embed_texts(sample, tokenizer, model, device, prefix, pool, batch_size=64)
        raw_n = _normalize(raw)
        max_cos = np.full(raw_n.shape[0], -1.0, dtype=np.float32)
        min_l2_raw = np.full(raw.shape[0], np.inf, dtype=np.float32)
        min_l2_unit = np.full(raw_n.shape[0], np.inf, dtype=np.float32)
        chunk = 8192
        raw_sq = np.sum(raw * raw, axis=1, keepdims=True)
        for start in range(0, official_n.shape[0], chunk):
            block = official[start : start + chunk]
            block_n = official_n[start : start + chunk]
            max_cos = np.maximum(max_cos, (raw_n @ block_n.T).max(axis=1))
            bb = np.sum(block * block, axis=1)
            dist2 = np.maximum(raw_sq + bb - 2.0 * (raw @ block.T), 0.0)
            min_l2_raw = np.minimum(min_l2_raw, np.sqrt(dist2.min(axis=1)))
            # Unit vectors: ||a-b||^2 = 2 - 2 cos.
            cos = raw_n @ block_n.T
            min_l2_unit = np.minimum(min_l2_unit, np.sqrt(np.maximum(2.0 - 2.0 * cos, 0.0).min(axis=1)))
        n_cos = int(np.sum(max_cos > 0.999))
        n_l2_raw = int(np.sum(min_l2_raw < 0.05))
        n_l2_unit = int(np.sum(min_l2_unit < 0.05))
        normalize_output = n_l2_unit > n_l2_raw
        record = {
            "model": name,
            "prefix": prefix,
            "pool": pool,
            "normalize": normalize_output,
            "n_sample": int(len(sample)),
            "n_cosine_gt_0.999": n_cos,
            "n_l2_lt_0.05": int(max(n_l2_raw, n_l2_unit)),
            "max_cosine": float(max_cos.max()),
            "min_l2": float(min(min_l2_raw.min(), min_l2_unit.min())),
            "median_max_cosine": float(np.median(max_cos)),
            "seconds": round(time.time() - started, 1),
        }
        results.append(record)
        print(json.dumps(record), flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        if best is None or (record["n_l2_lt_0.05"], n_cos, record["max_cosine"]) > (
            best["n_l2_lt_0.05"],
            best["n_cosine_gt_0.999"],
            best["max_cosine"],
        ):
            best = record
        if record["n_l2_lt_0.05"] >= 15 or n_cos >= 15:
            break

    chosen = best
    matched = chosen is not None and (chosen["n_l2_lt_0.05"] >= 5 or chosen["n_cosine_gt_0.999"] >= 5)
    payload = {
        "official_norm_mean": float(official_norms.mean()),
        "official_norm_std": float(official_norms.std()),
        "matched": matched,
        "chosen": chosen,
        "results": results,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not matched:
        raise SystemExit(f"no encoder matched the released queries; see {out_path}")
    return payload


def _embed_texts(texts, tokenizer, model, device, prefix: str, pool: str, batch_size: int) -> np.ndarray:
    import torch

    rows = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = [prefix + text for text in texts[start : start + batch_size]]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.no_grad():
            hidden = model(**tokens).last_hidden_state
            if pool == "cls":
                vec = hidden[:, 0]
            else:
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                vec = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            vec = vec.float()
        rows.append(vec.cpu().numpy().astype(np.float32))
    return np.concatenate(rows, axis=0)


def _clean_rephrase(text: str) -> str:
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    for prefix in ("Rephrased question:", "Rephrased:", "Question:", "Paraphrase:"):
        if line.lower().startswith(prefix.lower()):
            line = line[len(prefix) :].strip()
    return line.strip(" \"'`-")


def _acceptable(text: str, banned: set[str]) -> bool:
    words = text.split()
    if len(words) < 4 or len(words) > 60:
        return False
    if text in banned or text.casefold() in {item.casefold() for item in banned}:
        return False
    lowered = text.casefold()
    if lowered.startswith(("sure", "here", "i ", "the rephrase", "paraphrase")):
        return False
    return "?" in text or lowered.split()[0] in {
        "who", "what", "which", "where", "when", "why", "how", "in", "at", "name", "whose",
    }


def generate_paraphrases(
    pending_by_original: dict[str, list[str]],
    banned_by_original: dict[str, set[str]],
    resume_path: str,
    batch_size: int,
    force_keys: set[str] | None = None,
) -> dict[str, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    force_keys = force_keys or set()
    done: dict[str, str] = {}
    resume = Path(resume_path)
    if resume.exists():
        for line in resume.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row["key"] not in force_keys:
                    done[row["key"]] = row["text"]
    remaining = {key: keys for key, keys in pending_by_original.items()}
    flat = [(original, key) for original, keys in remaining.items() for key in keys if key not in done]
    if not flat:
        return done
    print(f"generating {len(flat)} paraphrases ({len(done)} already resumed)", flush=True)

    name = os.environ.get("PARAPHRASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)
    model.to("cuda")
    model.eval()
    resume.parent.mkdir(parents=True, exist_ok=True)
    used_by_original = {original: set(banned) for original, banned in banned_by_original.items()}
    for original in pending_by_original:
        used_by_original.setdefault(original, set()).add(original)
    for original, keys in pending_by_original.items():
        for key in keys:
            if key in done:
                used_by_original[original].add(done[key])

    still = [(original, key) for original, key in flat if key not in done]
    attempts: dict[str, int] = {}
    rounds = 0
    while still:
        rounds += 1
        batch = still[:batch_size]
        prompts = []
        for original, _key in batch:
            banned = list(used_by_original[original])[:8]
            banned_txt = "\n".join(f"- {item}" for item in banned)
            user = (
                "Rephrase the trivia question as one new single-line question with the same meaning. "
                "Reply with only that question. Do not answer it. Do not copy a wording already used.\n\n"
                f"Question: {original}\n"
                f"Already used:\n{banned_txt}"
            )
            if hasattr(tokenizer, "apply_chat_template"):
                prompts.append(tokenizer.apply_chat_template(
                    [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
                ))
            else:
                prompts.append(f"<s>[INST] {user} [/INST]")
        tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        tokens = {key: value.to(model.device) for key, value in tokens.items()}
        with torch.no_grad():
            output = model.generate(
                **tokens,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = output[:, tokens["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        retry = []
        with resume.open("a", encoding="utf-8") as handle:
            for (original, key), raw in zip(batch, decoded):
                text = _clean_rephrase(raw)
                if _acceptable(text, used_by_original[original]):
                    used_by_original[original].add(text)
                    done[key] = text
                    handle.write(json.dumps({"key": key, "text": text}) + "\n")
                else:
                    attempts[key] = attempts.get(key, 0) + 1
                    if attempts[key] > 40:
                        raise SystemExit(f"could not produce a new paraphrase for {key} ({original!r})")
                    retry.append((original, key))
        still = retry + still[batch_size:]
        if rounds % 20 == 0 or not still:
            print(f"paraphrases done={len(done)} left={len(still)} rounds={rounds}", flush=True)
        if rounds > 100000:
            raise SystemExit("paraphrase generation did not converge")
    return done


def embed_unique(texts: list[str], chosen: dict, batch_size: int) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(chosen["model"])
    model = AutoModel.from_pretrained(chosen["model"], torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model.to(device)
    model.eval()
    matrix = _embed_texts(texts, tokenizer, model, device, chosen["prefix"], chosen["pool"], batch_size)
    if chosen.get("normalize", True):
        matrix = _normalize(matrix).astype(np.float32)
    return matrix


def build(args: argparse.Namespace) -> None:
    groups = load_groups(args.questions)
    planned = plan_streams(
        groups, args.n_stream, args.seed, tuple(args.skew), args.distribution
    )
    frozen = (
        plan_frozen_uniform(load_raw_groups(args.questions), args.n_stream, args.seed)
        if args.frozen_uniform
        else None
    )
    for skew, stream in planned.items():
        print(
            f"skew {skew}: pool={stream['n_pool']} new={stream['n_pending']} "
            f"max_count={stream['max_count']} overflowing_topics={stream['topics_overflowing']}",
            flush=True,
        )
    if frozen is not None:
        print(
            f"simzipf uniform: n_stream={args.n_stream} pool_lines={GROUP_LINES} "
            f"topics={len(load_raw_groups(args.questions))}",
            flush=True,
        )
    if args.plan_only:
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_path = out_dir / "encoder.json"
    if args.probe_json:
        probe = json.loads(Path(args.probe_json).read_text(encoding="utf-8"))
    elif probe_path.exists() and not args.reprobe:
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
    else:
        sample_questions = [line for group in groups[:20] for line in group]
        # load_groups already dropped duplicate lines, so rebuild the raw sample
        # from the file head to stay aligned with released wording.
        probe = probe_encoder(read_question_lines(args.questions), args.official, str(probe_path))
    if not probe.get("matched"):
        raise SystemExit(f"encoder probe did not match: {probe_path}")
    chosen = probe["chosen"]
    print(f"encoder {chosen['model']} prefix={chosen['prefix']!r}", flush=True)

    originals = [group[0] for group in groups]
    pending_by_original: dict[str, list[str]] = {}
    banned_by_original = {originals[i]: set(groups[i]) for i in range(len(groups))}
    for stream in planned.values():
        for key in stream["pending"]:
            topic = int(key.split(":")[1])
            pending_by_original.setdefault(originals[topic], []).append(key)
    generated = generate_paraphrases(
        pending_by_original,
        banned_by_original,
        str(out_dir / "paraphrases.jsonl"),
        args.batch_size,
    )
    for stream in planned.values():
        for key in stream["pending"]:
            stream["texts"][key] = generated[key]

    def casefold_collisions() -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for skew, stream in planned.items():
            seen: set[str] = set()
            bad: list[str] = []
            for key in stream["keys"]:
                token = stream["texts"][key].casefold()
                if token in seen:
                    bad.append(key)
                else:
                    seen.add(token)
            if bad:
                found[skew] = bad
        return found

    def replace_keys(collisions: dict[str, list[str]]) -> None:
        pending_fix: dict[str, list[str]] = {}
        banned_fix: dict[str, set[str]] = {}
        force: set[str] = set()
        for skew, bad_keys in collisions.items():
            stream = planned[skew]
            bad_set = set(bad_keys)
            seen: set[str] = set()
            banned_texts: set[str] = set()
            for key in stream["keys"]:
                if key in bad_set:
                    continue
                token = stream["texts"][key].casefold()
                if token not in seen:
                    seen.add(token)
                    banned_texts.add(stream["texts"][key])
            for key in bad_keys:
                topic = int(key.split(":")[1])
                original = originals[topic]
                pending_fix.setdefault(original, []).append(key)
                ban = banned_fix.setdefault(original, set(groups[topic]))
                ban.update(banned_texts)
                ban.add(stream["texts"][key])
                force.add(key)
        print(f"replacing {len(force)} duplicate texts", flush=True)
        generated_fix = generate_paraphrases(
            pending_fix,
            banned_fix,
            str(out_dir / "paraphrases.jsonl"),
            args.batch_size,
            force,
        )
        for stream in planned.values():
            for key in force:
                if key in stream["texts"]:
                    stream["texts"][key] = generated_fix[key]

    for _round in range(6):
        collisions = casefold_collisions()
        if not collisions:
            break
        replace_keys(collisions)
    if casefold_collisions():
        raise SystemExit("could not remove duplicate texts from the streams")

    def embed_cached(texts: list[str]) -> np.ndarray:
        texts_path = out_dir / "unique_texts.txt"
        emb_path = out_dir / "unique_embeddings.npy"
        cached_matrix = None
        cached_index: dict[str, int] = {}
        cached_texts: list[str] = []
        if texts_path.exists() and emb_path.exists():
            cached_texts = texts_path.read_text(encoding="utf-8").split("\n")
            if cached_texts and cached_texts[-1] == "":
                cached_texts.pop()
            loaded = np.load(emb_path)
            if len(cached_texts) == loaded.shape[0]:
                cached_matrix = loaded
                cached_index = {text: i for i, text in enumerate(cached_texts)}
            else:
                cached_texts = []
        missing = [text for text in texts if text not in cached_index]
        print(
            f"embedding {len(missing)} new questions ({len(texts) - len(missing)} cached)",
            flush=True,
        )
        fresh = embed_unique(missing, chosen, args.embed_batch_size) if missing else None
        if cached_matrix is not None:
            dim = int(cached_matrix.shape[1])
        elif fresh is not None:
            dim = int(fresh.shape[1])
        else:
            raise SystemExit("no texts to embed")
        out = np.empty((len(texts), dim), dtype=np.float32)
        fresh_at = {text: i for i, text in enumerate(missing)}
        for i, text in enumerate(texts):
            if text in fresh_at:
                out[i] = fresh[fresh_at[text]]
            else:
                out[i] = cached_matrix[cached_index[text]]
        if cached_matrix is not None and fresh is not None:
            stored = np.concatenate([np.asarray(cached_matrix), fresh], axis=0)
            stored_texts = cached_texts + missing
        elif cached_matrix is not None:
            stored = np.asarray(cached_matrix)
            stored_texts = cached_texts
        else:
            stored = fresh
            stored_texts = list(missing)
        np.save(emb_path, stored)
        texts_path.write_text("\n".join(stored_texts) + "\n", encoding="utf-8")
        return out

    matrix = None
    index_of: dict[str, int] = {}
    for _round in range(4):
        unique_texts: list[str] = []
        index_of = {}
        embed_streams = list(planned.values())
        if frozen is not None:
            embed_streams.append(frozen)
        for stream in embed_streams:
            for key in stream["keys"]:
                text = stream["texts"][key]
                if text not in index_of:
                    index_of[text] = len(unique_texts)
                    unique_texts.append(text)
        matrix = embed_cached(unique_texts)
        vector_collisions: dict[str, list[str]] = {}
        for skew, stream in planned.items():
            seen_vec: set[bytes] = set()
            bad: list[str] = []
            for key in stream["keys"]:
                token = matrix[index_of[stream["texts"][key]]].tobytes()
                if token in seen_vec:
                    bad.append(key)
                else:
                    seen_vec.add(token)
            if bad:
                vector_collisions[skew] = bad
        if not vector_collisions:
            break
        replace_keys(vector_collisions)
    else:
        raise SystemExit("could not remove duplicate vectors from the streams")
    if matrix is None:
        raise SystemExit("embeddings were not built")

    for skew, stream in planned.items():
        rows = np.stack([matrix[index_of[stream["texts"][key]]] for key in stream["keys"]])
        tag = f"simplus-{args.n_stream // 1000}k-{skew}"
        skew_value = None if skew == "uniform" else float(skew)
        stream_dir = out_dir / tag
        stream_dir.mkdir(parents=True, exist_ok=True)
        np.save(stream_dir / "query.npy", rows.astype(np.float32))
        np.save(stream_dir / "topics.npy", stream["topics"])
        write_query_bin(rows, str(stream_dir / "query.bin"))
        (stream_dir / "query.txt").write_text(
            "\n".join(stream["texts"][key] for key in stream["keys"]) + "\n",
            encoding="utf-8",
        )
        unique_rows = len({row.tobytes() for row in np.ascontiguousarray(rows)})
        stats = {
            "construction": "triviaqa_simzipf_plus_unique_rephrase",
            "skew": skew_value,
            "distribution": "uniform" if skew == "uniform" else "zipfian",
            "seed": args.seed,
            "n_topics": len(groups),
            "n_stream": int(rows.shape[0]),
            "dim": int(rows.shape[1]),
            "dtype": "float32",
            "unique_rows": int(unique_rows),
            "n_from_released_pool": stream["n_pool"],
            "n_llm_paraphrases": stream["n_pending"],
            "encoder": chosen["model"],
            "query_prefix": chosen["prefix"],
            "pooling": chosen["pool"],
            "zipf_scrambled": skew != "uniform",
        }
        (stream_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(stats), flush=True)
        if unique_rows != rows.shape[0]:
            raise SystemExit(f"{tag} has {rows.shape[0] - unique_rows} duplicate vectors")

    if frozen is not None:
        rows = np.stack([matrix[index_of[frozen["texts"][key]]] for key in frozen["keys"]])
        tag = f"sim-{args.n_stream // 1000}k-uniform"
        stream_dir = out_dir / tag
        stream_dir.mkdir(parents=True, exist_ok=True)
        np.save(stream_dir / "query.npy", rows.astype(np.float32))
        np.save(stream_dir / "topics.npy", frozen["topics"])
        write_query_bin(rows, str(stream_dir / "query.bin"))
        (stream_dir / "query.txt").write_text(
            "\n".join(frozen["texts"][key] for key in frozen["keys"]) + "\n",
            encoding="utf-8",
        )
        unique_rows = len({row.tobytes() for row in np.ascontiguousarray(rows)})
        stats = {
            "construction": "triviaqa_simzipf_uniform_frozen_pool",
            "skew": None,
            "distribution": "uniform",
            "seed": args.seed,
            "n_topics": len(groups),
            "n_stream": int(rows.shape[0]),
            "group_lines": GROUP_LINES,
            "dim": int(rows.shape[1]),
            "dtype": "float32",
            "unique_rows": int(unique_rows),
            "exact_repeat_rows": int(rows.shape[0] - unique_rows),
            "encoder": chosen["model"],
            "query_prefix": chosen["prefix"],
            "pooling": chosen["pool"],
            "zipf_scrambled": False,
        }
        (stream_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(stats), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build TriviaQA simZipf+ query streams")
    parser.add_argument("--questions", required=True, help="Aker triviaqa.txt")
    parser.add_argument("--official", required=True, help="Released simZipf npy used to identify the encoder")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-stream", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skew", type=float, nargs="+", default=list(SKEWS))
    parser.add_argument("--distribution", default="zipfian", choices=["zipfian", "uniform"])
    parser.add_argument(
        "--frozen-uniform",
        action="store_true",
        help="Also write simZipf uniform (sample the frozen 51-line pool with replacement)",
    )
    parser.add_argument("--probe-json", default="", help="Reuse an encoder probe instead of running one")
    parser.add_argument("--reprobe", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32, help="Paraphrase generation batch")
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
