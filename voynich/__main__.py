"""
CLI for the Voynich decipherment workbench.

  python -m voynich setup                 download all datasets
  python -m voynich gui [--port 5000]     launch the browser GUI
  python -m voynich solve [...]           run a decipherment attempt
  python -m voynich sweep [...]           run one attempt per reference language, ranked
  python -m voynich benchmark [...]       validate the solver on a known cipher
"""

from __future__ import annotations

import argparse
import json
import sys


def _progress_printer(every: int = 5000):
    last = {"step": -every}

    def cb(p: dict) -> None:
        step = p["restart"] * p["total_iterations"] + p["iteration"]
        if step - last["step"] >= every:
            last["step"] = step
            print(
                f"  restart {p['restart'] + 1}/{p['restarts']} "
                f"iter {p['iteration']}/{p['total_iterations']} "
                f"T={p['temperature']:.2e} best={p['best_score']:.4f} bits/char"
            )

    return cb


def cmd_setup(args) -> int:
    from . import corpus

    status = corpus.ensure_all(force=args.force)
    print(json.dumps(status, indent=2))
    return 0


def cmd_gui(args) -> int:
    from .webapp.app import main as gui_main

    print(f"Voynich Decipherment Workbench: http://{args.host}:{args.port}/")
    gui_main(host=args.host, port=args.port, debug=args.debug)
    return 0


def cmd_solve(args) -> int:
    from .pipeline import save_report, solve_voynich

    config = {
        "currier_language": args.language,
        "section": args.section,
        "reference": args.reference,
        "hypothesis": args.hypothesis,
        "abjad": args.abjad,
        "lock_rounds": args.lock_rounds,
        "order": args.order,
        "bpe_merges": args.bpe_merges,
        "iterations": args.iterations,
        "restarts": args.restarts,
        "seed": args.seed,
    }
    print(f"Solving: {config}")
    report = solve_voynich(config, progress=_progress_printer())
    path = save_report(report)

    s = report["scores"]
    print("\n=== Result ===")
    print(f"train best      : {s['train_best']:.4f} bits/char")
    print(f"held-out score  : {s['test_heldout']:.4f} bits/char")
    print(f"random-key floor: {s['random_key_floor']:.4f}")
    print(f"language ceiling: {s['reference_ceiling']:.4f}")
    print(f"gap closed      : {s['gap_closed'] * 100:.1f}%")
    print(f"dict word match : {s['word_match_rate'] * 100:.1f}%")
    for entry in report.get("locking", []):
        print(f"  lock round {entry['round'] + 1}: {entry['locked_tokens']} tokens frozen")
    print(f"\n{report['verdict']}\n")
    print("Decoded held-out sample:")
    for row in report["decoded_sample"][:10]:
        print(f"  {row['ref']:>10}  {row['text']}")
    print(f"\nFull report: {path}")
    return 0


def cmd_sweep(args) -> int:
    from .pipeline import save_report, sweep_references

    config = {
        "currier_language": args.language,
        "section": args.section,
        "hypothesis": args.hypothesis,
        "abjad": args.abjad,
        "order": args.order,
        "bpe_merges": args.bpe_merges,
        "iterations": args.iterations,
        "restarts": args.restarts,
        "seed": args.seed,
    }
    refs = args.references.split(",") if args.references else None
    print(f"Sweeping {len(refs) if refs else 'all'} references: {config}")

    def lang_progress(p):
        if p["iteration"] == p["total_iterations"]:
            print(f"  [{p['lang_index'] + 1}/{p['n_langs']}] {p['language']}: "
                  f"restart {p['restart'] + 1}/{p['restarts']} done "
                  f"(best {p['best_score']:.4f})")

    report = sweep_references(config, references=refs, progress=lang_progress)
    path = save_report(report)

    print(f"\n=== Ranked results ===")
    print(f"{'#':>2} {'gap':>7} {'match':>7} {'held-out':>9} {'floor':>8} {'ceiling':>8}  reference")
    for i, row in enumerate(report["table"], 1):
        print(f"{i:>2} {row['gap_closed'] * 100:6.1f}% {row['word_match_rate'] * 100:6.1f}% "
              f"{row['test_heldout']:9.3f} "
              f"{row['random_key_floor']:8.3f} {row['reference_ceiling']:8.3f}  {row['label']}")
        print(f"{'':19}sample: {row['sample'][:70]}")
    print(f"\n{report['note']}")
    print(f"Full report: {path}")
    return 0


def cmd_benchmark(args) -> int:
    from . import corpus
    from .synthetic import run_benchmark

    text = corpus.load_reference(args.reference)
    print(f"Benchmark: {args.reference}, {args.mode}, {args.cipher_chars} chars, "
          f"{args.iterations} iters x {args.restarts} restarts")
    rep = run_benchmark(
        text,
        order=args.order,
        cipher_chars=args.cipher_chars,
        iterations=args.iterations,
        restarts=args.restarts,
        seed=args.seed,
        mode=args.mode,
        progress=_progress_printer(),
    )
    print(f"\nletters recovered : {rep['accuracy'] * 100:.1f}%")
    print(f"solver score      : {rep['best_score']:.4f} bits/char")
    print(f"true-key score    : {rep['true_key_score']:.4f}")
    print(f"random-key floor  : {rep['random_key_score_mean']:.4f}")
    print(f"elapsed           : {rep['elapsed_sec']:.1f}s")
    print(f"\ndecoded : {rep['decoded_preview'][:160]}")
    print(f"truth   : {rep['plaintext_preview'][:160]}")
    return 0 if rep["accuracy"] > 0.9 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voynich", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("setup", help="download datasets")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_setup)

    p = sub.add_parser("gui", help="launch the browser GUI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--debug", action="store_true")
    p.set_defaults(fn=cmd_gui)

    from .corpus import REFERENCE_SOURCES
    from .pipeline import HYPOTHESES

    references = list(REFERENCE_SOURCES)

    p = sub.add_parser("solve", help="run a decipherment attempt")
    p.add_argument("--language", default="A", choices=["A", "B", "all"])
    p.add_argument("--section", default=None)
    p.add_argument("--reference", default="latin", choices=references)
    p.add_argument("--hypothesis", default="simple", choices=list(HYPOTHESES))
    p.add_argument("--abjad", action="store_true",
                   help="score against the reference's consonant skeleton")
    p.add_argument("--lock-rounds", type=int, default=0,
                   help="crib-locking rounds: freeze dictionary-supported "
                        "tokens and re-anneal (simple/anagram)")
    p.add_argument("--order", type=int, default=4, choices=[3, 4])
    p.add_argument("--bpe-merges", type=int, default=30)
    p.add_argument("--iterations", type=int, default=60000)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(fn=cmd_solve)

    p = sub.add_parser("sweep", help="rank all reference languages on one configuration")
    p.add_argument("--language", default="A", choices=["A", "B", "all"])
    p.add_argument("--section", default=None)
    p.add_argument("--references", default=None,
                   help="comma-separated subset (default: all)")
    p.add_argument("--hypothesis", default="simple", choices=list(HYPOTHESES))
    p.add_argument("--abjad", action="store_true",
                   help="score against consonant skeletons")
    p.add_argument("--order", type=int, default=4, choices=[3, 4])
    p.add_argument("--bpe-merges", type=int, default=30)
    p.add_argument("--iterations", type=int, default=40000)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("benchmark", help="validate solver on a known cipher")
    p.add_argument("--reference", default="english", choices=references)
    p.add_argument("--mode", default="substitution",
                   choices=["substitution", "abbreviation", "anagram"])
    p.add_argument("--cipher-chars", type=int, default=4000)
    p.add_argument("--order", type=int, default=4, choices=[3, 4])
    p.add_argument("--iterations", type=int, default=20000)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(fn=cmd_benchmark)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
