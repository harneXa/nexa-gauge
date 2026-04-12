import json
from pathlib import Path

GROUNDING_FILE = Path(__file__).parent.parent.parent / "data" / "grounding.json"
OUTPUT_FILE = Path(__file__).parent / "grounding_metrics.json"


def main():
    with open(GROUNDING_FILE) as f:
        records = json.load(f)

    output = []
    total_cost = 0.0
    total_input_tokens = 0.0
    total_output_tokens = 0.0

    for entry in records:
        case_id = entry.get("record", {}).get("case_id", "unknown")
        gen = entry.get("inputs", {}).get("generation", {})
        gen_id = gen.get("id", "unknown")
        gen_text = gen.get("text", "")

        grounding_metrics = entry.get("grounding_metrics", {})
        metrics = grounding_metrics.get("metrics", [])
        cost_entry = grounding_metrics.get("cost", {})

        input_tokens = cost_entry.get("input_tokens") or 0.0
        output_tokens = cost_entry.get("output_tokens") or 0.0
        cost = cost_entry.get("cost") or 0.0

        total_cost += cost
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        parsed_metrics = []
        for metric in metrics:
            claims = [
                {
                    "id": r.get("item", {}).get("id"),
                    "text": r.get("item", {}).get("text"),
                    "tokens": r.get("item", {}).get("tokens"),
                    "verdict": r.get("verdict"),
                    "confidence": r.get("confidence"),
                }
                for r in metric.get("result", [])
            ]
            parsed_metrics.append(
                {
                    "name": metric.get("name"),
                    "category": metric.get("category"),
                    "score": metric.get("score"),
                    "error": metric.get("error"),
                    "claims": claims,
                }
            )

        record_entry = {
            "case_id": case_id,
            "generation_id": gen_id,
            "generation_text": gen_text,
            "llm_call": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
            },
            "metrics": parsed_metrics,
        }
        output.append(record_entry)

        # --- print ---
        print(f"case_id={case_id}  record={gen_id}  ------------")
        print(f"text = {gen_text[:80]}{'...' if len(gen_text) > 80 else ''}")
        print(f"input_tokens={input_tokens}, output_tokens={output_tokens}, cost={cost:.9f}")
        if parsed_metrics:
            for metric in parsed_metrics:
                print(
                    f"  metric={metric['name']}  category={metric['category']}  score={metric['score']}"
                )
                for claim in metric["claims"]:
                    print(f"    claim_text    = {claim['text']}")
                    print(f"    claim_tokens  = {claim['tokens']}")
                    print(f"    verdict       = {claim['verdict']}")
                    print(f"    confidence    = {claim['confidence']}")
        else:
            print("  (no metrics)")
        print()

    print("=" * 60)
    print(
        f"TOTAL: input_tokens={total_input_tokens}, output_tokens={total_output_tokens}, cost={total_cost:.9f}"
    )

    result = {
        "total": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cost": total_cost,
        },
        "records": output,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nOutput written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
