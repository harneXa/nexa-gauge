import json
from pathlib import Path

CLAIMS_FILE = Path(__file__).parent.parent.parent / "data" / "claims.json"
OUTPUT_FILE = Path(__file__).parent / "generation_claims_cost.json"


def main():
    with open(CLAIMS_FILE) as f:
        records = json.load(f)

    total_cost = 0.0
    total_input_tokens = 0.0
    total_output_tokens = 0.0
    output = []

    for entry in records:
        inputs = entry.get("inputs", {})
        if not inputs.get("has_generation"):
            continue

        gen = inputs.get("generation", {})
        gen_id = gen.get("id", "unknown")
        gen_text = gen.get("text", "")

        generation_claims = entry.get("generation_claims", {})
        costs = generation_claims.get("cost", [])

        input_tokens = sum(c.get("input_tokens", 0) for c in costs)
        output_tokens = sum(c.get("output_tokens", 0) for c in costs)
        cost = sum(c.get("cost", 0) for c in costs)

        total_cost += cost
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        claims = generation_claims.get("claims", [])
        claim_entries = [
            {
                "text": claim.get("item", {}).get("text", ""),
                "tokens": claim.get("item", {}).get("tokens"),
            }
            for claim in claims
        ]

        record_entry = {
            "id": gen_id,
            "text": gen_text,
            "text_tokens": gen.get("tokens"),
            "llm_call": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": round(cost, 9),
            },
            "claims": claim_entries,
        }
        output.append(record_entry)

        # --- print ---
        print(f"record {gen_id}  ------------")
        print(f"text = {gen_text[:80]}{'...' if len(gen_text) > 80 else ''}")
        print(f"text_tokens = {gen.get('tokens')}")
        print(f"input_tokens={input_tokens}, output_tokens={output_tokens}, cost={cost:.9f}")
        for c in claim_entries:
            print(f"  claim_text   = {c['text']}")
            print(f"  claim_tokens = {c['tokens']}")
        print()

    print("=" * 60)
    print(
        f"TOTAL: input_tokens={total_input_tokens}, output_tokens={total_output_tokens}, cost={total_cost:.9f}"
    )

    result = {
        "total": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cost": round(total_cost, 9),
        },
        "records": output,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nOutput written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
