from aer.models.encoder.hypo_encoder import HypothesisEncoder, encode_hypothesis_set

def main():
    print("\n=== INITIALIZING ENCODER ===")
    encoder = HypothesisEncoder()


    h_a = "A man fired twice at Shinzo Abe, causing his assassination."
    h_b = "Former Prime Minister Shinzo Abe was shot and killed during a public speech."
    h_wrong = [
        "The event was caused by an unrelated economic policy.",
        "A natural disaster triggered this event."
    ]

    print("\n=== ENCODING HYPOTHESES ===")
    bundle = encode_hypothesis_set(encoder, h_a, h_b, h_wrong)

    print("\n--- EMBEDDING SHAPES ---")
    print("H_a:", bundle.emb_a.shape)
    print("H_b:", bundle.emb_b.shape)
    print("H_wrong (count):", len(bundle.emb_wrong))

    print("\n--- SAMPLE VALUES (first 5 dims) ---")
    print("H_a:", bundle.emb_a[:5])
    print("H_b:", bundle.emb_b[:5])

    if bundle.emb_wrong:
        print("H_wrong[0]:", bundle.emb_wrong[0][:5])

    print("\n=== DONE ===\n")


if __name__ == "__main__":
    main()

