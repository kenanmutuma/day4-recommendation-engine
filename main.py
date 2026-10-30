"""Runs the recommendation engine end to end:  python main.py"""
from recommender import DATA_PATH, OUTPUT_PATH, RecommendationEngine, read_purchases


def print_matrix(engine: RecommendationEngine, count: int = 5) -> None:
    """Task 3 sanity check: values in [0, 1], 1.00 on the diagonal."""
    ids = engine.customers[:count]
    print("Similarity matrix (first {} customers)".format(count))
    print("       " + "".join("{:>7}".format(c) for c in ids))
    for a in ids:
        print("{:>6} ".format(a) + "".join("{:>7.2f}".format(engine.similarity(a, b)) for b in ids))
    print()


def show_customer(engine: RecommendationEngine, customer_id: str) -> None:
    """Console layout from expected_output.md for one customer."""
    bought = sorted(engine.history(customer_id))
    print("Customer {} ({}) bought: {}".format(customer_id, engine.customer_name(customer_id),
                                                "{" + ", ".join(bought) + "}"))
    # Show exactly the neighbours the recommendations are built from.
    print("Most similar customers:")
    neighbours = engine.similar_customers(customer_id, engine.k_neighbours)
    if not neighbours:
        print("  (none: no other customer shares a product)")
    for other, value in neighbours:
        print("  {}  jaccard={:.2f}".format(other, value))
    print("Recommendations:")
    recs = engine.recommend(customer_id, 3)
    if not recs:
        print("  (nothing to recommend: this customer has bought every product)")
    for rec in recs:
        if rec["fallback"]:
            reason = "it is popular ({} customers bought it)".format(rec["popularity"])
        else:
            count = len(rec["supporting_customers"])
            reason = "{} similar customer{} bought it".format(count, "" if count == 1 else "s")
        print("  {}. {} ({})  score={:.3f}  because {}".format(
            rec["rank"], rec["product_name"], rec["category"], rec["score"], reason))
    print()


def main() -> None:
    valid, rejected = read_purchases(DATA_PATH)
    total = len(valid) + len(rejected)

    print("Rejected rows")
    for row in rejected:
        print("  line {:>2}  {}  ({})".format(row["line"], row["purchase_id"], row["reason"]))
    print()

    engine = RecommendationEngine(valid)
    print_matrix(engine)

    # A normal customer, and one whose ID was written in lower case in the raw file.
    show_customer(engine, "C107")
    show_customer(engine, "C106")

    rows = engine.recommend_all(n=3)
    engine.export_csv(rows, OUTPUT_PATH)

    written = sum(1 for row in rows if row["product_id"])
    print("Run summary")
    print("Purchases read: {} | used: {} | rejected: {}".format(total, len(valid), len(rejected)))
    print("Customers: {} | Products: {} | Recommendations written: {}".format(
        len(engine.customers), len(engine.products), written))
    print("Saved to {}".format(OUTPUT_PATH))


if __name__ == "__main__":
    main()
