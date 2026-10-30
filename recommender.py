"""
Day 4 - Customer Recommendation Engine (Mama Mboga Online)
Squad Python Specialization - Data Science, AI & Analytics

Core Python only (csv, collections, datetime, pathlib). No ML libraries.

Idea in one sentence: find the customers whose baskets overlap most with yours
(Jaccard similarity), then suggest what they bought that you have not, ranked by
how similar the buyers are plus a small bonus for products many customers buy.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DATA_PATH = Path("data") / "purchases.csv"
OUTPUT_PATH = Path("output") / "recommendations.csv"

REQUIRED_COLUMNS = (
    "purchase_id", "customer_id", "customer_name",
    "product_id", "product_name", "category", "purchase_date", "quantity",
)

# Ranking rule: score = sum of supporter similarities + POPULARITY_WEIGHT * popularity share
POPULARITY_WEIGHT = 0.5

CSV_COLUMNS = (
    "customer_id", "customer_name", "rank", "product_id",
    "product_name", "category", "score", "supporting_customers",
)


class UnknownCustomerError(KeyError):
    """Raised when a customer ID is not in the purchase history."""

    def __str__(self) -> str:  # KeyError normally shows the repr of its argument
        return "Unknown customer ID: " + str(self.args[0])


# ---------------------------------------------------------------------------
# Task 1: reading the data
# ---------------------------------------------------------------------------
def _clean(value: str | None) -> str:
    return (value or "").strip()


def read_purchases(path: Path = DATA_PATH) -> tuple[list[dict], list[dict]]:
    """Return (valid_rows, rejected_rows).

    Rules (each one is also listed in the README):
      * IDs are trimmed and upper-cased, so 'c106' is repaired to 'C106' (kept, not rejected).
      * A row is rejected when customer_id, product_id or product_name is blank.
      * A row is rejected when quantity is not a whole number above zero.
      * A row is rejected when purchase_date is not a valid YYYY-MM-DD date.
      * A repeated purchase_id is rejected as a duplicate; the first copy is kept.
    Every rejected row is returned with its line number and a reason, so nothing vanishes silently.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Purchase file not found: " + str(path))

    valid: list[dict] = []
    rejected: list[dict] = []
    seen_ids: set[str] = set()

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError("Purchase file is missing columns: " + ", ".join(missing))

        for line_number, raw in enumerate(reader, start=2):  # header is line 1
            row = {key: _clean(raw.get(key)) for key in REQUIRED_COLUMNS}
            row["customer_id"] = row["customer_id"].upper()
            row["product_id"] = row["product_id"].upper()

            reason = None
            if not row["customer_id"]:
                reason = "blank customer_id"
            elif not row["product_id"] or not row["product_name"]:
                reason = "blank product"
            else:
                try:
                    quantity = int(row["quantity"])
                except ValueError:
                    quantity = 0
                if quantity <= 0:
                    reason = "quantity is not a whole number above zero"
                else:
                    row["quantity"] = quantity
                    try:
                        datetime.strptime(row["purchase_date"], "%Y-%m-%d")
                    except ValueError:
                        reason = "invalid purchase_date"
            if reason is None and row["purchase_id"] in seen_ids:
                reason = "duplicate purchase_id"

            if reason:
                rejected.append({"line": line_number, "reason": reason, **raw})
            else:
                seen_ids.add(row["purchase_id"])
                valid.append(row)
    return valid, rejected


# ---------------------------------------------------------------------------
# Task 3: Jaccard similarity
# ---------------------------------------------------------------------------
def jaccard(a: set, b: set) -> float:
    """|A intersection B| / |A union B|.

    Two empty sets share nothing and describe no one, so the similarity is defined as 0.0
    (this also avoids dividing by zero).
    """
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
class RecommendationEngine:
    """Recommends products a customer has not bought, based on similar customers."""

    def __init__(self, purchases: list[dict], min_similarity: float = 0.0,
                 k_neighbours: int = 5) -> None:
        # Everything is built once here. Later methods never re-scan `purchases`.
        self.min_similarity = min_similarity
        self.k_neighbours = k_neighbours

        histories: dict[str, set[str]] = defaultdict(set)
        names: dict[str, str] = {}
        products: dict[str, dict] = {}
        buyers: dict[str, set[str]] = defaultdict(set)

        for row in purchases:
            customer, product = row["customer_id"], row["product_id"]
            histories[customer].add(product)
            names[customer] = row["customer_name"]
            products[product] = {"name": row["product_name"], "category": row["category"]}
            buyers[product].add(customer)

        self._histories = dict(histories)
        self._names = names
        self._products = products
        # Popularity = number of different customers who bought the product
        # (not units sold, so one big basket cannot inflate it).
        self._popularity = {product: len(who) for product, who in buyers.items()}
        self._customers = sorted(self._histories)

    # -- simple lookups ----------------------------------------------------
    @property
    def customers(self) -> list[str]:
        return list(self._customers)

    @property
    def products(self) -> list[str]:
        return sorted(self._products)

    def customer_name(self, customer_id: str) -> str:
        self.history(customer_id)  # raises if unknown
        return self._names[customer_id]

    def product_name(self, product_id: str) -> str:
        return self._products[product_id]["name"]

    def category(self, product_id: str) -> str:
        return self._products[product_id]["category"]

    def history(self, customer_id: str) -> set[str]:
        """The set of product IDs this customer bought (a copy, so callers cannot change it)."""
        try:
            return set(self._histories[customer_id])
        except KeyError:
            raise UnknownCustomerError(customer_id) from None

    # -- Task 3 ------------------------------------------------------------
    def similarity(self, customer_a: str, customer_b: str) -> float:
        return jaccard(self.history(customer_a), self.history(customer_b))

    # -- Task 4 ------------------------------------------------------------
    def similar_customers(self, customer_id: str, k: int = 3) -> list[tuple[str, float]]:
        """Top-k other customers as (customer_id, score), best first.

        Excluded: the customer, anyone below min_similarity, and anyone with similarity 0
        (no shared product means no evidence). Ties are broken by customer ID, ascending,
        so the answer is the same on every run.
        """
        own = self.history(customer_id)
        scored = []
        for other in self._customers:
            if other == customer_id:
                continue
            value = jaccard(own, self._histories[other])
            if value > 0 and value >= self.min_similarity:
                scored.append((other, value))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]

    # -- Task 5 ------------------------------------------------------------
    def candidate_products(self, customer_id: str, k: int = 3) -> dict[str, dict]:
        """Products the k nearest neighbours bought that this customer has not.

        Evidence per product: name, category, the supporting customers with their
        similarity, the sum of those similarities, and its popularity.

        Fallback: if the neighbours add nothing (new customer, or no overlap), the most
        popular products the customer has not bought are offered instead, with no
        supporters. The same score() rule is applied to them.
        """
        own = self.history(customer_id)
        candidates: dict[str, dict] = {}

        for neighbour, value in self.similar_customers(customer_id, k):
            for product in sorted(self._histories[neighbour] - own):
                entry = candidates.setdefault(product, self._blank_evidence(product))
                entry["supporters"].append((neighbour, value))
                entry["similarity_sum"] += value

        if not candidates:
            for product in self._products:
                if product not in own:
                    candidates[product] = self._blank_evidence(product)
                    candidates[product]["fallback"] = True
        return candidates

    def _blank_evidence(self, product: str) -> dict:
        return {
            "product_id": product,
            "name": self._products[product]["name"],
            "category": self._products[product]["category"],
            "supporters": [],
            "similarity_sum": 0.0,
            "popularity": self._popularity[product],
            "fallback": False,
        }

    # -- Task 6 ------------------------------------------------------------
    def score(self, evidence: dict) -> float:
        """score = sum of supporter similarities + 0.5 x popularity share.

        Signal 1 (similarity_sum): each similar customer who bought the product votes with
        a weight equal to how similar they are, so many close neighbours beat one weak one.
        Signal 2 (popularity share): the share of all customers who bought it, a small
        bonus that also orders the fallback list and breaks near-ties in favour of
        products the shop knows sell.
        """
        share = evidence["popularity"] / len(self._customers)
        return evidence["similarity_sum"] + POPULARITY_WEIGHT * share

    def recommend(self, customer_id: str, n: int = 3) -> list[dict]:
        """Up to n ranked recommendations. Ties: higher popularity, then product ID."""
        candidates = self.candidate_products(customer_id, self.k_neighbours)
        ranked = sorted(
            candidates.values(),
            key=lambda e: (-self.score(e), -e["popularity"], e["product_id"]),
        )
        results = []
        for rank, evidence in enumerate(ranked[:n], start=1):
            results.append({
                "customer_id": customer_id,
                "customer_name": self._names[customer_id],
                "rank": rank,
                "product_id": evidence["product_id"],
                "product_name": evidence["name"],
                "category": evidence["category"],
                "score": round(self.score(evidence), 3),
                "supporting_customers": [c for c, _ in evidence["supporters"]],
                "fallback": evidence["fallback"],
                "popularity": evidence["popularity"],
            })
        return results

    # -- Tasks 7 and 8 -----------------------------------------------------
    def recommend_all(self, n: int = 3) -> list[dict]:
        """One list of rows for everyone. A customer with nothing to recommend still gets
        one row with blank product columns, so nobody disappears from the output."""
        rows: list[dict] = []
        for customer in self._customers:
            recs = self.recommend(customer, n)
            if recs:
                rows.extend(recs)
            else:
                rows.append({
                    "customer_id": customer, "customer_name": self._names[customer],
                    "rank": "", "product_id": "", "product_name": "", "category": "",
                    "score": "", "supporting_customers": [], "fallback": False,
                    "popularity": "",
                })
        return rows

    def export_csv(self, rows: list[dict], path: Path = OUTPUT_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_COLUMNS)
            for row in rows:
                score = row["score"]
                writer.writerow([
                    row["customer_id"], row["customer_name"], row["rank"],
                    row["product_id"], row["product_name"], row["category"],
                    "" if score == "" else "{:.3f}".format(score),
                    ";".join(row["supporting_customers"]),
                ])
