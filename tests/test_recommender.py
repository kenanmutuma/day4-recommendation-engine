"""Run from the project folder with:  python -m unittest -v"""
import csv
import tempfile
import unittest
from pathlib import Path

from recommender import (
    CSV_COLUMNS, DATA_PATH, RecommendationEngine, UnknownCustomerError,
    jaccard, read_purchases,
)

HEADER = "purchase_id,customer_id,customer_name,product_id,product_name,category,purchase_date,quantity\n"


def make_rows(baskets):
    """baskets: {customer: [products]} -> purchase dicts the engine accepts."""
    rows = []
    for customer, products in baskets.items():
        for product in products:
            rows.append({
                "purchase_id": customer + product, "customer_id": customer,
                "customer_name": "Name " + customer, "product_id": product,
                "product_name": "Product " + product, "category": "Cat",
                "purchase_date": "2026-10-01", "quantity": 1,
            })
    return rows


class TestJaccard(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(jaccard({1, 2, 3}, {2, 3, 4}), 0.5)
        self.assertEqual(jaccard({1}, {1}), 1.0)
        self.assertEqual(jaccard({1}, {2}), 0.0)

    def test_both_empty_is_zero_not_an_error(self):
        self.assertEqual(jaccard(set(), set()), 0.0)

    def test_one_empty_is_zero(self):
        self.assertEqual(jaccard({1}, set()), 0.0)

    def test_symmetric(self):
        self.assertEqual(jaccard({1, 2}, {2, 3, 4}), jaccard({2, 3, 4}, {1, 2}))


class TestReadPurchases(unittest.TestCase):
    def read(self, body):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "p.csv"
            path.write_text(HEADER + body, encoding="utf-8")
            return read_purchases(path)

    def test_rules(self):
        valid, rejected = self.read(
            "T1,C1,A,P1,Milk,Dairy,2026-10-01,2\n"
            "T1,C1,A,P1,Milk,Dairy,2026-10-01,2\n"        # repeated purchase_id
            "T2,C1,A,,,Dairy,2026-10-01,1\n"              # blank product
            "T3,C1,A,P2,Eggs,Fresh,2026-10-01,0\n"        # zero quantity
            "T4,C1,A,P2,Eggs,Fresh,2026-10-01,abc\n"      # not a number
            "T5,C1,A,P2,Eggs,Fresh,2026-13-45,1\n"        # impossible date
            "T6,c2, B ,p3,Tea,Beverages,2026-10-02,1\n"   # messy ID, repaired
        )
        self.assertEqual([r["purchase_id"] for r in valid], ["T1", "T6"])
        self.assertEqual(valid[1]["customer_id"], "C2")
        self.assertEqual(valid[1]["product_id"], "P3")
        self.assertEqual(valid[1]["customer_name"], "B")
        reasons = [r["reason"] for r in rejected]
        self.assertEqual(len(rejected), 5)
        self.assertIn("duplicate purchase_id", reasons)
        self.assertIn("blank product", reasons)
        self.assertIn("invalid purchase_date", reasons)

    def test_missing_file_and_columns(self):
        with self.assertRaises(FileNotFoundError):
            read_purchases(Path("no_such_file.csv"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            path.write_text("a,b\n1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_purchases(path)


class TestEngine(unittest.TestCase):
    def setUp(self):
        self.engine = RecommendationEngine(make_rows({
            "C1": ["P1", "P2"],
            "C2": ["P1", "P2", "P3"],
            "C3": ["P1", "P4"],
            "C4": ["P9"],               # shares nothing with anyone
            "C5": ["P1", "P2", "P3", "P4", "P9"],  # has bought everything
        }))

    def test_similarity_matrix_properties(self):
        for a in self.engine.customers:
            self.assertEqual(self.engine.similarity(a, a), 1.0)
            for b in self.engine.customers:
                value = self.engine.similarity(a, b)
                self.assertTrue(0.0 <= value <= 1.0)
                self.assertEqual(value, self.engine.similarity(b, a))

    def test_similar_customers_excludes_self_and_zero_overlap(self):
        ids = [c for c, _ in self.engine.similar_customers("C1", 10)]
        self.assertNotIn("C1", ids)
        self.assertNotIn("C4", ids)
        self.assertEqual(ids[0], "C2")  # 2/3 beats the others

    def test_min_similarity_filters(self):
        strict = RecommendationEngine(make_rows({"C1": ["P1", "P2"], "C2": ["P1", "P2", "P3"],
                                                 "C3": ["P1", "P4"]}), min_similarity=0.6)
        self.assertEqual([c for c, _ in strict.similar_customers("C1", 5)], ["C2"])

    def test_ties_are_broken_by_customer_id(self):
        engine = RecommendationEngine(make_rows({"C1": ["P1"], "C3": ["P1", "P2"], "C2": ["P1", "P3"]}))
        self.assertEqual([c for c, _ in engine.similar_customers("C1", 5)], ["C2", "C3"])

    def test_recommendations_are_new_products_only(self):
        recs = self.engine.recommend("C1", 5)
        owned = self.engine.history("C1")
        self.assertTrue(recs)
        for rec in recs:
            self.assertNotIn(rec["product_id"], owned)

    def test_fallback_for_customer_with_no_neighbours(self):
        engine = RecommendationEngine(make_rows({"C1": ["P1", "P2"], "C2": ["P1"], "C3": ["P7"]}))
        self.assertEqual(engine.similar_customers("C3", 5), [])  # shares nothing with anyone
        recs = engine.recommend("C3", 2)
        self.assertEqual(len(recs), 2)
        self.assertTrue(all(r["fallback"] for r in recs))
        self.assertEqual(recs[0]["product_id"], "P1")  # most popular product first

    def test_customer_who_bought_everything_gets_a_blank_row(self):
        self.assertEqual(self.engine.recommend("C5", 3), [])
        rows = [r for r in self.engine.recommend_all(3) if r["customer_id"] == "C5"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["product_id"], "")

    def test_unknown_customer_raises_clear_error(self):
        for call in (self.engine.history, self.engine.recommend, self.engine.customer_name,
                     self.engine.similar_customers, self.engine.candidate_products):
            with self.assertRaises(UnknownCustomerError):
                call("C999")
        self.assertIsInstance(UnknownCustomerError("C999"), KeyError)
        self.assertIn("C999", str(UnknownCustomerError("C999")))

    def test_history_returns_a_copy(self):
        self.engine.history("C1").add("P99")
        self.assertNotIn("P99", self.engine.history("C1"))

    def test_score_combines_similarity_and_popularity(self):
        base = {"similarity_sum": 0.5, "popularity": 0}
        more_popular = {"similarity_sum": 0.5, "popularity": 5}
        self.assertLess(self.engine.score(base), self.engine.score(more_popular))
        self.assertEqual(self.engine.score(base), 0.5)

    def test_same_output_on_every_run(self):
        self.assertEqual(self.engine.recommend_all(3), self.engine.recommend_all(3))

    def test_export_columns_and_every_customer_present(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out" / "recs.csv"  # folder is created for us
            self.engine.export_csv(self.engine.recommend_all(3), path)
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertEqual(tuple(reader.fieldnames), CSV_COLUMNS)
        self.assertEqual({r["customer_id"] for r in rows}, set(self.engine.customers))


@unittest.skipUnless(DATA_PATH.exists(), "run from the project folder")
class TestRealData(unittest.TestCase):
    def test_counts(self):
        valid, rejected = read_purchases(DATA_PATH)
        self.assertEqual((len(valid) + len(rejected), len(valid), len(rejected)), (63, 60, 3))
        self.assertEqual({r["purchase_id"] for r in rejected}, {"T9010", "T9020", "T9033"})
        engine = RecommendationEngine(valid)
        self.assertEqual((len(engine.customers), len(engine.products)), (20, 15))
        self.assertIn("C106", engine.customers)          # 'c106' was repaired
        self.assertEqual(len(engine.history("C106")), 3)  # P05, P06, P15

    def test_worked_example_in_readme(self):
        valid, _ = read_purchases(DATA_PATH)
        engine = RecommendationEngine(valid)
        top = engine.recommend("C106", 1)[0]
        self.assertEqual(top["product_id"], "P01")
        self.assertEqual(top["score"], 1.075)
        self.assertEqual(top["supporting_customers"], ["C118", "C102", "C114"])


if __name__ == "__main__":
    unittest.main()
