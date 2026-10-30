# Day 4: Customer Recommendation Engine (Mama Mboga Online)

## Approach
Mama Mboga Online wants a "customers like you also bought" suggestion at checkout that a shop owner can understand. The engine is written in core Python only (`csv`, `collections`, `datetime`, `pathlib`) and works in four steps:

1. **Read and clean** the October 2026 order export (`read_purchases`), keeping every rejected row with a reason.
2. **Represent** each customer as the *set* of products they have bought (a dict of customer to set).
3. **Compare** customers with Jaccard similarity: shared products divided by all products either customer bought.
4. **Recommend** products that the most similar customers bought and this customer has not, ranked by a simple scoring rule (below).

The engine builds all its lookups once in `RecommendationEngine.__init__`: customer histories, product names and categories, and product popularity. No later method re-scans the raw rows.

## How to run
```
python main.py                 # cleans the data, prints two customers, writes output/recommendations.csv
python -m unittest -v          # runs the 20 tests
```
Run both from the project folder (the one that contains `data/`). No packages are needed (`requirements.txt` is empty on purpose). The whole workflow is `RecommendationEngine(purchases).recommend_all(n=3)`.

## Files
```
day4-recommendation-engine/
├── recommender.py          read_purchases, jaccard, RecommendationEngine
├── main.py                 runs everything end to end and prints the run summary
├── data/purchases.csv      the raw order export, untouched
├── output/recommendations.csv
├── tests/test_recommender.py
├── requirements.txt
└── README.md
```

## Data-cleaning rules
The file has 63 purchase rows: 60 are used and 3 are rejected.

| Case | Rule | In this file |
|---|---|---|
| Repeated row | A repeated `purchase_id` is rejected as a duplicate. The first copy is kept. | T9010 (Kiprop's eggs, listed twice) |
| Blank product | A row with a blank `product_id` or `product_name` cannot be recommended from, so it is rejected. | T9020 (Njeri) |
| Zero quantity | `quantity` must be a whole number above zero. Zero means nothing was delivered, so it is not a purchase. | T9033 (Nekesa, Omo 2kg) |
| Inconsistent ID | IDs are trimmed and upper-cased. The row is repaired and kept, not rejected. | T9041 (`c106` becomes `C106`) |
| Bad date | A date that is not a real `YYYY-MM-DD` is rejected. | none in this file |
| Missing file or columns | A clear `FileNotFoundError` or `ValueError`. | none |

A repeated *product* by the same customer on different days is a real second purchase, but a history set only records that the customer bought it, so it counts once. Quantity does not change the set.

## Similarity
`jaccard(a, b) = |A ∩ B| / |A ∪ B|`, so it is always between 0 and 1, and a customer compared with themselves gives 1.0. **Two empty sets return 0.0** (they describe no one, and it avoids dividing by zero); this is tested.

`similar_customers(customer, k)` returns the top-k others, best first. It excludes the customer, anyone below `min_similarity`, and anyone with similarity 0, because no shared product means no evidence. **Ties are broken by customer ID, ascending**, so the result is identical on every run.

The similarity matrix printed by `main.py` for C101 to C105 shows 1.00 on the diagonal and 0.00 elsewhere. That is correct and not a bug: these five customers genuinely share no product (for example C101 bought only P04 and P08, C103 only P10 and P11).

## Recommendations
`candidate_products(customer, k)` collects products that the k nearest neighbours bought and the customer has not, and records which neighbours support each one. The engine uses `k_neighbours = 5` by default. Five rather than three because with only about 3 products per customer, three neighbours often add nothing new.

**Fallback.** If the neighbours add nothing (no overlap with anyone, or the neighbours have only bought things the customer already owns), the engine offers the most popular products the customer has not bought, with no supporting customers. This is the safe, explainable default for a shop: "most customers buy this". In this data, only C107 (Chebet) needs it. Her three closest neighbours (C111, C119, C103) have only bought products she already has. The console says "because it is popular" for these rows. A customer who has already bought all 15 products gets one row with blank product columns (tested).

A customer can receive fewer than 3 recommendations when the neighbours simply do not know more products they lack (C108, C111, C119 and C120 get 2). The engine does not pad the list with weak guesses.

## Ranking rule
```
score = (sum of the similarities of the neighbours who bought it)
        + 0.5 x (customers who bought it / all customers)
```
Two signals are combined:
- **Similarity-weighted votes.** Every neighbour who bought the product votes with a weight equal to how similar they are to this customer. Three close neighbours beat one distant one.
- **Popularity share.** A small bonus (weight 0.5) for products many customers buy. Popularity counts different customers, not units, so one large basket cannot inflate it. It also orders the fallback list and nudges close calls towards proven sellers.

Ties are broken by higher popularity, then product ID, so the order is deterministic.

### Worked example: Mwangi (C106)
Mwangi bought Pishori Rice (P06), Kimbo (P05) and Kenchic Whole Chicken (P15). Her nearest neighbours are C118 (0.50), C102 (0.25), C105 (0.25), C114 (0.20) and C107 (0.14).

Take **Kienyeji Eggs (P01)**, which she has not bought:
- Bought by three of her neighbours: C118 (similarity 0.50), C102 (0.25) and C114 (0.20). Similarity sum = 0.50 + 0.25 + 0.20 = **0.95**.
- Five of the 20 customers bought eggs, so popularity share = 5 / 20 = 0.25, and the bonus is 0.5 × 0.25 = **0.125**.
- **score = 0.95 + 0.125 = 1.075**, which ranks first.

Brookside Milk (P02) scores 0.518 (supported by C105 and C107) and Pampers (P10) scores 0.293 (supported by C107), so they rank second and third. The eggs win because three similar customers vouch for them, not because they are popular. Check C106's console output when you run `main.py`.

C118 vs C106 for reference: C106 = {P05, P06, P15}, C118 = {P01, P05, P06}. Shared = 2, all products = 4, so 2 / 4 = 0.50.

## Output
`output/recommendations.csv` has one row per customer and rank, with the columns `customer_id, customer_name, rank, product_id, product_name, category, score, supporting_customers` (score to 3 decimal places, supporters joined with `;`). All 20 customers appear. The run summary:
```
Purchases read: 63 | used: 60 | rejected: 3
Customers: 20 | Products: 15 | Recommendations written: 56
```

## Error handling
An unknown customer ID raises `UnknownCustomerError` (a `KeyError` subclass with the message "Unknown customer ID: C999") from `history`, `similarity`, `similar_customers`, `candidate_products`, `recommend` and `customer_name`. It never returns an empty or made-up answer. `history()` returns a copy, so callers cannot change the engine's data by accident.

## Assumptions
- A customer's identity is the upper-cased, trimmed `customer_id`. The name comes from the latest valid row.
- A purchase means the customer received the product, so quantity must be above zero.
- Only whether a customer bought a product matters for similarity, not how much or when. Dates are validated but not used in the score.
- With this small data set (about 3 products per customer), similarity values are coarse and some neighbours are weak. The rule shows its reasoning for each recommendation so the team can judge it. More history would make the neighbours more reliable.
