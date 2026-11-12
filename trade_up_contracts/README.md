# Own trade-up contracts (Appendix A, Table A2)

Two trade-up contracts of ten Mil-Spec items each, used to test the float rule of Equation (1) and the trade hold of a trade-up output. The inputs and the predictions were committed before either contract was executed.

**Files**
- `inputs.csv`: the inputs of contracts A and B, plus one StatTrak item for the test on the contract screen. Float, pattern, and "Tradable after" date as shown in the Steam inventory; `item_de` is the name shown by the German client.
- `predict.py`: reads `inputs.csv`, takes the float range, collection, and rarity of each skin from `meta_cache/skins.json`, and writes `predictions.csv`. Before predicting, it checks that each contract has ten inputs of the same rarity, does not mix StatTrak and regular items, and that every float lies in the range of its skin and matches the wear level. Run before the contracts: `python trade_up_contracts/predict.py`.
- `predictions.csv`: for every skin of the next rarity in the collections of the inputs,
  - `float_normalized`: Equation (1). Each input float is normalized to the range of its own skin, the ten positions are averaged to m, and the output float is f_min + m (f_max − f_min) of the output skin;
  - `float_raw`: the raw average of the ten input floats instead of m;
  - `probability`: number of inputs from the collection / 10 / number of output skins in the collection. Reported for completeness; two contracts do not test it.
- `observed.csv`: the outcome of each contract (time of the contract, output, wear, float, pattern, "Tradable after" date).
- `observed.py`: compares `observed.csv` with `predictions.csv` and writes Table A2 to `output/tables/thesis_tableA2.csv` and `.tsv`. Run after the contracts: `python trade_up_contracts/observed.py`.
- `screenshots/`: `inputs/` (every input in the Steam inventory), `contract_a/` and `contract_b/` (contract screen, result, output with float and "Tradable after" date, Steam inventory history), `wear_levels/` (Steam Community Market filtered by wear level and sorted by float).

The float ranges in `skins.json` agree with Valve's `items_game.txt` (commit `39df8b2a`, as in `config.ITEM_CHECK_COMMIT`) for all inputs and outputs.

**Hit rule:** a rule is confirmed if the observed output float differs from its prediction by less than 1e-6. The game stores floats as 32-bit numbers, so deviations of the order of 1e-8 are expected. In contract A the two rules differ by at least 0.03 for every possible output, in contract B by at least 0.06, so at most one of them can hit.

**Trade hold:** the inputs of contract B were bought on November 10, 2026 (tradable after November 17, 17:00), those of contract A on November 11 (tradable after November 18, 19:00; one input 22:00). If the output inherits the hold of its inputs, it shows the date of its inputs (for contract A, the latest of them). If a trade-up starts a new hold, the output shows a date seven days after the contract. If it sets no hold, it is tradable at once.

**Contract screen:** a StatTrak Mil-Spec (`stattrak_ump_45_roadblock_910.png`) cannot be added next to regular items, and an eleventh item cannot be added.

**Results** (Table A2)
- Contract B, executed on November 11, 2026, 22:06: Desert Eagle | Trigger Discipline with float 0.509881258. Equation (1) predicted 0.509881232 (difference 2.6e-8), the raw average 0.363832370. The output is tradable after November 17, 17:00, the date of its inputs.
- Contract A, executed on November 12, 2026, 19:32: M4A4 | Etch Lord with float 0.507800162. Equation (1) predicted 0.507800141 (difference 2.1e-8), the raw average 0.432282128. The output is tradable after November 18, 22:00, the latest date of its inputs.
- Contract screen: once the StatTrak item is placed, only StatTrak items remain eligible; with ten items placed, no further item is eligible (`contract_b/01_stattrak_placed.png`, `contract_b/03_ten_items.png`).
- Wear levels: the market filter assigns 0 to 0.07, 0.07 to 0.15, 0.15 to 0.38, 0.38 to 0.45, and 0.45 to 1. The highest floats listed in the first four are 0.069997996, 0.149844512, 0.379595071, and 0.449945658.
