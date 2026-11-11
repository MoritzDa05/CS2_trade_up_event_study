# Own trade-up contracts (Appendix A, Table A2)

Two trade-up contracts of ten Mil-Spec items each, used to test the float rule of Equation (1) and the trade hold of a trade-up output. The inputs and the predictions were committed before either contract was executed.

**Files**
- `inputs.csv`: the inputs of contracts A and B, plus one StatTrak item for the test on the contract screen. Float, pattern, and "Tradable after" date as shown in the Steam inventory; the screenshots are in `screenshots/inputs/`. `item_de` is the name shown by the German client.
- `predict.py`: reads `inputs.csv`, takes the float range, collection, and rarity of each skin from `meta_cache/skins.json`, and writes `predictions.csv`. Before predicting, it checks that each contract has ten inputs of the same rarity, does not mix StatTrak and regular items, and that every float lies in the range of its skin and matches the wear level. Run from the project folder: `python trade_up_contracts/predict.py`.
- `predictions.csv`: for every skin of the next rarity in the collections of the inputs,
  - `float_normalized`: Equation (1). Each input float is normalized to the range of its own skin, the ten positions are averaged to m, and the output float is f_min + m (f_max − f_min) of the output skin;
  - `float_raw`: the raw average of the ten input floats instead of m;
  - `probability`: number of inputs from the collection / 10 / number of output skins in the collection. Reported for completeness; two contracts do not test it.

The float ranges in `skins.json` agree with Valve's `items_game.txt` (commit `39df8b2a`, as in `config.ITEM_CHECK_COMMIT`) for all inputs and outputs.

**Hit rule:** a rule is confirmed if the observed output float differs from its prediction by less than 1e-6. The game stores floats as 32-bit numbers, so deviations of the order of 1e-8 are expected. In contract A the two rules differ by at least 0.03 for every possible output, in contract B by at least 0.06, so at most one of them can hit.

**Trade hold:** the inputs of contract B were bought on November 10, 2026 (tradable after November 17, 17:00), those of contract A on November 11 (tradable after November 18, 19:00; one input 22:00). If the output inherits the hold of its inputs, it shows the date of its inputs (for contract A, the latest of them). If a trade-up starts a new hold, the output shows a date seven days after the contract. If it sets no hold, it is tradable at once.

**Contract screen:** a StatTrak Mil-Spec (`stattrak_ump_45_roadblock_910.png`) cannot be added next to regular items, and an eleventh item cannot be added.
