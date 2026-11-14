# Replication Package for "When the Publisher Intervenes"

Code and data for the bachelor's thesis "When the Publisher Intervenes: Price and Volatility Responses to an Exogenous Mechanic Change in the CS2 Skin Market" (Moritz Dahmen, University of St. Gallen, 2026). The data, method, and results are described in the thesis. This file only explains what the repository contains and how to reproduce the results.

## Contents

| Path | Contents |
|---|---|
| `build_item_universe.py` | Builds the item list and assigns each item to a group |
| `check_item_list.py` | Checks the item list against Valve's game files (Appendix B) |
| `steam_pricehistory_collector.py` | Downloads the price histories from the Steam Community Market |
| `config.py` | All parameters of the analysis (windows, thresholds, groups) |
| `analysis/` | Analysis scripts `00` to `10`, plus `plotstyle.py` (common figure style) and `inference.py` (pools as clusters and cluster inference for 05 to 07 and 09) |
| `items/` | Item list (`items_full.csv`), mapping of cases to their knife and glove pools (`case_pools.csv`), and smaller lists for test retrievals |
| `meta_cache/` | ByMykel CSGO-API metadata of September 19, 2026, from which the item list is built, with ByMykel's license (`LICENSE_ByMykel`) |
| `data/full/` | Raw data: price histories of all 5,275 items (`panel_long.parquet`) and the retrieval log (`manifest.jsonl`, `collector.log`; the collector of that run still wrote its messages in German) |
| `data/check/` | Second retrieval of five sticker capsules for the currency check (Appendix B) |
| `output/` | Tables, figures, and logs of the run on which the thesis is based |
| `trade_up_contracts/` | Own trade-up contracts (Appendix A, Table A2): inputs, outcomes, screenshots, `predict.py` (predictions made before the contracts), and `observed.py` (Table A2); details in its own `README.md` |

## Setup

Tested on Windows with Python 3.14.3. The package versions are listed in `requirements.txt`; `requirements-lock.txt` also lists the exact versions of their dependencies. The environment was created with uv:

```
uv venv --python 3.14.3
.venv\Scripts\activate
uv pip install -r requirements.txt
```

Without uv, use `python -m venv .venv` and `pip install -r requirements.txt`. On macOS and Linux, activate the environment with `source .venv/bin/activate`.

## Reproducing the Results

The data in the repository are sufficient; no Steam account or internet access is needed. Run the scripts in order from the project folder:

```
python analysis/00_prepare.py
python analysis/01_screening.py
python analysis/02_descriptives.py
python analysis/03_event_plots.py
python analysis/04_abnormal_returns.py
python analysis/05_cross_section.py
python analysis/06_arbitrage.py
python analysis/07_volatility.py
python analysis/08_volume.py
python analysis/09_thesis_tables.py
python analysis/10_appendix_tables.py
python trade_up_contracts/observed.py
```

Each script builds on the outputs of the previous ones. The full run takes a few minutes. Intermediate results are written to `data/clean/`, and tables, figures, and logs to `output/`. All random draws use a fixed seed, so a new run produces the same tables and figures as those in `output/`; only the times in the logs differ. `10_appendix_tables.py` also reads `output/tables/check_item_list.csv`, which `check_item_list.py` writes (see below; this script needs internet access).

| Script | What it does | Thesis |
|---|---|---|
| `00_prepare` | Cleans the raw panel and flags spikes, liquidity, and high prices | Section 4.1, Appendix B |
| `01_screening` | Selects the main sample and the sample variants | Section 4.1, Appendix B |
| `02_descriptives` | Describes the sample and the returns before the update | Section 4.2, Appendix C |
| `03_event_plots` | Price paths without a model | Section 5.1 |
| `04_abnormal_returns` | Abnormal returns and their tests | Sections 4.3, 4.4, 5.1, Appendix C |
| `05_cross_section` | Cross-sectional regressions | Section 5.2, Appendix D |
| `06_arbitrage` | Expected value of the trade-ups and the no-arbitrage bound | Sections 3.1, 5.3, Appendices A, D |
| `07_volatility` | Volatility, noise, and coupling to the pool | Section 5.4, Appendix E |
| `08_volume` | Trading volume, echo, and unit balance | Section 5.4, Appendix E |
| `09_thesis_tables` | Tables 2 to 6 | |
| `10_appendix_tables` | Appendix tables except Table A2 | |
| `trade_up_contracts/observed` | Table A2: own trade-up contracts against their predictions | Appendix A |

**Tables.** Every table of the thesis is in `output/tables/thesis_table<No>.csv`, for example `thesis_table3.csv` or `thesis_tableC4.csv`. Next to it is a `.tsv` version for pasting into Word; the table notes are in the thesis. Table 1 is not produced by the code.

**Figures** (`output/figures/`, each as PNG and SVG):

| Thesis | File |
|---|---|
| Figure 2 | `03_fig1_event_path_short` |
| Figure 3 | `05_fig4_wear_pattern` |
| Figure 4 | `06_fig1_ratio_path` |
| Figure 5 | `06_fig3_placebo` |
| Figure 6 | `07_fig1_signature` |
| Figure 7 | `08_fig4_long_run` |
| Figure B1 | `02_fig2_liquidity_distribution` |
| Figure C1 | `04_fig1_caar_path` |
| Figure D1 | `06_fig2_decomposition` |
| Figure E1 | `08_fig1_volume_path` |
| Figure E2 | `08_fig2_echo` |
| Figure E3 | `07_fig2_event_time` |
| Figure E4 | `07_fig3_coupling` |

Figure 1 is a diagram without data. The other files in `output/figures/` are additional figures that do not appear in the thesis.

Most numbers that the text reports but no table shows are in the logs in `output/logs/`. All parameters are set in `config.py`; after changing one, rerun the scripts from `00`.

## Collecting the Data Again (Optional)

The raw data come from retrievals on August 28, September 1, and September 18, 2026 (Appendix B). A new retrieval does not give exactly the same data: Steam converts the entire price history into the wallet currency at the exchange rate of the retrieval day, and the history extends further.

1. **Item list.** `python build_item_universe.py` builds the files in `items/` from `meta_cache/` and reproduces the list used in the thesis. With `--refresh`, the script instead downloads the current state of the ByMykel API, which adds the items released since.
2. **Check against the game files.** `python check_item_list.py` downloads Valve's game files (`items_game.txt`, `csgo_english.txt`, and the file list `pak01_dir.txt`) in the version fixed by `ITEM_CHECK_COMMIT` in `config.py` and writes `output/tables/check_item_list.csv` and `output/logs/check_item_list.log`. This requires internet access.
3. **Prices.** This requires a Steam account with a wallet in euros, because all thresholds of the analysis are in euros. The script checks the currency at startup and stops otherwise.
   - The script logs in with the cookie `steamLoginSecure`. Log in at steamcommunity.com, open the developer tools (F12), and copy the value of `steamLoginSecure` from the cookies of `https://steamcommunity.com` under "Application" (Chrome, Edge) or "Storage" (Firefox). It begins with the Steam ID (`7656119...`).
   - Copy `.env.example` to `.env` and enter the value after `STEAM_LOGIN_SECURE=`. The cookie gives access to the Steam account, so do not share `.env`; `.gitignore` excludes it.
   - Write the retrieval to a new folder so that the data of the thesis in `data/full/` are kept:

     ```
     python steam_pricehistory_collector.py --items items/items_full.csv --outdir data/new --min-delay 4 --max-delay 6
     ```

   - With 5,312 items and a pause of 4 to 6 seconds between requests, this takes a little over seven hours, plus waiting time when Steam throttles the requests. Each item is saved immediately, and an interrupted run resumes where it stopped. If the cookie expires, the script stops; enter a new cookie and restart.
   - `--limit 150` retrieves only the first 150 items, for testing. `--help` lists all options.
   - To analyze the new data, point `RAW_PANEL` and `MANIFEST` in `config.py` to the new folder. To repeat the currency check of Appendix B, retrieve `items/items_currency_check.csv` a few days later into another folder and point `CHECK_PANEL` and `CHECK_MANIFEST` to it.

## Data Sources

- Prices: Steam Community Market, `pricehistory` endpoint (CS2, app ID 730)
- Item list: ByMykel, CSGO-API, https://github.com/ByMykel/CSGO-API (MIT License, see `meta_cache/LICENSE_ByMykel`)
- Game files for the check: SteamDatabase, GameTracking-CS2, https://github.com/SteamDatabase/GameTracking-CS2

## License

The code in this repository is licensed under the MIT License (`LICENSE`). The license does not cover the data: the prices come from the Steam Community Market, the item metadata from ByMykel (MIT License, see `meta_cache/LICENSE_ByMykel`), and the screenshots show Counter-Strike 2 and Steam. Counter-Strike 2, Steam, and the game content belong to Valve Corporation.
