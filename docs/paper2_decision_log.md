# HADES Paper 2: decision log

Every modelling decision for Paper 2, with its reason, its source and the commit that implemented it. Newest entries at the bottom of each section. Entries are added when a decision is made, not reconstructed later.

Each entry gives: **Decision** (what was chosen), **Why** (the reason and the evidence), **Source** (papers, datasets or tests behind it), **Code** (where it lives and the commit), **Paper 1** (whether Paper 1 results change), and **Methods text** where a sentence for the paper exists.

Branches: `main` and tag `waste-management-submission` hold the submitted Paper 1 code and are not modified. All Paper 2 work is on `paper2-dev`.

---

## A. Repository and workflow

### A1. Paper 1 code frozen, Paper 2 on its own branch (2026-11-20)
**Decision.** Paper 1 stays on `main` and the tag `waste-management-submission`; Paper 2 is developed on `paper2-dev`. `case_study.ipynb` is the Paper 1 case study and is not modified. The Streamlit app gets the Paper 2 options on `paper2-dev` only.
**Why.** Reviewers of Paper 1 must be able to reproduce the submitted results.
**Code.** Tag `waste-management-submission`; `main` last commit 5f1743f.
**Paper 1.** Unchanged.

### A2. One notebook for single runs, a separate one for sweeps (2026-11-22)
**Decision.** `notebooks/HADES.ipynb` carries every Paper 2 option for a practitioner running one or two scenarios. Large simulation campaigns get a separate sweep notebook.
**Code.** 12b32e0.

### A3. Decision log kept by request, no CLAUDE.md (2026-11-22)
**Decision.** This log is updated with every decision, in the same commit as the code change. No instruction file is added to the repository; the user asks for the log to be kept updated at the start of each session.

---

## B. Coding traps fixed before sweeps

### B1. Single source for T, RT and feed; input checks; safe LCA cache (2026-11-21)
**Decision.** T and RT are defined once. `build_system` checks its inputs, including T within 160 to 280 °C (range of the PW-ADM1 S_I composition). LCA unit scores are cached with the project and database "modified" stamps, so a changed background never returns a stale score.
**Why.** In the Paper 1 OAT, the T −20 % case ran at 144 °C: S_I was clipped to its 160 °C values and gas yield clamped to 0 without any error.
**Code.** 187d66c.
**Paper 1.** Unchanged (frozen regression passes).

### B2. Every binding cap is recorded (2026-11-21)
**Decision.** Each cap or clamp that binds in a run is written to `caps_active` and `qa_flags`, with its kind (correlation or design), the raw value and the capped value.
**Why.** A capped value must never replace a computed one silently.
**Code.** 31d9d8c.

### B3. O by difference kept; unused `perturb_feed` removed (2026-11-21)
**Decision.** Oxygen stays computed by difference (C + 20 % means O down). No dry-ash-free O option.
**Why.** The user's decision; sweeps must state that varying C, H, N, S or ash changes O.
**Code.** 67b124c.

---

## C. Structural points (ranked by bias on a feedstock guidebook)

### C0. HTC surrogate generalization: deferred (2026-11-21)
**Decision.** Not changed yet; to be discussed with supervisors.
**Why.** Random 5-fold CV mixes rows of the same study across folds. Grouped by study (GroupKFold on `Ref`, same hyperparameters, 938 rows), R² drops: yield 0.866 → 0.275, HHV 0.912 → 0.584, C_char 0.944 → 0.755. Single deterministic split, so indicative.
**Source.** Roberts et al. (2017) Ecography 40:913 (grouped CV); Chen & Guestrin (2016) (monotone constraints); Funke & Ziegler (2010) Biofuels Bioprod. Bioref. 4:160; Overend & Chornet (1987) Phil. Trans. R. Soc. A 321:523 (severity factor); Sahigara et al. (2012) Molecules 17:4791 (applicability domain).
**Open.** Comparison of four variants (random vs grouped CV, grouped retune, monotone constraints, severity factor) offered for the supervisor meeting.

### C1. Pathway B digester feed: feedstock-specific degradability (2026-11-21)
**Decision.** Each feedstock gets its biodegradable fraction fD (X_I = 1 − fD, S_I = 0), the soluble share of the degradable COD, and k_hyd, each with a range. Replaces X_I 0.08, S_I 0.02, soluble 0.27, k_hyd 10 d⁻¹ (BSM2) for every feedstock.
**Why.** With the old inputs, manure, straw and sludge digested like food waste (99.5 % hydrolysis at 20 d HRT). Against 13 independent studies the old inputs passed 3; per-feedstock values passed 10.
**Source.** Fisgativa et al. (2020) Data in Brief 29:105212 (Tables 3, 5); Labatut, Angenent & Scott (2022) Environ. Sci.: Water Res. Technol. 8:1507, ESI S7 and S8; Jimenez et al. (2020) Waste Manag. 101:150; Trzcinski & Stuckey (2012) Environ. Eng. Sci. 29:848; Zerback et al. (2022) Processes 10:1048; Sotemann et al. (2005) Water SA 31(4); Hoover (INL, 2016). Values and ranges in `ADM1.PATHWAY_B_FEED_PARAMS`.
**Code.** f9857fc, 671eed3. Overrides for sweeps and practitioners: `ctx.b_feed_overrides` (12b32e0).
**Paper 1.** Pathway A unchanged. Pathway B base case (food waste) changes: CH4 −13.1 %, hydrochar +20.7 %; frozen B values updated on purpose.

### C2. Digestate composition from ADM1: not done (2026-11-21)
**Decision.** Keep the measured digestate coefficients (H/C, O/C, FC).
**Why.** The user's earlier attempts to derive a clean digestate from ADM1 output failed.

### C3. Pathway A process-water methane: multiplier (2026-11-21)
**Decision.** One multiplier m on the methane (and CO2) leaving the process-water digester, sampled 0.55 to 1.45, 1.0 by default; methane capped at 350 mL CH4 per g of process-water COD fed.
**Why.** A per-feedstock S_I correlation (9 HTC studies, 31 points, leave-one-study-out) predicted worse than the calibrated temperature law (mean error 87 vs 66 mL CH4/g COD). Study-level measured BMP / PW-ADM1 ranges from about 0.58 (Mannarino 2022) to 1.44 (Aragón-Briceño 2020), rounded outward. Applied to methane, not to S_I, because at 180 °C S_I = 0.25 and m > 1.33 on S_I would make it negative.
**Limitation.** Marin-Batista 2020 at 240 °C shows complete inhibition, outside any multiplier.
**Code.** 7c21c38. Notebook setting `PW_CH4_MULTIPLIER` (12b32e0).
**Paper 1.** Unchanged at m = 1.

### C4. Gas yield: unchanged (2026-11-21)
**Decision.** Kept as is; small effect on the LCA.

### C5. LCA context: attributional (2026-11-21)
**Decision.** Attributional framing: average mixes for supply and for the displaced products.
**Why.** The user's decision; consequential (marginal) framing not used.

---

## D. LCA context (point C5)

### D1. Heat supply per consumer (2026-11-21)
**Decision.** The HTC reactor, the dryer and the digester each take `"electricity"`, an ecoinvent heat dataset (delivered MJ, no extra efficiency) or `"own_biogas"`. Own biogas is raw biogas diverted before PSA upgrading, so exported biomethane shrinks. Allocation order: digester, then HTC, then dryer; any shortfall from a backup (default natural gas heat).
**Why.** Electric reactor heating is a design assumption that dominates outside Quebec's hydro grid; real plants on a fossil grid would not heat the HTC electrically.
**Result.** Base case: A on own biogas burns all 260 MJ/t of its methane and still needs 128 MJ (HTC) and 162 MJ (dryer) from backup; B covers its full 371 MJ/t and still exports 23 Nm³ biomethane (−30 %).
**Limitation.** Ecoinvent heat datasets carry no temperature level; HTC needs 180 to 260 °C.
**Code.** ed5dd10. Notebook settings `HEAT_SUPPLY`, `HEAT_BACKUP` (12b32e0).
**Paper 1.** Unchanged by default.

### D2. Biomethane displaces heat only (2026-11-22)
**Decision.** No grid injection or vehicle-fuel end use.
**Why.** In attributional substitution, injected gas still ends as heat or fuel: it either duplicates "heat" or needs a defined end use with vehicle datasets.

### D3. Regions: Canadian provinces only (2026-11-22)
**Decision.** Presets for the 10 provinces and 3 territories (`ADM1/lca_contexts.py`). `QC_PAPER1` reproduces Paper 1; `QC` is Quebec with the Paper 2 data.
**Code.** 1a7fdfd.

### D4. Paper 1 Quebec heat mix: source confirmed (2026-11-22)
**Decision.** The Paper 1 mix 50.6 / 29.5 / 14.8 / 4.2 / 0.8 is kept for Paper 1 and cited to Graphique 25.
**Why.** EEQ 2026, Graphique 25 (p. 36), industry bar: 7 / 52 / 24 / 14 / 2 / 0.4 %; dropping electricity (52 %) and renormalizing gives exactly the Paper 1 split. An earlier claim that the split did not trace to a published table looked at Graphique 32 and was wrong.
**Caveats.** Normalized from rounded percentages (coal and coke at "2 %" could be 3.2 to 5.3 % of the mix). The 0.8 % is LGN (propane and butane), not LNG; Paper 1 maps it to the natural gas market.
**Source.** Whitmore, J. and Pineau, P.-O. (2026) État de l'énergie au Québec 2026, HEC Montréal, Graphique 25.

### D5. Paper 2 heat mix: NRCan 2023 for every province (2026-11-22)
**Decision.** Industrial combustion heat mix per region from NRCan CEUD, Industrial Sector, Aggregated Industries, Table 1, 2023, stored verbatim in `ADM1/data/nrcan_ceud_2023_industrial_table1.csv`. Electricity and "Other" (steam, cement waste fuels) excluded, then renormalized. Groups: natural gas; bioenergy (wood waste and pulping liquor); petroleum (heavy fuel oil, diesel, light fuel oil and kerosene); coal and coke; LPG and gas-plant NGL (to the propane furnace). Splits inside petroleum and coal/coke come from each region's PJ. Atlantic provinces share the ATL mix, BC and the Territories the BC_TER mix. Canada = sum of the seven regions (2951.6 PJ).
**Why.** One source and one method for all provinces, Quebec included; every share traceable to a published table.
**Check.** Table rows add to the printed totals within 0.3 PJ; with refinery fuels included, the shares reproduce the independent earlier computation exactly (QC 48.9 / 24.5 / 21.8 / 3.9 / 0.9).
**Code.** e17f029, 6ef9a7d.
**Methods text.** "Industrial combustion heat mixes from NRCan's Comprehensive Energy Use Database (OEE), Industrial Sector, Aggregated Industries, Table 1, 2023; electricity, other fuels (steam, cement waste fuels), still gas and petroleum coke excluded and the remainder renormalized."

### D6. Still gas and petroleum coke excluded (2026-11-22)
**Decision.** Excluded from every regional mix (switch `hce.INCLUDE_REFINERY_FUELS`).
**Why.** Mostly refinery and upgrader by-product fuels burned where they are produced; hydrochar does not displace them. Petroleum coke bought by cement kilns is a counter-case that Table 1 cannot separate.
**Effect.** Quebec 48.9 → 54.3 % natural gas.

### D7. Dataset locations and lookups (2026-11-22)
**Decision.** Newfoundland and Labrador is CA-NF and Yukon CA-YK (ecoinvent codes). A lookup with no dataset in the priority list now stops with an error. The digester natural gas heat is the exact boiler used in Quebec ("condensing modulating >100kW") in every province.
**Why.** With CA-NL, Newfoundland's grid silently resolved to AE (United Arab Emirates). Outside Quebec, a name-prefix query picked a different, smaller boiler.
**Code.** 9ed4b90, aba5fbe.

### D8. What is regionalized outside Quebec (2026-11-22)
**Finding.** From `ADM1/check_lca_contexts.py` on ecoinvent 3.12 cut-off: provincial datasets exist for grid electricity (all 13) and the wood chips CHP (ON, AB, BC, NS, PE). Natural gas, fuel oil, propane and coal heat, tap water and wastewater are CA-QC in Quebec and RoW elsewhere; no CA-level versions.
**Methods text.** "Regional variation is represented through provincial grid electricity and provincial industrial fuel shares; fuel-specific heat datasets are Quebec-specific (CA-QC) or global (RoW) elsewhere, as ecoinvent 3.12 provides no other Canadian versions."

### D9. Natural gas heat credit: boilers and furnaces, not the RoW market (2026-11-22)
**Decision.** Province presets represent natural gas heat (displaced by biomethane, and the natural gas share of the hydrochar heat mix) by the four boiler and furnace technologies of the ecoinvent 3.12 CA-QC market, with its weights (boiler modulating 0.2532, industrial furnace 0.2408, furnace low-NOx 0.2370, boiler condensing modulating 0.2690), from each province's location list. `QC_PAPER1` keeps the market.
**Why.** The RoW market is about 94 % combined heat and power (Russian conventional plants 36 %, RoW conventional 24 %; boilers and furnaces 6.4 %), 0.0377 vs 0.0690 kg CO2 eq/MJ for CA-QC. In the Ontario test run it halved the biomethane credit (B: −81.7 → −44.6 kg CO2 eq/t, 37 of the 41 kg lost from QC to ON) while the RoW boiler costs 1.06 times the Quebec boiler.
**Code.** e0eeb15.
**Methods text.** "Natural gas heat displaced by biomethane and by hydrochar was represented by industrial boilers and furnaces, using the technology split of the ecoinvent 3.12 CA-QC heat market, in every province. The RoW market available outside Quebec is dominated by combined heat and power with allocated burdens (0.038 vs 0.069 kg CO2 eq/MJ) and does not represent displaced industrial heat."
**Open.** To confirm with supervisors (methods choice).

### D10. First regional run after D9: Ontario vs Quebec (2026-11-22)
**Result.** Food waste base case, `HADES.ipynb`, GWP100 (IMPACT World+ climate change short term), kg CO2 eq/t wet feedstock:

| Term | A, QC_PAPER1 | A, ON | B, QC_PAPER1 | B, ON |
|---|---|---|---|---|
| Total | −81.5 | −107.2 | −86.3 | −100.5 |
| HTC and drying electricity | 4.1 | 18.6 | 1.6 | 7.0 |
| Credit, industrial heat mix | −77.9 | −121.5 | −33.3 | −51.9 |
| Credit, natural gas heat | −17.4 | −18.6 | −81.7 | −87.5 |

**Reading.** After D9 the natural gas credit is 1.07 times Quebec's in both pathways, consistent with the RoW boiler (1.06 times the CA-QC boiler); the artifact is gone. The hydrochar credit per MJ rises from 0.057 to 0.089 kg CO2 eq/MJ with Ontario's mix (60 % natural gas, 21 % coal and coke). The ranking flips: B ahead of A by 4.8 in QC_PAPER1, A ahead of B by 6.7 in Ontario. Both gaps are 5 to 7 % of the totals, so the ranking is close in both provinces. Ecosystem quality for B in Ontario is −198 PDF·m²·yr (the +25 before D9 was the artifact).
**Caveat.** QC_PAPER1 vs ON also changes the heat mix source (EEQ vs NRCan) and the natural gas credit datasets; the clean regional comparison is `QC` vs `ON`.

### D11. Same-method comparison: Quebec (Paper 2) vs Ontario (2026-11-22)
**Result.** Food waste base case, GWP100, kg CO2 eq/t wet feedstock, both on the Paper 2 method (NRCan mix, boiler and furnace natural gas credit):

| Term | A, QC | A, ON | B, QC | B, ON |
|---|---|---|---|---|
| Total | −83.0 | −107.2 | −86.9 | −100.5 |
| HTC and drying electricity | 4.1 | 18.6 | 1.6 | 7.0 |
| Credit, industrial heat mix | −79.4 | −121.5 | −33.9 | −51.9 |
| Credit, natural gas heat | −17.4 | −18.6 | −81.7 | −87.5 |
| Dilution water, wastewater, other | 1.5 | 5.9 | 1.7 | 5.6 |

**Reading.** In Quebec the boiler and furnace natural gas credit equals the Paper 1 market credit exactly (−17.4, −81.7), as the CA-QC market is made of the same four technologies: D9 changes nothing in Quebec and only removes the CHP artifact elsewhere. Moving from the Paper 1 to the NRCan Quebec mix changes the hydrochar credit by 1.8 % (−77.9 → −79.4). Ranking: B ahead of A by 4.0 in Quebec, A ahead of B by 6.7 in Ontario; the flip holds on the same method. Quebec → Ontario, A gains 24.2 (heat mix credit −42.1, electricity +14.5), B gains 13.6 (heat mix credit −18.0, natural gas credit −5.8, electricity +5.5). Both gaps are 5 to 8 % of the totals, inside the expected uncertainty of the process-water multiplier and the surrogate; whether the ranking flips must be tested in the sweep with uncertainty.

---

## F. Sweep design

### F1. Feedstocks for the first sweep (2026-11-22)
**Decision.** Food waste, OFMSW, sewage sludge, and a fourth class (proposed: dairy manure, the only other class with both Pathway B degradability and digestate coefficients). Real compositions from the HTC training dataset, not independently varied elements.
**Note.** In Pathway B, OFMSW uses the food waste degradability and digestate values (`PATHWAY_B_FEED_PARAMS`, `AD_CHARACTERIZATION`), so OFMSW differs from food waste only through composition and moisture.

### F2. HTC surrogate prediction error not sampled for now (2026-11-22)
**Decision.** The first sweep samples process and LCA parameters only; surrogate error is not included.
**Consequence.** Probabilities that one pathway beats the other are conditional on the surrogate's point predictions and will be overconfident; they can rank scenarios but are not publishable as probabilities until surrogate error is added (see C0).

---

## E. Open items

1. Alberta and Saskatchewan mixes (85 to 88 % natural gas) likely include upstream oil and gas; subtracting NRCan's upstream mining table would fix it, otherwise state as a limitation.
2. Quebec vs RoW dataset test: Quebec preset run with RoW heat datasets, to size the CA-QC vs RoW artifact on credits.
3. Sweep notebook: provinces x pathways x feedstocks with the uncertain parameters (PW multiplier, B fD, k_hyd) sampled, to test whether the A/B ranking flips beyond uncertainty.
4. HTC surrogate (C0) with supervisors.
5. Sweep notebook.
