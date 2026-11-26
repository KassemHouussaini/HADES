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
**Amendment (2026-11-23).** `main` received one bug-fix commit (2edbcde, B4): Pathway A batch mode and the `ADM1_MODE` option in `HADES.ipynb`. Continuous results, and so the article results, are unchanged; the tag still points to the submitted code.

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

### C0b. HTC surrogate applicability domain: two layers (2026-11-24)
**Decision.** Layer 1 (existing): each of the 10 surrogate inputs within the training 1st to 99th percentile. Layer 2 (new): distance to the nearest training point in the standardized joint input space (k = 1, NaN-tolerant Euclidean, at least 6 inputs in common; 26 % of training rows lack VM and FC), threshold D_T = mean + 2 sd of the nearest-neighbour distances (Netzeva et al. 2005; Sahigara et al. 2012; Roy, Kar & Ambure 2015). Every run records the distance (`ad_knn_distance`) and flags runs beyond D_T in `qa_flags`.
**Key choice.** The threshold is computed from each training point's nearest neighbour in **another study**: rows of one study are the same feedstock at other conditions, so within-study neighbours make the data look twice as dense as it is for a new feedstock (mean nearest-neighbour distance 0.56 within studies vs 1.18 between; Z = 2 thresholds 1.34 vs 2.50). D_T = 2.50; 3.1 % of training points beyond.
**Validation (grouped 5-fold CV by study, out-of-fold errors of the 7 targets vs distance).** Error rises gradually, without a sharp cliff: median relative error 0.9 to 1.1 for the nearest half (distance up to 1.12), 1.25 to 1.65 for the next 40 %, 2.0 for the farthest 10 % (above 1.72); hydrochar yield error 4.9 to 7.6 points in the nearest half, up to 11.6 beyond. Beyond D_T (2.50) the error is 2.4 times that inside. The within-study threshold (1.34) would flag 31 % of new-study points whose error is only 1.5 times higher. Reproducible: `python "ML training scripts and models/applicability_domain.py"`.
**Sweep feedstocks.** All inside (T 180 to 240 °C, RT 60 to 240 min, their moisture ranges): food waste 0.33 to 0.74, dairy manure 0.45 to 0.90, OFMSW 0.67 to 1.07, sewage sludge 0.81 to 1.42.
**Note.** The data issues described in the user's earlier AD analysis (an N entry of 33 %, a shifted yield block, pyrolysis rows at 450 to 600 °C) are not in the repository dataset (max N 13.1 %, max T 300 °C); the user's laptop copy may differ (to confirm). The domain check flags extrapolation but does not reduce the surrogate's error inside the domain (C0).
**Code.** see commit "Applicability domain".

### C1. Pathway B digester feed: feedstock-specific degradability (2026-11-21)
**Decision.** Each feedstock gets its biodegradable fraction fD (X_I = 1 − fD, S_I = 0), the soluble share of the degradable COD, and k_hyd, each with a range. Replaces X_I 0.08, S_I 0.02, soluble 0.27, k_hyd 10 d⁻¹ (BSM2) for every feedstock.
**Why.** With the old inputs, manure, straw and sludge digested like food waste (99.5 % hydrolysis at 20 d HRT). Against 13 independent studies the old inputs passed 3; per-feedstock values passed 10.
**Source.** Fisgativa et al. (2020) Data in Brief 29:105212 (Tables 3, 5); Labatut, Angenent & Scott (2022) Environ. Sci.: Water Res. Technol. 8:1507, ESI S7 and S8; Jimenez et al. (2020) Waste Manag. 101:150; Trzcinski & Stuckey (2012) Environ. Eng. Sci. 29:848; Zerback et al. (2022) Processes 10:1048; Sotemann et al. (2005) Water SA 31(4); Hoover (INL, 2016). Values and ranges in `ADM1.PATHWAY_B_FEED_PARAMS`.
**Code.** f9857fc, 671eed3. Overrides for sweeps and practitioners: `ctx.b_feed_overrides` (12b32e0).
**Paper 1.** Pathway A unchanged. Pathway B base case (food waste) changes: CH4 −13.1 %, hydrochar +20.7 %; frozen B values updated on purpose.

### C1b. Hydrolysis kinetics and HRT: effect on Paper 1 (2026-11-23)
**Finding.** With the Paper 1 feed (X_xc = 0, so the BSM2 disintegration step, k_dis 0.5 d⁻¹, is bypassed; k_hyd 10 d⁻¹), hydrolysis has a half-life of 0.07 d: a batch BMP of Pathway B plateaus within about one day, whereas measured food waste BMPs take 15 to 30 days (test end criterion: Holliger et al. 2016, Water Sci. Technol. 74:2515). With the Paper 2 feed (k_hyd 0.18 d⁻¹, 44 % soluble), the batch curve rises over one to three weeks (checked by the user in `HADES.ipynb`, batch mode).
**HRT.** 30 d is a sound design HRT; the Paper 1 kinetics made any HRT above a few days look complete. Pathway B, food waste, continuous, CH4 kg/t (mock dry run):

| HRT | Paper 1 code | Paper 2 code |
|---|---|---|
| 20 d | 27.74 | 23.22 |
| 30 d | 28.09 | 24.40 |
| 40 d | 28.35 | 25.14 |
| 20 → 40 d | +2.2 % | +8.3 % |

The small HRT sensitivity in Paper 1 is a model artifact. At 30 d, a steady-state digester hydrolyses k·HRT/(1 + k·HRT) = 84 % of the particulate COD, so about 91 % of the degradable COD is converted (vs about 100 % in Paper 1); with fD 0.86 instead of 0.90 this gives 0.91 × 0.86 / 0.90 = 0.87, matching the modelled −13.1 % (about two thirds kinetics, one third fD).
**Effect on Paper 1 results.** Pathway A unaffected (process water is soluble). Pathway B, Quebec, GWP: biomethane credit −94.0 → −81.7, hydrochar credit −27.6 → −33.3, with the digester heat change B loses about 6 kg CO2 eq/t; B's lead over A narrows from about 11 to about 5 kg CO2 eq/t. Direction unchanged.
**Proposed limitation sentence for Paper 1 (revision).** "Pathway B uses BSM2 hydrolysis kinetics without the disintegration step, which overstates methane at 30 d HRT; with feedstock-specific kinetics (Fisgativa et al. 2020) methane decreases by 13 % and hydrochar increases by 21 % for food waste, narrowing the GWP advantage of Pathway B over Pathway A in Quebec from about 11 to about 5 kg CO2 eq/t without reversing it." To decide with supervisors.
**Side observation.** Paper 2 hydrochar energy of B is not monotonic in HRT (594, 583, 615 MJ/t at 20, 30, 40 d), most likely the step-shaped response of the HTC surrogate to small digestate changes (see C0); sweeps will contain such jumps.

### C1c. OFMSW degradability separate from food waste (2026-11-25)
**Decision (user).** Pathway B OFMSW (source separated) gets fD 0.60 to 0.80, central 0.70, instead of the food waste 0.80 to 0.90. f_sol and k_hyd stay those of food waste.
**Why.** The food waste fD (0.86) is Fisgativa et al. 2020 "biowaste": kitchen waste and leftovers from a catering establishment (checked in their Table 1), not household OFMSW with garden waste. Lignin lowers anaerobic biodegradability: B = 0.830 − 0.028 X, X = lignin in % of VS (Chandler et al. 1980, Predicting methane fermentation biodegradability, OSTI 5595813, Eq. 24; read in the report). The intercept underpredicts low-lignin wastes (0.68 for food waste at 5.3 % lignin in VS, from the data article's median 4.9 % dry, against 0.86 measured), so only the slope is used: fD(OFMSW) = fD(food waste) − 0.028 × (lignin OFMSW − lignin food waste). With 8 to 15 % lignin in VS (assumed for a kitchen plus garden mix, not measured) this gives 0.60 to 0.80. Check of the BMP-ratio route on Fisgativa's biowaste: measured BMP 472 NL CH4/kg VS against 519 computed from the F15 food waste composition, ratio 0.91 against their fD 0.86.
**Open.** Measured lignin or BMP of source-separated OFMSW would replace the assumed lignin; k_hyd likely also lower than food waste (not changed). Ali et al. 2020 (Renew. Sustain. Energy Rev., doi:10.1016/j.rser.2020.110637) report OFMSW BMPs of 177 to 580 NL CH4/kg VS.
**Code.** `ADM1.PATHWAY_B_FEED_PARAMS["ofmsw"]`; test `test_pathway_b_feed.py`. 128 tests pass.
**Paper 1.** Unchanged (Paper 1 Pathway B is food waste). F1's note that OFMSW differs from food waste only through composition and moisture no longer holds for Pathway B degradability. Morris results for OFMSW (F9, F10) used fD 0.80 to 0.90; with 0.60 to 0.80 Pathway B for OFMSW will produce less methane.

### C1d. Pathway B feed chemistry from Fisgativa et al. 2020: option under test (2026-11-26)
**Problem (pilot, main_v1).** Pathway B feeds carried almost no alkalinity: feed inorganic carbon from equilibrium with air (0.0007 kmol C/m³, raised to the 0.01 floor in all 36 B points of every class), feed ammonium zero, ions from a charge balance at pH 7.2 around 0.04. Dairy manure B ran at pH 6.4 to 6.6, and at HRT 20 d the solver hit its pH floor (4) and methane was zero (6 of 36 pilot points). The B lipid share came from (1 − O/C)/0.9 and sat at its 0.40 cap for food waste, OFMSW and sludge.
**Option.** `chemistry="fisgativa"` (`SystemContext.b_feed_chemistry`, sweep point key `b_feed_chemistry`) follows Fisgativa et al. 2020, Data in Brief 29:105212, with their measured data per kg TS so that it scales to the diluted digester feed: inorganic carbon = measured TIC (Table 2); ammonium = measured NH4+ share of TKN applied to the sampled feed N, protein from the organic N; cations and anions = their ADM1 inputs (Table 5); lipid share of COD = measured (NMR). Food waste and OFMSW = "Biowaste" (catering kitchen waste); dairy manure = "Manure"; sewage sludge = mean of primary and secondary per kg TS. Their charge-balance equation (section 2.4, step 5) does not reproduce their own Table 5 (horse feed: 0.10 vs 0.23 kmol/m³), so the published inputs are used. Their inert-N content (Ni) is not used: with X_xc = 0 the model releases no N from inerts, so Ni has no effect here. Continuous mode only. Default stays "legacy" until the user decides.
**Test (mock background, ON, all natural gas, 220 °C, 60 min, samples 0 and 1 of main_v1, HRT 20 and 30 d).** Dairy manure: pH 6.43–6.61 → 6.96–7.07; the failed point (sample 1, HRT 20 d) now produces 9.8 kg CH4/t instead of 0; elsewhere methane −0.9 to −1.0 %; total ammonia 0.1–0.25 → 0.21–0.43 g N/L. Food waste: methane and pH unchanged (< 0.01 % and +0.005). Sewage sludge: methane −0.8 to −1.0 %, pH −0.03, ammonia +5 %. Hydrochar energy moves −23 % to +5 % in some points with almost the same methane, through the surrogate's response to small digestate composition changes (F18).
**Code.** `ADM1.FISGATIVA_FEED_CHEM`, `translate_feedstock_to_adm1(..., chemistry=)`; tests `test_pathway_b_feed.py` (manure values per kg TS, Table 5 recovered at the substrate's own TS, legacy default unchanged). 139 tests pass, frozen Paper 1 values unchanged.

### C1e. Pathway B biochemical split from the composition (Kleerebezem & van Loosdrecht 2006): option under test (2026-11-26)
**Option.** `chemistry="fisgativa_kvl"`: inorganic carbon, ammonium, cations and anions as in C1d (Fisgativa et al. 2020, measured per class), but the protein / lipid / carbohydrate COD split of the degradable feed computed from each sampled composition with the electron balance of Kleerebezem, R. & van Loosdrecht, M.C.M. (2006), Water Sci. Technol. 54(4):167, doi:10.2166/wst.2006.538: per C-mol CH_yO_zN_v, eta_LIP = (y − 2z − 3v − u)/(gamma_LIP − 4) (eq. 9, gamma_LIP 5.68, u = 0), eta_PR = v/0.26 (Table 1), carbohydrate the rest; COD shares from eta × gamma (eq. 10). Elemental route used, not their COD/TOC route. Bounds: lipid in [0, 1 − protein] (cap recorded). Assumption: the degradable part has the split of the whole organic matter.
**Why.** The legacy lipid estimate (1 − O/C)/0.9 sat at its 0.40 cap for food waste, OFMSW and sludge in every pilot point and gave 0.36 for manure against 0.047 measured. Kil et al. (2017, Chem. Eng. Commun., doi:10.1080/00986445.2017.1367670) was reviewed and not used: it estimates the feed from effluent measurements of an operating digester.
**Checks.** Reference molecules of their Table 1 recovered (carbohydrate, protein, tripalmitin each 100 %). Central F15 compositions: lipid COD share 0.30 food waste, 0.33 OFMSW, 0.33 sludge, 0.066 manure (Fisgativa measured 0.40, —, 0.14–0.26, 0.047); protein 0.18 food waste and 0.19 manure (Fisgativa 0.19 and 0.16). Over the 100 sampled compositions per class: food waste median 0.29 (5–95 %: 0.04–0.51; 4 % at 0), OFMSW 0.34 (0.27–0.40), sludge 0.32 (0.12–0.60), manure 0.09 (0.00–0.21; 8 % at 0). Food waste against the user's data article (lipids median 14.9 % dw, IQR 11.3–20.8 %, converted with Fisgativa's biowaste COD/VS 1.67): median 0.26, IQR 0.19–0.36. Sludge upper end (0.6) is high.
**Digester test (mock background, ON, all natural gas, 220 °C, 60 min, samples 0 and 1, HRT 20 and 30 d).** Against "fisgativa": methane within 0.1 % (manure, sludge) and −0.8 % (food waste sample 1); pH 6.98–7.11 manure, 7.11–7.17 food waste (−0.07), 7.43–7.49 sludge; ammonia within 20 %; no failed digester. Hydrochar energy −10 % to +4 % at some points (surrogate response to small digestate changes, F18). The lipid share now varies with the sample (food waste 0.40 and 0.24 for samples 0 and 1).
**Code.** `ADM1.kvl_cod_split`, `derive_adm1_fractions` (`split="kvl"`), `translate_feedstock_to_adm1(chemistry="fisgativa_kvl")`; `adm1_lipid_share` recorded in every run and sweep row. Tests `test_kvl_pure_compounds`, `test_kvl_mixture_and_bounds`, `test_fisgativa_kvl_option`. 142 tests pass; default still "legacy", frozen Paper 1 values unchanged.

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

### B4. ADM1 mode selectable; Pathway A batch result fixed (2026-11-23)
**Decision.** `HADES.ipynb` offers `ADM1_MODE`: "continuous" (default; full-scale digester at steady state, used for the LCA of a plant and for all sweeps) or "batch" (BMP-test simulation with inoculum and blank correction, for comparison with lab BMP data), with `BATCH_DAYS` and `ISR_BATCH`.
**Bug found.** In batch mode, Pathway A recorded the raw test-bottle methane (inoculum included) instead of the blank-corrected value, and did not scale from the bottle to the FU. The bottle is filled at the substrate COD concentration over the whole liquid volume, inoculum included, so it holds more COD than the process water of the FU (base case 60.15 vs 27.46 kg). Base food waste: 17.83 kg CH4/t recorded (above the 6.86 kg theoretical maximum of the COD fed); corrected 4.69 kg CH4/t (239 mL CH4/g COD), against 5.20 in continuous mode. Pathway B batch was already correct (its bottle holds exactly the FU COD).
**Scope.** Paper 1 results used continuous mode only and are not affected. The PW-ADM1 validation figure (batch) normalizes by the COD loaded in the bottle and is correct.
**Paper 1 code.** The same fix and the `ADM1_MODE` option were applied to `main` at the user's request (commit 2edbcde; batch fix, notebook option and batch regression test only, no other Paper 2 change; 56 tests pass, frozen Paper 1 values unchanged). The submitted version remains under the tag `waste-management-submission` (5f1743f). See A1.
**Code.** `paper2-dev` 7329c7b, `main` 2edbcde.

---

## F. Sweep design

### F1. Feedstocks for the first sweep (2026-11-22)
**Decision.** Food waste, OFMSW, sewage sludge, and a fourth class (proposed: dairy manure, the only other class with both Pathway B degradability and digestate coefficients). Real compositions from the HTC training dataset, not independently varied elements.
**Note.** In Pathway B, OFMSW uses the food waste degradability and digestate values (`PATHWAY_B_FEED_PARAMS`, `AD_CHARACTERIZATION`), so OFMSW differs from food waste only through composition and moisture.

### F2. HTC surrogate prediction error not sampled for now (2026-11-22)
**Decision.** The first sweep samples process and LCA parameters only; surrogate error is not included.
**Consequence.** Probabilities that one pathway beats the other are conditional on the surrogate's point predictions and will be overconfident; they can rank scenarios but are not publishable as probabilities until surrogate error is added (see C0).

### F3. Uncertain parameters exposed per run (2026-11-23)
**Decision.** Every parameter with a range in the code can now be set per run without editing modules: `ctx.hi_overrides` for the heat-integration constants (`SWEEP_HI_PARAMS`: ambient temperature 5 to 20 °C, extra HTC loss 0 to 0.4, dewatering temperature loss 0 to 10 K, press electricity 20 to 30 kWh/t TS, dryer heat 1.8 to 4.32 MJ/kg water, separation temperature 60 to 80 °C, digester loss fraction 0.1 to 0.3), `ctx.u_fc` for the digestate fixed-carbon ratio (position in its measured range), plus the existing `eps_HR` (0.5 to 0.6), `MC_char` (0.30 to 0.50), `nox_conv` (0.01 to 0.25), `pw_ch4_multiplier` (0.55 to 1.45) and `b_feed_overrides` (fD, f_sol, k_hyd per class). Overrides are applied for one `build_system` call and restored; values outside their range are refused.
**Not sampled.** The digestate f_NH4 (used only as a fallback when ADM1 ammonium is out of range).
**Range quality.** Sourced: press electricity, dryer heat, MC_char, eps_HR, NOx conversion, PW multiplier, Pathway B degradability, FC ratio. Model assumptions (to state in the paper): ambient temperature, separation temperature (60 to 80 °C, no citation in the code), dewatering temperature loss, digester loss fraction (0.2 checked against a wall-loss calculation), extra HTC loss (0 default, 0.4 reported by one design).
**Code.** see commit "Sweep parameters".

### F4. Sweep engine: in-memory builds, cached ADM1, precomputed unit scores (2026-11-23)
**Decision.** Sweeps run through `ADM1/sweep_engine.py`. Workers build each run in memory (no Brightway writes) and only read ecoinvent; each point runs ADM1 once and is evaluated for every LCA context and heat configuration (ADM1 results cached by their exact inputs, caps recorded inside ADM1 replayed on a cache hit; HTC surrogate loaded once per worker). Impacts are the walk of the foreground times unit scores of the background activities and characterization factors of the foreground emissions, computed once per design (`precompute_scores`) and checked against a full Brightway LCA (`validate_point`, tolerance 0.01 %). Every point is saved to its own file (resumable). One thread per worker.
**Why.** Profiling (mock dry run): ADM1 integration is 97 % of the run time (spin-up and main run about half each); everything else is under 1 s. Parallel Brightway runs on one SQLite project fail on concurrent writes, and a full LCA per worker needs about 1 GB of memory (likely cause of the earlier crashes above 8 workers). Measured: one process run 16 to 21 s, about 250 MB, single thread.
**Check.** Mock run: frozen base-case values reproduced in every context, 2 ADM1 runs per point for 4 builds, identical caps on cached builds, totals equal category sums, resume skips finished points.
**Validation on the real ecoinvent 3.12 project (2026-11-23).** `sweep.ipynb` steps 3 to 5 on the user's laptop: 70 background activities, 35 foreground flows; fast unit-score path (redo_lcia) used, 0.4 min for 3 methods. Engine vs full Brightway LCA, relative difference about 2e-8 for all three methods on two probe points (food waste A, Ontario, own biogas; food waste B, Quebec, Paper 1 heat). 12 points on 12 workers ran in parallel on Windows without error (2.7 min including worker start-up).
**Code.** see commit "Sweep engine".

### F5. Morris screening design (2026-11-23)
**Decision.** 15 factors (PW methane multiplier; fD, k_hyd, f_sol; digestate FC position; heat-recovery effectiveness; hydrochar moisture after press; fuel N to NOx; ambient temperature; extra HTC loss; dewatering temperature loss; press electricity; dryer heat; separation temperature; digester loss fraction), uniform over their ranges, Morris one-at-a-time trajectories, r = 20, 4 levels (320 samples), paired A and B. Scenarios: food waste and OFMSW (Paper 1 compositions and moisture), T 180 and 260 °C, RT 60 min, contexts QC, ON, AB, heat configurations Paper 1, all natural gas, own biogas. 2,560 points, about 2 h on 12 workers. Importance: mu* (Campolongo et al. 2007) for A, B and A − B, scaled by the largest mu* per scenario and summarized by the median over scenarios.
**Pending.** Sewage sludge (composition: median of 15 compositions from 9 studies in the HTC dataset) and dairy manure (one composition with VM/FC, manure type not stated, 2 studies only) need an as-received moisture with a source before they enter.
**Code.** `notebooks/sweep.ipynb`.

### F6. Design variables in the screening; sludge and manure compositions (2026-11-23)
**Decision.** The Morris screening also includes four design variables as factors (19 in total, 400 samples, 3,200 points for food waste and OFMSW): HRT 20 to 40 d; HTC feed moisture target 0.75 to 0.90 (inside the surrogate training data, 71 to 95 % slurry moisture, 5th to 95th percentile); process-water COD to the digester 5 to 40 kg COD/m³ (Pathway A; range set by the author); digester feed solids 5 to 12 % TS (Pathway B; range set by the author). T and RT stay on a grid (surrogate steps), province and heat supply as categories. Design variables are screened for importance only; the main sweep puts those that matter on a grid.
**Dilution logic (unchanged, as in the code).** A feed drier than a target gets water; a wetter feed is never dried. HTC reactor moisture = max(feed moisture, target); digester feed TS and process-water COD are diluted only when above the target. Exception: the Pathway B digestate (about 97 % water) is mechanically dewatered (centrifuge) to the HTC moisture target, with water added back if the cake is drier.
**Compositions.** Sewage sludge: mean of 13 compositions from 9 studies in the HTC dataset (C 31.25, H 4.52, N 4.14, S 1.75, ash 40.26, VM 52.32, FC 7.42 % dry); two rows with 0.1 % ash (doi:10.1016/j.biombioe.2018.11.016) excluded as not sludge. Dairy manure: mean of 3 raw manure compositions from 2 studies (Belete et al. 2021, doi:10.1016/j.biortech.2021.125164; doi:10.1016/j.renene.2020.07.003) (C 37.80, H 5.03, N 2.37, S 0.47, ash 25.37 % dry; VM 60.77, FC 13.87 from the single row reporting VM/FC); the dataset's "manure" class also contains two Belete digestate rows (ash 44.8, 45.7 %), excluded.
**Pending.** As-received moisture of sludge and manure (none in the code or usable in the dataset). Secondary figure found: lactating dairy cow manure about 88 % water (MWPS-18, cited in an Ohio administrative code appendix); the primary standard (ASABE D384.2) could not be checked. Sources for the COD and TS ranges to be added.
**Code.** see commit "Design factors".

### F7. Feedstock moisture: central values and ranges, sampled in the screening (2026-11-23)
**Decision.** Feed moisture (% wet basis) is a natural feedstock property and is sampled within each feedstock's range (Morris factor `feed_WC`, position in the class range). Central values and ranges:

| Feedstock | Central | Range | Basis |
|---|---|---|---|
| Food waste | 85 (Paper 1) | 74 to 90 | Zhang et al. 2007, Bioresour. Technol. 98, 929 |
| OFMSW | 70 (Paper 1) | 50 to 92 | As-received OFMSW characterizations, strongly collection-dependent; 67 to 70 % in a one-year German study |
| Sewage sludge (dewatered) | 78 | 71 to 85 | Mechanically dewatered sludges 71 to 85 % (Energies, MDPI, wastewater sludge characterization); 75 to 82 % typical (Brunel thesis); HTC studies at 74.9 and 77.2 % |
| Dairy manure | 85 | 83 to 88 | Fresh manure 83 to 84 % (16 to 17 % TS, lab studies), 85.7 % as excreted (Cornell report), about 88 % (MWPS-18 via Ohio code); ASABE D384.2 not checked |

The no-moisture entries in the code are thereby filled; sludge and manure now run. Screening: 20 factors, 420 samples, 4 feedstocks, 6,720 points.
**To verify.** The sources are from web searches (abstracts and snippets); the full texts should be checked and cited properly before the paper.
**Code.** see commit "Feed moisture".

### F8. Screening temperatures 180 and 240 °C (2026-11-23)
**Decision.** The Morris screening runs at T = 180 and 240 °C (was 180 and 260), user's choice. Sweep folder `morris_v2`; `morris_v1` holds only the 12 timing points of the 180/260 design. Size unchanged: 6,720 points, about 12 h on 12 workers.

### F9. Morris screening results (2026-11-24)
**Run.** 6,720 points (420 samples, 4 feedstocks, T 180 and 240 °C, RT 60 min), each for QC, ON, AB and three heat supplies: 72 scenarios, 0 failed evaluations, 0 missing unit scores, about 84 s per point on 12 workers. Sanity check on the first 252 points: Pathway B methane / theoretical maximum from feed COD (N to NH3) = 0.70 (food waste, OFMSW), 0.53 (sludge), 0.37 (manure), consistent with fD.
**Result.** Median over scenarios of mu* scaled by the largest mu* per scenario, A − B, GWP / human health / ecosystem quality: feed moisture 0.82 / 0.88 / 0.75; process-water COD target 0.79 / 0.76 / 0.81; digester feed TS target 0.69 / 0.47 / 0.61; HTC moisture target 0.38 / 0.44 / 0.44; ambient temperature 0.38 / 0.33 / 0.27; PW methane multiplier 0.30 / 0.30 / 0.30; fD 0.26 / 0.26 / 0.39; extra HTC loss 0.23 / 0.27 / 0.20; HRT 0.18 / 0.20 / 0.30; preheat effectiveness 0.16 / 0.17 / 0.13; k_hyd and digestate FC ratio 0.10 to 0.25. Below 0.15 in all three methods: hydrochar moisture after press, dryer heat, digester loss fraction, separation temperature, dewatering temperature loss, press electricity, soluble share, fuel N to NOx (NOx capped at the emission limit in almost all runs).
**Reading.** The A − B comparison is driven mainly by feed moisture and the two dilution targets (design choices), ahead of the genuine uncertainties. mu* scales with the range width, so the COD (5 to 40) and TS (5 to 12 %) ranges, set by the author, must be supported by literature before the main sweep. Ambient temperature (assumed range) ranks fifth and is a site property. "All natural gas" and "own biogas" give nearly equal results because own biogas converts methane to heat without boiler loss (fixed assumption F4 in the variable table), which makes the two equivalent. *Corrected in F12: the equality is an energy identity of the substitution, not a missing loss.* RT was not screened (60 min only).
**Proposed for the main sweep (to confirm).** Feed moisture as a grid axis (3 levels per class); COD (A) and TS (B) as grid axes with sourced ranges; ambient temperature per province (climate normals); HTC moisture target and HRT fixed at defaults with separate sensitivity; the 8 negligible factors fixed at defaults; Latin hypercube over the PW multiplier, fD, k_hyd, digestate FC ratio, preheat effectiveness and extra HTC loss; T 180/220/240, RT 60/240. About 8,640 points with N = 20 (17 h on the laptop).

### F10. Screening: who wins and why (GWP, indicative) (2026-11-24)
**Share of samples where A beats B** (all heat supplies, T 180/240, QC/ON/AB): OFMSW 10 %, sewage sludge 18 %, food waste 24 %, dairy manure 48 %. Morris samples are correlated discrete levels, so shares are indicative only; surrogate error not included.
**Mechanism (all natural gas, medians, kg CO2 eq/t).** A earns its credit from hydrochar (food waste −126, manure −102), B from biomethane (food waste −109, OFMSW −242, manure −46). A's main burden is heating the whole wet feed in the HTC (food waste +51 vs +18 for B). Highly degradable feeds (fD 0.86) give B a biomethane credit that outweighs A's larger hydrochar credit; poorly degradable manure (fD 0.47) balances; high-ash sludge (40 %) gives low-HHV hydrochar, weakening A.
**Drivers.** Wetter feeds favour B for all four feedstocks (food waste: A wins 36 % at 74 % moisture, 14 % at 90 %). Less dilution favours the pathway concerned (manure: A wins 9 % at 5 kg COD/m³, 60 to 69 % above 17; sludge: A wins 36 % at 5 % TS, 1 % at 12 %). 240 °C mostly favours B. Quebec is B's strongest province (27 % wood in the industrial heat mix lowers the hydrochar credit); Ontario (21 % coal and coke) is A's strongest; Alberta in between. Electric HTC heating helps A in Quebec (food waste: A wins 1 % with gas, 28 % with electricity) and ruins it in Alberta (totals positive).
**Preliminary statements (to confirm in the main sweep).** OFMSW and sewage sludge: B. Food waste: B in most cases, A competitive in Ontario with cheap HTC heat. Dairy manure: no general answer, depends on province and dilution. Wetter feeds favour B.
**Caveats.** Surrogate error (C0) not sampled; COD and TS ranges unsourced yet are drivers; HTC heat without temperature level favours A; own biogas distorted by the no-boiler-loss assumption (*corrected in F12: near-equality by construction*); manure composition from 3 samples, 2 studies.

### F11. Ambient temperature per province from climate normals (2026-11-25)
**Decision.** Each province preset carries an ambient temperature, the annual daily-average air temperature of the ECCC Canadian Climate Normals 1991–2020 at the station of the province's largest city; it replaces the sampled 5 to 20 °C range in the main sweep. Values (°C): QC Montreal-Trudeau 7.0; ON Toronto Pearson 8.6; AB Calgary 4.5; MB Winnipeg 2.9; SK Saskatoon 2.4; BC Vancouver Intl 10.5; NS Halifax Stanfield 6.9; NB Moncton 5.7; PE Charlottetown 5.9; NL St. John's 5.3; YT Whitehorse 0.2; NT Yellowknife −4.0; NU Iqaluit −8.6. `QC_PAPER1` keeps the Paper 1 default (10 °C).
**Why.** Ambient temperature ranked fifth in the screening (F9, scaled mu* 0.38 for A − B in GWP) and is a site property, not an uncertainty. One station of the largest city per province, as plants are sited near the waste.
**Range change.** `T_AMB_RANGE_C` lowered from (5, 20) to (0, 20) so that the Prairie provinces fit. NT and NU (below 0 °C) are refused by `ambient_overrides`: a negative annual air mean cannot stand for the temperature of a liquid feed and water main; set `T_AMB_C` explicitly there.
**Assumption to state.** The annual mean air temperature is used for the raw feed, the tap (dilution) water and the outside of the digester. Water mains are usually warmer than the annual air mean in cold climates, so this slightly overstates heat demand in the Prairies.
**Source.** ECCC Canadian Climate Normals 1991–2020, bulk download from climate.weather.gc.ca/climate_normals (accessed 2026-11-25); stored with station names, climate IDs and the ECCC normals code (A to D, completeness of record) in `ADM1/data/eccc_normals_1991_2020_annual_mean_T.csv`. Calgary, Moncton, Charlottetown and St. John's carry code C (lower completeness than A or B).
**Code.** `lca_contexts.ambient_overrides(preset)`; `sweep_engine.make_context` applies it unless `T_AMB_C` is sampled (point key `ambient`: "province", default, or "default" for 10 °C); `parameter_space(..., exclude=("T_AMB_C",))` drops it from a sampled design; `HADES.ipynb` applies it to the chosen preset. Tests: `tests/test_ambient_presets.py`.
**Effect.** Earlier single runs in Ontario and Quebec (D10, D11) used 10 °C; rerunning `HADES.ipynb` now gives 8.6 and 7.0 °C (slightly higher heat demand, not yet quantified). Morris results (F9, F10) are unaffected (T_AMB_C sampled there).
**Paper 1.** Unchanged (`QC_PAPER1` keeps 10 °C; frozen regression passes).
**Methods text.** "Ambient temperature, applied to the raw feedstock, dilution water and digester heat losses, is the 1991–2020 annual mean air temperature of the province's largest city (Environment and Climate Change Canada, Canadian Climate Normals 1991–2020)."

### F12. Own-biogas heat: efficiency parameter; the F9 reading corrected (2026-11-25)
**Finding.** F9 attributed the near-equality of "own biogas" and "all natural gas" to a missing boiler loss. That reading was wrong. Exported biomethane is credited at 1 MJ of natural gas heat per MJ of methane, and own biogas delivers 1 MJ of heat per MJ of methane. Diverting X MJ of methane to cover X MJ of heat demand therefore removes exactly X MJ of natural gas heat credit, which is what the natural gas configuration pays. The two differ only by the PSA methane loss (3 %) and the PSA electricity avoided on the diverted share, and by the combustion emissions. This is an energy identity and holds for any efficiency applied consistently to both. A real difference exists only if raw biogas is burned on site less efficiently than the displaced natural gas boiler.
**Decision.** `SystemContext.own_biogas_heat_eff` (MJ heat per MJ methane burned on site, relative to the natural gas boiler of the credit; default 1.0, the Paper 1 convention; allowed (0, 1]). The `heat_biomethane` exchanges of own biogas are now written per MJ of methane burned (as for exported biomethane), so combustion emissions follow the fuel when the efficiency is below 1. Recorded in every run (`own_biogas_heat_eff`) and settable per sweep point (`params["own_biogas_heat_eff"]`).
**Not done.** No value below 1 and no range is set: none is sourced yet. Needed: the efficiency of an on-site raw biogas boiler relative to the ecoinvent "heat production, natural gas, at boiler condensing modulating >100kW" (whose efficiency can be read from its natural gas input on the laptop database).
**Open consequence.** The export credit (1 MJ biomethane = 1 MJ delivered heat) is exact only if the displaced boiler has an LHV efficiency of 1. A condensing boiler is close to that; a market average with lower efficiency would make the 1:1 credit too generous to Pathway B. To check with the ecoinvent fuel input per MJ of the four boiler and furnace datasets (D9).
**Code.** `system_builder.plan_heat_supply(..., heat_per_MJ_CH4)`, `SystemContext.own_biogas_heat_eff`, sweep_engine `make_context`. Tests: `tests/test_heat_supply.py::test_own_biogas_heat_efficiency`.
**Paper 1.** Unchanged (default 1.0; 127 tests pass, frozen regression unchanged).


### F13. Main sweep design: what is gridded, sampled and fixed (2026-11-25)
**Decision (user).** Paper 2 is a decision map: probability that A beats B by feedstock class, province and heat supply, break-even conditions, and the drivers. LCA only (no economics for now); surrogate error not sampled until the supervisor meeting (F2, C0). Feedstocks are no longer fixed compositions: composition varies within each class.
**Rule.** Grid what the reader of the guidebook knows or chooses; sample what is uncertain or varies within a class; fix design choices that would otherwise dominate the result.

| Variable | Treatment |
|---|---|
| Feedstock class (food waste, OFMSW, sewage sludge, dairy manure) | Grid |
| Composition within class | Sampled: real measured compositions, equal weight per study (sources to settle) |
| Feed moisture | Sampled within the class range (F7) |
| Province (10) | Grid (no extra ADM1 runs, F4) |
| Heat supply (Paper 1, all natural gas, own biogas) | Grid (no extra ADM1 runs) |
| HTC temperature 180 / 220 / 240 °C | Grid |
| HTC residence time 60 / 240 min | Grid |
| HRT | Grid for Pathway B only (user), 20 / 30 / 40 d proposed; Pathway A fixed at 30 d |
| PW methane multiplier, fD, k_hyd, f_sol, digestate FC, preheat effectiveness, extra HTC loss and the other ranged parameters | Sampled (Latin hypercube; fixing factors saves no runs with LHS) |
| HTC moisture target, COD target (A), digester TS (B) | Fixed (user): 0.85, 25 kg COD/m³, 10 % TS |
| Surrogate error | Not sampled for now |

**Why.** F9: the A − B comparison was driven by feed moisture and the dilution targets; the targets are design choices with author-set ranges. Composition is knowable by a practitioner, so its variation shows whether a recommendation holds for a class and which property flips it.
**HRT for B only (user, 2026-11-25).** Pathway A digests process water that is already soluble, so hydrolysis is never limiting and HRT barely changes its methane (C1b); gridding it for A would spend a third of the runs on an inactive variable. B (particulate feed, k_hyd 0.18 d⁻¹ for food waste) is HRT-sensitive (+8.3 % CH4 from 20 to 40 d, C1b). Size with 100 samples per cell: 4 feedstocks × 3 T × 2 RT × (1 A + 3 B) = 96 cells, about 9,600 points, about 19 h on 12 workers (resumable). Code: `make_points(..., HRT_levels={"A": [30], "B": [20, 30, 40]})`.
**Fixed design values (user, 2026-11-25).** HTC moisture 0.85 (inside the surrogate training data: median 0.90, 5th to 95th percentile 0.71 to 0.95); COD of the Pathway A digester feed 25 kg COD/m³ (code default was 10, unsourced); Pathway B digester feed 10 % TS (wet digestion; code default was 8 %). Only feeds above a target are diluted (F6). Gridding them was considered and rejected for the main sweep: less dilution always favours the pathway concerned (F10), so a grid shows a known monotone trend, includes designs beyond process limits, makes the A vs B pairing combinatorial, and costs about 4 times the runs. A separate design sensitivity run (about 1,000 points) will vary the process limits (ammonia threshold, TS limit, moisture) one at a time. Values remain to be supported by literature (user's research). Code: `sweep_engine.MAIN_SWEEP_FIXED_DESIGN`, `MAIN_SWEEP_HRT_LEVELS`.
**To check in the pilot.** Total ammonia N in the Pathway A digester at 25 kg COD/m³ (protein-rich feeds such as sludge) against inhibition thresholds (1.7 to 14 g N/L, Chen, Cheng & Creamer 2008, Bioresour. Technol. 99:4044).
**Open.** HRT levels (20 / 30 / 40 d to confirm); composition sources per class; sample size per cell (100 proposed).

### F14. Feedstock composition data: coverage and cleaning (proposed, 2026-11-25)
**Finding.** Distinct compositions in the HTC training dataset after cleaning, one mean per study, % dry basis (median; min to max of study means). Food waste, 17 compositions, 13 studies: C 44.5 (36.2–53.9), H 6.2 (4.1–8.9), N 3.4 (1.7–7.0), S 0.2 (0–0.5), ash 9.1 (1.4–21.6), VM 76.4 (67.6–84.8), FC 11.4 (5.1–20.6). Sewage sludge, 17 compositions, 12 studies: C 35.4 (20.7–43.0), H 5.2 (1.6–5.9), N 4.5 (2.5–7.2), S 1.0 (0–1.7), ash 32.0 (13.6–60.9), VM 52.8 (36.8–78.5), FC 8.3 (2.0–12.7). Dairy manure, 4 compositions, 3 studies: C 36.8 (32.7–39.9), N 2.2 (1.4–2.7), ash 15.5 (9.2–30.3). "Solid waste" (OFMSW not confirmed), 3 compositions, 2 studies.
**Proposed exclusions (to confirm).** Rows failing mass closure (O by difference below 5 %: one sludge row with ash 56.6 % and C 40.1 %); S above 5 % (one sludge row, 9.5 %); N below 0.05 % or C above 70 % (two solid-waste rows); a food waste row identical to another of the same study except N 10.0 instead of 1.0 (doi:10.1016/j.biortech.2021.126395); an ash series of one solid-waste sample in 1 % steps (doi:10.1016/j.biortech.2015.01.118, base row kept); non-sludge and digestate rows as in F6. Flagged, kept: food waste N 9.7 % (doi:10.1016/j.biortech.2020.124410); cow dung O 50.9 % by difference (doi:10.1016/j.jenvman.2022.116854).
**Reading.** Food waste and sewage sludge are covered well enough to resample; dairy manure and OFMSW need literature compositions before the main sweep.

### F15. Feedstock input ranges for the main sweep (2026-11-25)
**Decision (user).** Ranges and central values per feedstock, % dry basis unless stated, synthesized from the user's literature table, the food waste data article table (Table 3, n up to 167; citation to add) and the HTC dataset (F14). Sewage sludge is raw mixed sludge (primary + waste activated), not primary only, undigested; it enters as dewatered cake.

| Input | Food waste | Sewage sludge (raw mixed) | Dairy manure | OFMSW (source separated) |
|---|---|---|---|---|
| Moisture (% wb) | 70–90 (80) | 70–85 (78) | 82–90 (86) | 65–80 (73) |
| C | 37–54 (47) | 28–45 (35) | 33–45 (40) | 42–52 (47) |
| H | 4.5–8.0 (6.6) | 4.0–7.0 (5.3) | 4.5–6.0 (5.3) | 5.5–7.0 (6.3) |
| N | 1.7–6.0 (2.8) | 3.0–7.5 (4.8) | 1.4–3.0 (2.3) | 2.0–3.8 (2.8) |
| S | 0.05–0.6 (0.25) | 0.5–1.7 (1.0) | 0.2–0.6 (0.4) | 0.1–0.4 (0.25) |
| Ash | 2–15 (8) | 18–42 (30) | 9–25 (16) | 5–25 (12) |
| VM | 68–85 (79) | 50–75 (62) | 60–77 (69) | 65–82 (75) |
| FC | 5–21 (13) | 3–13 (8) | 11–18 (15) | 8–24 (13) |
| O by difference (result, not sampled) | about 30–44 (37) | about 15–30 (24) | about 28–42 (36) | about 28–40 (32) |

**Use.** Marginals of a log-ratio (Aitchison) composition sampler; correlations between elements from the real compositions of each class (pooled classes for manure and OFMSW). VM and FC are sampled as the FC/(VM+FC) ratio so that VM + FC + ash = 100. Sludge above 45 % ash (likely digested) is outside the class.
**Synthesis choices.** Food waste: centrals of C, N, O and moisture follow the data article (median C 48.2, N 2.6, O 38.8, TS 20 %); ash widened to 15 % (article Q3 12.7, HTC dataset median 9.1). Sludge: ash widened from 20–40 to 18–42 for waste-activated-rich sludges. Manure: C lower bound 33 and ash lower bound 9 from the HTC dataset (cow dung). OFMSW: user's table only.
**Not changed.** Pathway B degradability and digestate FC ratio per class (C1, F3). OFMSW still uses the food waste values (fD 0.80–0.90), which likely overstates degradability for source-separated OFMSW containing garden waste (to revisit).

### F16. Composition sampler (2026-11-25)
**Decision.** Each sample of the main sweep gets its own feedstock composition, drawn by `ADM1/feedstock_sampler.py` in additive log-ratio space with O as reference (Aitchison 1986, The Statistical Analysis of Compositional Data): location at the F15 central composition, spread such that each F15 element range is about ±2 sd, correlations between log-ratios from the real compositions of the class (food waste 17, sewage sludge 16 after F14 cleaning) or, for dairy manure and OFMSW, the within-class correlation pooled over the four classes (40 rows, each class centred on its own mean). Draws outside any F15 range (elements, O, VM, FC) are rejected. The FC share FC/(VM + FC) is drawn separately (logit-normal on the F15 VM and FC ranges) so that VM + FC + ash = 100. S = 0 in the data is replaced by 0.01 % before taking logs. Moisture stays a sweep factor (feed_WC).
**Check (200 draws per class, seed 1).** Medians within 1 % point of the F15 centrals; C and ash negatively correlated in every class (−0.69 to −0.88), as in real feedstocks; every draw inside the HTC surrogate's joint applicability domain (C0b) at T 180/240 °C and RT 60/240 min, at the HTC slurry moisture. Median nearest-neighbour distance 0.7 to 1.0 for food waste, manure and OFMSW, 1.2 to 1.6 for raw sewage sludge (threshold 2.50): raw sludge sits in a sparser part of the training data (F15).
**Code.** `feedstock_sampler.sample(cls, n, seed)`, `central(cls)`, `domain_share(...)`; `sweep_engine.make_points` takes `compositions` (one per sample) in a feed entry and records them in `x` as `comp_*`. Tests: `tests/test_feedstock_sampler.py` (closure, ranges, reproducibility, ash–carbon correlation, centres, surrogate domain, one composition per sample). 135 tests pass.
**Paper 1.** Unchanged.
**Methods text.** "Feedstock compositions were sampled within class-specific ranges in additive log-ratio space (Aitchison, 1986), with the correlation structure estimated from measured compositions in the HTC training dataset, so that every sampled composition closes to 100 % and preserves the co-variation of carbon, hydrogen and ash observed in real feedstocks."

### F17. Main sweep notebook (2026-11-25)
**Decision.** `notebooks/main_sweep.ipynb` runs the F13 design; `notebooks/sweep.ipynb` stays the Morris screening (morris_v2 reproducible). Grid: 4 feedstocks, 11 contexts (10 provinces and Yukon; NT, NU excluded, F11), 3 heat supplies, T 180/220/240 °C, RT 60/240 min, HRT 20/30/40 d for B and 30 d for A. Latin hypercube, N = 100 samples per cell, seed 20261125, 15 sampled factors (all ranged parameters of F3 except ambient temperature, plus feed moisture in the F15 class range); one composition per sample from `feedstock_sampler` (F16); fixed design from `MAIN_SWEEP_FIXED_DESIGN`. 9,600 points (2,400 A, 7,200 B), each scored for 11 × 3 = 33 contexts and heat supplies (316,800 evaluations), about 19 h on 12 workers.
**Pilot.** The first 2 samples of the design (192 points), run first and kept by the full run. Step 6 checks failures, missing unit scores, binding caps, surrogate domain flags, and digester ammonia and pH; total ammonia N and free NH3 (g N/L) are now recorded in every sweep row (`_TAN_gN_L`, `_NH3_free_gN_L`).
**Results cell.** Pairs each B point (three HRTs) with the A point of the same sample, feedstock, T and RT; share of samples where A has the lower impact by feedstock × heat supply × province, then by T, RT and HRT; pairs saved to `pairs.csv`.
**Check (mock background, no ecoinvent).** Design cell builds the 9,600 points; sample 0 at 220 °C, 60 min, food waste and sewage sludge, A and B at HRT 20 and 30 d, built and scored for ON and AB × 3 heat supplies without error. Pathway A sludge digester: 2.4 g N/L total ammonia, 0.06 g N/L free NH3, pH 7.3, at the low end of reported inhibition thresholds (1.7 to 14 g N/L, Chen, Cheng & Creamer 2008); food waste A 1.3 g N/L. A sampled food waste composition triggered the ADM1 lipid fraction cap (0.49 → 0.40): to count in the pilot.
**Paper 1.** Unchanged.

**Amendment (2026-11-26).** Main sweep set to `B_FEED_CHEMISTRY = "fisgativa_kvl"` (C1d, C1e) in folder `main_v2`, user's decision; `main_v1` holds the legacy-chemistry pilot (192 points, manure B acidified, C1d).

### F18. Propagation check before the main sweep (2026-11-25)
**Test.** One input at a time from the low to the high end of its main-sweep range, all others at the centre (central F15 composition, mid-range parameters), food waste and sewage sludge, A and B, 220 °C, 60 min, HRT 30 d, Ontario, all natural gas; mock background (process results exact, impacts not). 172 builds, no failures. Plus a surrogate-only scan: each composition input in 20 steps across its F15 range, four classes.
**Propagates (as intended).** Every composition input changes the surrogate outputs (12 to 20 distinct values of yield, HHV, char C and ash over 20 steps, all classes; no flat region), and the hydrochar energy, methane and heat demands of the pathway concerned. Feed moisture changes every output (dry matter per tonne, dilution water, heating) although the HTC slurry (0.85) and B digester feed (10 % TS) are held at their targets. B-only parameters (fD, k_hyd, f_sol, digestate FC) act on B only; the PW methane multiplier on A only. Hydrochar moisture after press, dryer heat, extra HTC loss and preheat effectiveness act on the heat demands.
**Does not propagate, by design.** (1) Feedstock VM and FC in Pathway B: ADM1 reads elements only and the digestate FC comes from the measured digestate FC ratio (C2); feedstock H and O reach B's HTC only through ADM1 (digestate H/C and O/C are measured per class). (2) Fuel N to NOx (0.01 to 0.25): no effect, because NOx, SO2 and dust stack emissions sit at the emission limit values in practically every run (hydrochar_enduse ELV cap); hydrochar N and S therefore do not change those emissions, and the flue gas treatment needed to meet the limits is not inventoried. (3) Pathway A digester heating is zero in nearly all runs (hot process water and E2 heat cover it), so the digester loss fraction, separation temperature and dewatering loss do not act on A. (4) Press electricity changes totals by 0.2 % at most (genuinely small).
**Amplified.** Pathway A methane comes from process-water carbon by difference (feed C minus hydrochar and gas C; char C / feed C 0.58 to 0.84 over the sampled compositions). Small changes in the surrogate's char C therefore move A's methane by 20 to 130 % for composition changes with little physical effect on methane (e.g. food waste S 0.05 → 0.6 %: −23 %). One corner case (food waste at 2 % ash with every other element central) gave char C above feed C and zero A methane (recorded cap); among the 100 sampled compositions per class, 0 to 1 % have char C / feed C above 0.95. This sensitivity is part of the surrogate error (C0, F2), not sampled yet.
**Not covered.** Interactions (one at a time around one centre); real impact factors (mock unit scores).

### F19. Emission limits: removed mass recorded, burden as a bounding check after the sweep (2026-11-25)
**Finding.** Required removal to meet the limits at the centre of every range (F18 runs, from the recorded caps): SO2 median 29 % (food waste A), 74 % (food waste B), 89 % (sewage sludge A and B); NOx 59 to 84 %; dust 70 to 94 %. This exceeds dry sorbent injection (about 80 % SO2) and SNCR (about 50 % NOx) for sludge and most food waste B runs, so a single simple flue gas treatment model would be extrapolated in much of the sweep.
**Decision (user).** No flue gas treatment model now. Every run records the mass removed per FU, uncontrolled minus emitted (`enduse_SO2_removed_kg`, `enduse_NOx_removed_kg`, `enduse_PM_removed_kg`), with the emitted masses, the flue gas volume and the dry hydrochar mass, in the sweep rows. After the main sweep, a deliberately generous treatment burden (one reagent per pollutant at a high reagent ratio: production, CO2, residue; a few ecoinvent unit scores) is added in post-processing, and the decision map is compared with and without it. If no ranking changes, one limitation sentence with the magnitude; if it changes (e.g. sludge), model treatment for that case only.
**Why.** The sweep stores results per point, not full inventories, so a burden cannot be added after the run without rebuilding every point (correction of an earlier statement that it could be rescored); the removed masses make the post-processing possible. Emitted values are unchanged.
**Still open (for the paper, not blocking).** Sulfur retention in ash (Ca- and P-rich sludge ash), lower fuel-N conversion for high-N fuels, N2O from urea SNCR, CO2 from bicarbonate versus the CO2 taken up in its production.
**Code.** `hydrochar_enduse.compute_enduse_quantities` (`*_removed_kg`), `system_builder` (`enduse_*_removed_kg` on the result), `sweep_engine.META_KEYS`. Test `test_removed_mass_is_uncontrolled_minus_emitted`. 136 tests pass; frozen Paper 1 values unchanged.

---

## E. Open items

1. Alberta and Saskatchewan mixes (85 to 88 % natural gas) likely include upstream oil and gas; subtracting NRCan's upstream mining table would fix it, otherwise state as a limitation.
2. Quebec vs RoW dataset test: Quebec preset run with RoW heat datasets, to size the CA-QC vs RoW artifact on credits.
3. Sweep notebook: provinces x pathways x feedstocks with the uncertain parameters (PW multiplier, B fD, k_hyd) sampled, to test whether the A/B ranking flips beyond uncertainty.
4. HTC surrogate (C0) with supervisors.
5. Sweep notebook.
