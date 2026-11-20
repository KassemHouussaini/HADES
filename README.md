# HADES
## Hydrothermal–Anaerobic Digestion Environmental Study

### A Parameterized Tool for Automated Life Cycle Assessment Modeling of Hydrothermal Carbonization and Anaerobic Digestion Systems

---

## 📖 Overview

Integrated **Hydrothermal Carbonization (HTC)** and **Anaerobic Digestion (AD)** systems offer significant potential for the sustainable valorization of wet biomass. However, the environmental conditions under which HTC–AD integration becomes advantageous—and which process configuration performs best—remain poorly understood.

**HADES** addresses this challenge by providing an open-source framework that combines:

- 🤖 Machine learning prediction of hydrochar properties
- 🧪 An extended **PW-ADM1** model with HTC process water inhibition kinetics
- 🌍 Automated **Life Cycle Assessment (LCA)** using Brightway2
- 💻 An easy-to-use **Streamlit** graphical interface

The tool enables users to propagate feedstock characteristics and operating conditions through the entire HTC–AD system and directly evaluate environmental impacts.

---

## 🔄 Integrated Pathways

HADES evaluates two process configurations:

### **Pathway A**
**HTC → AD**

Hydrothermal Carbonization followed by Anaerobic Digestion of the resulting process water.

### **Pathway B**
**AD → HTC**

Anaerobic Digestion followed by Hydrothermal Carbonization of the resulting digestate.

---

## 🛠️ Built With

- Python
- XGBoost
- Brightway2
- Streamlit
- NumPy
- Pandas
- Scikit-learn

---

## 🎯 Purpose

HADES bridges the gap between process simulation and environmental assessment by integrating mechanistic process models with Life Cycle Assessment into a single automated workflow.

The framework is intended for:

- Researchers
- LCA practitioners
- Waste-to-energy engineers
- Process designers
- Decision-makers evaluating biomass valorization technologies

---

## 📁 Repository layout

```
ADM1/                            PW-ADM1, system builder, heat integration, LCA contributions, tests
ML training scripts and models/  HTC surrogate (XGBoost), training data, training and tuning scripts
notebooks/                       HADES.ipynb (run one system) and the case study and PW-ADM1 validation notebooks
results/                         case study, sensitivity, LCI and Supporting Information tables
figures/                         figure scripts and exported figures of the article
app.py                           Streamlit interface
setup_hades.py                   creates a Brightway project with ecoinvent and IMPACT World+
```

---

## ⚙️ Installation

HADES runs in a Python 3.10 environment with Brightway 2, XGBoost, scikit-learn, SciPy, pandas, matplotlib and Streamlit:

```
conda create -n hades python=3.10
conda activate hades
pip install -r requirements.txt
```

## 🗄️ Setup: ecoinvent and IMPACT World+ (once)

HADES needs a Brightway project with an ecoinvent database (**your own ecoinvent licence is required**) and the IMPACT World+ 2.2.1 methods used in the article. `setup_hades.py` creates the project and imports both:

1. **ecoinvent.** Accept the licence on the ecoinvent website, then either download it directly:

   ```
   python setup_hades.py --project hades --version 3.12 --system-model cutoff
   ```

   (you are asked for your ecoinvent username and password, or set `EI_USERNAME` and `EI_PASSWORD`), or import the ecoSpold2 files you downloaded yourself:

   ```
   python setup_hades.py --project hades --ecospold "C:/path/to/ecoinvent 3.12_cutoff_ecoSpold02/datasets"
   ```

2. **IMPACT World+ 2.2.1.** Download the Brightway file for your ecoinvent version from the CIRAIG IMPACT World+ record on Zenodo, then:

   ```
   python setup_hades.py --project hades --skip-ecoinvent --iw "C:/path/to/impact_world_plus_2.2.1_....bw2package"
   ```

The script ends with a check listing what the project contains. Projects are stored on your computer only; the interface lists the projects of the computer it runs on that contain an ecoinvent database.

## ▶️ Running HADES

- **Interface:** `streamlit run app.py`, then choose the project, background database and LCIA indicators in the sidebar (section 5).
- **One system, your own inputs (code):** open `notebooks/HADES.ipynb`, set the user decisions in its first cell (pathway, feedstock, HTC and AD conditions, Brightway project) and run all cells.
- **Article case study:** run `notebooks/case_study.ipynb` (set `PROJECT` and `EI_NAME` in its first cell to your project), then `notebooks/PW-ADM1 verification figure.ipynb` and `ML training scripts and models/ML verification figure.ipynb`. Results are written to `results/` and figures to `figures/`.
- **Tests:** `cd ADM1` then `python -m pytest` (no Brightway needed).

---

## 📄 Citation

If you use HADES in your research, please cite:

> *Citation coming soon.*

---

## 📜 License

This project is released under the **MIT License**.

---

## 👨‍💻 Main author

**Kassem Ibrahim Al Houssaini**  
PhD Candidate, Polytechnique Montréal (CIRAIG)
