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
