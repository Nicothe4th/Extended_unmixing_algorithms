# Extended_unmixing_algorythms
Matlab and python, Extendend Spectral Unmixing Algorythms for Linear and Multilinear Mixing Models 

**Author:** Juan Nicolas Mendoza-Chavarria & Daniel Ulises Campos Delgado  

**MATLAB Version:** Tested on MATLAB R2021b+  
**Python Version:** Tested on Python 3.9.20  
joblib==1.1.1  
matplotlib==3.5.3  
numpy==1.22.3  
pandas==1.4.4  
scikit_learn==1.1.3  
scipy==1.7.3



##  Description
This toolbox implements state-of-the-art **hyperspectral unmixing** algorithms, including:
- **EBEAE**: Extended Blind End-Member and Abundance Extraction. (DOI: [10.1109/ACCESS.2019.2958985](https://doi.org/10.1109/ACCESS.2019.2958985))
- **EBEAETV**: Extended Blind End-Member and Abundance Extraction with Total Variation. (DOI: [0.1016/j.jfranklin.2023.08.027](https://doi.org/10.1016/j.jfranklin.2023.08.027))
- **EBEAESN**: Extended Blind End-Member and Abundance Extraction with Sparse Noise.
- **EBEAESNTV**: Extended Blind End-Member and Abundance Extraction with Sparse Noise and Total Variation.
- **ESSEAE**: Extended Semi-Supervised End-Member and Abundance Extraction.

- **NEBEAE**: Nonlinear Extended Blind End-Member and Abundance Extraction. (DOI: [10.1016/j.sigpro.2022.108718](https://doi.org/10.1016/j.sigpro.2022.108718))
- **NEBEAETV**: Extended Blind End-Member and Abundance Extraction with Total Variation. (DOI: [10.1016/j.jfranklin.2024.107282](https://doi.org/10.1016/j.jfranklin.2024.107282))
- **NEBEAESN**: Extended Blind End-Member and Abundance Extraction with Sparse Noise.
- **NEBEAESNTV**: Extended Blind End-Member and Abundance Extraction with Sparse Noise and Total Variation.
- **NESSEAE**: Extended Semi-Supervised End-Member and Abundance Extraction.

These methods are designed to estimate end-members and abundances in hyperspectral images while incorporating **spatial coherence, total variation, and nonlinear interactions**.
