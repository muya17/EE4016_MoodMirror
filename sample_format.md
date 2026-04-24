# TECHNICAL REPORT
**Title:** [Insert Descriptive Title Here]

**Authors:** François Pomerleau  
**Date:** April 28, 2020  
**Affiliation:** Université Laval, NOR Lab

---

## Revision History
| Version | Date | Comments |
| :--- | :--- | :--- |
| 0.1 | April 28, 2020 | Initial writing |

---

## Abstract
[Insert 3-4 sentence goal of the report here. Describe the primary objectives and methodology.]

---

## 1. Definitions
Use the symbols and functions defined in the tables below to maintain consistency with lab notation.

### Table 1: General symbol definitions
| Symbol | Explanation |
| :--- | :--- |
| *a* | A scalar |
| *v* | A vector of D dimensions, $v=[v_{1},v_{2},\cdot\cdot\cdot,v_{d}]^{T}$ |
| *S* (script) | A set of vectors with N elements, $\mathcal{S}=\{v_{1},v_{1},\cdot\cdot\cdot,v_{n}\}$ |
| *S* (matrix) | Matrix representation of size $D\times N.$ $S=[v_{1},v_{1},\cdot\cdot\cdot,v_{n}]$ |
| $a \cdot b = c$ | Dot product (inner product) |
| $||v||_{2}$ | Euclidean distance ($l^2$-distance) |

### Table 2: Symbol definitions for point clouds and registration
| Symbol | Explanation |
| :--- | :--- |
| *d* | Indices for dimension $\{1,2,\cdot\cdot\cdot,D\}$ |
| $p_i, \mathcal{P}, P$ | Point in reading cloud, Set of points, and Matrix version ($D \times I$) |
| $q_j, \mathcal{Q}, Q$ | Definitions for the reference (static) point cloud |
| $e_k, E$ | Matched error $e_{k}=p_{i}^{\prime}-q_{j}$ and its matrix version |
| $X, x_k$ | States of the robot (general and at time *k*) |
| $T, R$ | Rigid transformation matrix and Rotation matrix |

### Table 3: Function Definitions
| Function | Explanation |
| :--- | :--- |
| $T(x,p)$ | Function transforming/moving the reading point cloud |
| $J(x)=a$ | Objective function (loss/cost function) |
| $match(\mathcal{P}, \mathcal{Q})$ | Matching function generating error vectors |

---

## 2. Technical Content 1
[Placeholder for technical discussion. Insert references to literature here, e.g., Pomerleau 2013.]

> **[IMAGE PLACEHOLDER: Insert Diagram of Point Cloud Alignment or Experimental Setup here]**

---

## 3. Technical Content 2
[Placeholder for further analysis, results, or mathematical derivations.]

> **[IMAGE PLACEHOLDER: Insert Results Graph, Error Plot, or Comparison Table here]**

---
*Page X of 4*