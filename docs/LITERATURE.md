# Primary sources and implementation references

1. Mazzetto, A., and Upfal, E. (2023). **An Adaptive Algorithm for Learning with
   Unknown Distribution Drift.** NeurIPS36. Algorithm1 and the linear-regression
   application motivate the separate MU baseline, not the joint NN heuristics.
   https://proceedings.neurips.cc/paper_files/paper/2023/file/1fe6f635fe265292aba3987b5123ae3d-Paper-Conference.pdf
2. Brock, T., and Nagler, T. (2026). **Fast Rates for Nonstationary Weighted Risk
   Minimization.** arXiv:2602.05742. Weighted risk analysis and temporal weighting;
   no implemented data-driven selector is attributed to this paper.
   https://arxiv.org/abs/2602.05742
3. Herbster, M., and Warmuth, M. K. (1998). **Tracking the Best Expert.** Machine
   Learning32,151--178. Fixed-share expert tracking.
   https://doi.org/10.1023/A:1007424614876
4. River **ADWIN** official API and underlying research references.
   https://riverml.xyz/latest/api/drift/ADWIN/
5. River **PageHinkley** official API and underlying research references.
   https://riverml.xyz/latest/api/drift/PageHinkley/
6. River **ARFRegressor** official API and underlying research references.
   https://riverml.xyz/latest/api/forest/ARFRegressor/
7. River **HoeffdingAdaptiveTreeRegressor** official API and research references.
   https://riverml.xyz/latest/api/tree/HoeffdingAdaptiveTreeRegressor/
8. River0.22.0 release, pinned implementation used in this benchmark.
   https://pypi.org/project/river/0.22.0/

API pages may describe a newer version; the recorded environment pins River0.22.0.
Source code, settings and adaptation details are explicit in METHODS.md.

The benchmark's mathematical starting point is the author's supplied unpublished
manuscript **Error Decomposition for Neural Network Learning under Nonstationarity**
(2026), especially Sections1,3--5,6.6 and7. That manuscript is not republished in
this public repository. The empirical implementation/selection experiments are
separate from its deterministic-oracle and concentration results.
