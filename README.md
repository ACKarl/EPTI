# EPTI
# Evaluating the performance of a new tool for phylogenetic inference
Given genetic data of organisms, their evolutionary relationships can be inferred in a diagram called a phylogenetic tree. Accurately reconstructing these trees is fundamental to modern biology, as they provide the crucial framework needed to track the spread of infectious diseases, identify driver cancer mutations, and understand the evolutionary histories of species. Recent advances in deep learning, and in particular in transformer-based architectures, have enabled fast phylogenetic inference as compared to gold-standard Bayesian approaches that are highly time-consuming. This project will aim to evaluate the performance of the newly developed deep-learning tool for inferring phylogenetic trees. The project will involve simulating sequencing data along known phylogenetic trees using existing tools and using such data to assess the performance of the new tool, Phyloformer 2.


## Contributors

This project combines shared evaluation pipelines with simulation datasets and experimental results contributed by the following team members.

| GitHub username | Contributions |
|---|---|
| [ACKarl](https://github.com/ACKarl) | Established the repository; contributed shared tree datasets and LG tree-size × sequence-length experiments; contributed simulation data, Phyloformer evaluations, and baseline results to later evaluation rounds. |
| [SwethaSirmalla](https://github.com/SwethaSirmalla) | Contributed and maintained the shared pipeline notebooks, setup scripts, helper functions, and Cherry model resources; contributed clock-rate × sequence-length experiments and data and results for later evaluation rounds. |
| [DarrenPimenta3113](https://github.com/DarrenPimenta3113) | Contributed LG clock-rate × tree-size experiments, corrected and combined result tables, and simulation data and evaluation results for later studies, including branch-scaling settings of 2.0 and 2.5. |
| [Dipu3023958](https://github.com/Dipu3023958) | Contributed Cherry tree-size × sequence-length datasets and results; contributed later data generation and Phyloformer evaluations at branch-scaling settings of 0.5 and 4.0, together with additional baseline evaluations. |
| [Ghostt98](https://github.com/Ghostt98) | Contributed Cherry clock-rate × tree-size experiments and combined parameter-grid results; contributed later simulation datasets, Phyloformer evaluations, and baseline results, including branch-scaling settings of 1.0 and 3.5. |
