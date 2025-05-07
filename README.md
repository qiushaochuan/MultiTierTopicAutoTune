# MultiTierTopicAutoTune
This repository implements a scalable, multi-tiered framework for automated parameter tuning of Latent Dirichlet Allocation (LDA) models on very large text corpora (tens of thousands of documents or more). It conducts three sequential rounds of topic extraction—each progressively refining topic granularity—while employing coherence-based grid search to identify optimal hyperparameters (number of topics, α, η) in the second and third rounds. Designed for efficiency and interpretability, the pipeline incorporates chunked training, document-level filtering, and comprehensive logging, enabling researchers to obtain robust topic hierarchies and quantitative estimates of topic prevalence in massive datasets.

Authors:
This framework was designed and developed by Qiu Shaochuan, Ding Yiyuan, and Jiang Yinjie.

Original Application:
It was originally applied to analyze forum data from oral cancer patients, providing insights into patient concerns, treatment experiences, and community discussion patterns.

Sample Data:
Example datasets for demonstration (forum posts, thread titles, and replies) have been uploaded to this repository under the sample_data/ directory.

Key Features
Three-Stage Topic Extraction

Round 1 uses expert-recommended default settings to generate an initial set of coarse-grained topics.
Round 2 and Round 3 perform automated grid searches over predefined hyperparameter grids—optimizing for coherence scores—to yield finer subtopics and sub-subtopics.
Scalable to Large Corpora

Processes data in chunks (default chunksize=1000) and prunes extreme-frequency words (no_below, no_above) to reduce memory footprint.
Supports batch JSON inputs; each file may contain thousands of forum posts or social media entries.
Automated Hyperparameter Tuning

Leverages gensim’s CoherenceModel to evaluate candidate LDA configurations.
Records all parameter combinations and coherence metrics in JSON-formatted sweep reports for reproducibility.
Interpretability & Reporting

Exports per-topic keyword lists, sampled documents, and topic–document assignment probabilities at each hierarchy level.
Generates an integrated Markdown report summarizing topic distributions and exemplar texts for immediate inspection.
Extensible & Configurable

Users can modify filtering thresholds (no_below, no_above), grid search spaces (param_grid_r2, param_grid_r3), and document-topic probability cutoffs (threshold) via top-level constants.
Logging follows Python’s standard logging module, with adjustable verbosity for debugging or batch execution.
