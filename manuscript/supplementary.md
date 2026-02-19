# Supplementary Information

**MAGICC: ultra-fast genome quality assessment using core gene k-mer profiles and deep learning**

---

## Table S1: Motivating analysis results

Results from the motivating analysis demonstrating limitations of existing tools. These experiments used exclusively NCBI-finished reference genomes as dominant genomes. MAGICC was not included in the motivating analysis because these experiments were designed to characterize the gap in existing tools that motivated MAGICC's development.

### Table S1a: Set A -- Controlled completeness (1,000 genomes, 0% contamination)

| Tool | Comp MAE | Cont MAE |
|------|:--------:|:--------:|
| CheckM2 | 2.54% | 0.28% |
| CoCoPyE | 3.47% | 0.73% |
| DeepCheck | 4.15% | 0.43% |

### Table S1b: Set B -- Controlled contamination (1,000 genomes, 100% completeness, cross-phylum)

| Tool | Comp MAE | Cont MAE | Cont MAE (>20% true) |
|------|:--------:|:--------:|:--------------------:|
| CheckM2 | 0.42% | 17.08% | 25.8% |
| CoCoPyE | 2.57% | 19.14% | 30.6% |
| DeepCheck | 6.74% | 25.70% | 38.6% |

### Table S1c: Set C -- Realistic mixed (1,000 genomes: 200 pure + 200 complete + 600 other)

| Tool | Comp MAE | Cont MAE | Comp RMSE | Cont RMSE | Comp R^2 | Cont R^2 |
|------|:--------:|:--------:|:---------:|:---------:|:--------:|:--------:|
| CheckM2 | 9.13% | 18.13% | 14.26% | 26.69% | 0.4962 | 0.5330 |
| CoCoPyE | 4.99% | 20.32% | 7.27% | 29.37% | 0.8170 | 0.7379 |
| DeepCheck | 11.96% | 24.37% | 17.19% | 35.46% | 0.4295 | 0.3307 |

At true contamination levels above 20%, all existing tools produce MAEs of 25.8--38.6%, meaning a genome with 60% true contamination might be reported as having only 20--30% contamination. This systematic underestimation motivated the development of MAGICC's k-mer-based approach.

---

## Table S2: Detailed benchmark results per set per tool

All benchmark sets used NCBI-finished reference genomes (assembly level "Complete Genome" or "Chromosome") as dominant genomes, providing clean ground truth with independently verified assemblies. Sets A_v2 and B_v2 are controlled gradient experiments; Sets C and D test challenging lineages; Set E tests realistic mixed conditions.

### Table S2a: Completeness prediction (MAE and R^2)

| Set | MAGICC MAE | MAGICC R^2 | CheckM2 MAE | CoCoPyE MAE | DeepCheck MAE |
|-----|:----------:|:----------:|:-----------:|:-----------:|:-------------:|
| A_v2 (n=1,000) | **1.99%** | 0.9585 | 2.54% | 3.63% | 4.26% |
| B_v2 (n=1,000) | **0.77%** | -- | **0.45%** | 2.61% | 6.61% |
| C (n=1,000) | **2.96%** | 0.9064 | 7.99% | 15.73% | 17.97% |
| D (n=1,000) | **4.06%** | 0.8098 | 9.89% | 6.14% | 9.97% |
| E (n=1,000) | **3.92%** | 0.8241 | 9.14% | 5.02% | 11.69% |

### Table S2b: Contamination prediction (MAE and R^2)

| Set | MAGICC MAE | MAGICC R^2 | CheckM2 MAE | CoCoPyE MAE | DeepCheck MAE |
|-----|:----------:|:----------:|:-----------:|:-----------:|:-------------:|
| A_v2 (n=1,000) | 0.46% | -- | **0.27%** | 0.77% | 0.40% |
| B_v2 (n=1,000) | **2.78%** | 0.9763 | 17.66% | 19.14% | 25.70% |
| C (n=1,000) | **5.05%** | 0.9390 | 42.37% | 34.07% | 44.56% |
| D (n=1,000) | **5.40%** | 0.9241 | 36.92% | 25.73% | 41.77% |
| E (n=1,000) | **4.32%** | 0.9490 | 18.47% | 21.60% | 25.43% |

### Table S2c: Overall results (5,000 genomes across all 5 sets)

| Metric | MAGICC | CheckM2 | CoCoPyE | DeepCheck |
|--------|:------:|:-------:|:-------:|:---------:|
| Comp MAE | **2.74%** | 6.00% | 6.62% | 10.10% |
| Cont MAE | **3.60%** | 23.14% | 20.26% | 27.57% |
| Comp R^2 | **0.90** | 0.67 | 0.65 | 0.48 |
| Cont R^2 | **0.96** | 0.34 | 0.65 | 0.18 |

Benchmark set descriptions: Set A_v2 -- completeness gradient (50--100%), 0% contamination, 1,000 genomes from finished references. Set B_v2 -- contamination gradient (0--80%, cross-phylum), 100% completeness, 1,000 genomes. Set C -- 1,000 Patescibacteria genomes, uniform completeness (50--100%) and contamination (0--100%). Set D -- 1,000 Archaea genomes, uniform completeness (50--100%) and contamination (0--100%). Set E -- 1,000 mixed genomes (200 pure + 200 complete + 600 other with 70% cross-phylum / 30% within-phylum contamination). Bold indicates best performance per metric. All improvements of MAGICC over competitor tools on contamination were statistically significant (paired Wilcoxon signed-rank test, p < 0.001).

---

## Table S3: MIMAG classification F1 scores

MIMAG quality classification performance on benchmark Sets C (Patescibacteria) and D (Archaea), which span the full range of completeness and contamination. F1 scores are macro-averaged across three MIMAG quality categories: high-quality (>=90% complete, <5% contaminated), medium-quality (>=50% complete, <10% contaminated), and low-quality (all others).

| Tool | MIMAG F1 (macro-averaged) |
|------|:-------------------------:|
| **MAGICC** | **0.89** |
| CoCoPyE | 0.42--0.71 |
| CheckM2 | 0.28--0.37 |
| DeepCheck | 0.21--0.27 |

MAGICC's substantially higher F1 score demonstrates that its improved contamination estimation translates directly to more reliable quality tier assignments. The low F1 scores of competitor tools are driven primarily by their systematic underestimation of contamination, which causes heavily contaminated genomes to be misclassified as high- or medium-quality.

---

## Table S4: Computational resource comparison

All tools were benchmarked on Set E (1,000 mixed genomes) using `/usr/bin/time -v` for peak memory measurement. Speed is reported as wall-clock time under each tool's typical multi-threaded configuration.

| Tool | Threads | Peak memory (GB) | Wall-clock time (Set E) | Genomes/min/thread |
|------|:-------:|:----------------:|:-----------------------:|:------------------:|
| **MAGICC** | **1** | **0.66** | **97.5 s** | **1,451** |
| CheckM2 | 32 | 18.76 | 86 min 37 s | 0.82 |
| CoCoPyE | 48 | 15.93 | 60 min 32 s | 0.70 |
| DeepCheck | 1* | 1.46* | 12 min 5 s* | 0.82** |

\* DeepCheck memory and wall-clock time reflect inference only; DeepCheck requires CheckM2's full pipeline (gene prediction, DIAMOND alignment, KEGG annotation) for feature extraction, which must be run first.

\** DeepCheck's effective end-to-end speed equals CheckM2's speed since feature extraction dominates runtime.

| Derived metric | MAGICC | CheckM2 | CoCoPyE |
|----------------|:------:|:-------:|:-------:|
| Memory ratio vs MAGICC | 1x | 28x | 24x |
| Speed ratio vs MAGICC (per thread) | 1x | 1/1,700x | 1/2,100x |
| Projected time for 100K genomes (1 thread) | ~69 min | ~87 hours | ~60+ hours |

MAGICC's speed advantage stems from eliminating gene prediction, protein alignment, and database search steps. K-mer counting uses a Numba-accelerated rolling hash (17.5 ms per 5 Mb genome), and model inference uses ONNX Runtime (0.18 ms per sample at batch size 1,024).

---

## Table S5: Reference genome filtering statistics

Genome metadata was downloaded from the Genome Taxonomy Database (GTDB) and filtered using strict quality criteria to obtain high-quality reference genomes for training data synthesis. All filters were applied conjunctively (all must pass):

- CheckM2 completeness >= 98%
- CheckM2 contamination <= 2%
- Contig count < 100
- N50 > 20 kbp
- Longest contig > 100 kbp

| Stage | Bacterial | Archaeal | Total |
|-------|:---------:|:--------:|:-----:|
| Starting genomes (GTDB) | 715,230 | 17,245 | 732,475 |
| **After all filters** | **275,207 (38.5%)** | **1,976 (11.5%)** | **277,183** |
| Phyla represented | | | 110 |
| Genome size (median) | | | 3.8 Mbp |
| Genome size (range) | | | 0.3--13.6 Mbp |
| N50 (median) | | | 270 kbp |
| Contig count (median) | | | 38 |
| **Selected for project** | 98,024 | 1,976 | 100,000 |
| Selection method | Square-root proportional stratified sampling | All included | |
| Successfully downloaded | | | 99,957 (99.96%) |
| **Train / Val / Test split** | | | |
| Training set | | | 79,948 |
| Validation set | | | 10,010 |
| Test set | | | 9,999 |

Selection ensured all available genomes from underrepresented lineages were included: Patescibacteriota (1,609), DPANN archaea (24), and candidate phyla (586). Genomes were split 80/10/10 by phylum with no overlap between splits.

---

## Table S6: K-mer feature selection statistics

K-mer features were derived from canonical 9-mers counted in single-copy core gene DNA sequences. Representative genomes were selected from the training set only (no data leakage). Core genes were identified using Prodigal v2.6.3 for gene prediction and HMMER 3.4 with trusted cutoff thresholds for HMM profile searching.

| Parameter | Bacterial | Archaeal |
|-----------|:---------:|:--------:|
| Representative genomes | 1,000 | 1,000 |
| Phyla represented | 97 | 13 |
| Core gene HMM profiles | 85 (TIGR/JCVI) | 128 (Pfam) |
| Core genes per genome (mean) | 83.3 | 127.9 |
| Core genes per genome (median) | 84 | 129 |
| Core genes per genome (range) | 50--89 | 80--138 |
| Total unique canonical 9-mers observed | 131,072 | 131,072 |
| K-mers per genome (mean) | 40,517 | 51,320 |
| K-mers per genome (median) | 41,638 | 52,755 |
| K-mers per genome (range) | 23,069--49,624 | 32,392--67,234 |
| **Selected k-mers** | **9,000** | **1,000** |
| Selection criterion | Top prevalence | Top prevalence |
| Prevalence range (of 1,000 genomes) | 529--992 | 791--998 |
| Overlap between bacterial and archaeal sets | 751 | 751 |
| **Final merged k-mer set** | **9,249 unique canonical 9-mers** | |
| Bacteria-only k-mers | 8,249 | -- |
| Archaea-only k-mers | -- | 249 |
| Shared k-mers | 751 | 751 |

Note: 131,072 = 4^9 / 2 represents all possible canonical 9-mers (since 9 is odd, each k-mer has a distinct reverse complement).

---

## Table S7: Synthetic training data composition

One million synthetic genomes were generated from 99,957 high-quality reference genomes. Each batch of 10,000 genomes followed the composition below, maintaining consistent ratios across training, validation, and test splits.

| Sample type | Per batch (of 10,000) | Percentage | Description |
|-------------|:---------------------:|:----------:|-------------|
| Pure genomes | 1,500 | 15% | 0% contamination, 50--100% completeness |
| Complete genomes | 1,500 | 15% | 100% completeness (original contigs), 0--100% contamination |
| Within-phylum contamination | 3,000 | 30% | 1--3 contaminant genomes from same phylum |
| Cross-phylum contamination | 3,000 | 30% | 1--5 contaminant genomes from different phyla |
| Reduced-genome organisms | 500 | 5% | Small-genome taxa (e.g., Patescibacteriota) |
| Archaeal genomes | 500 | 5% | Archaeal dominant genomes |
| **Total per batch** | **10,000** | **100%** | |

| Dataset split | Batches | Total genomes |
|---------------|:-------:|:-------------:|
| Training | 80 | 800,000 |
| Validation | 10 | 100,000 |
| Test | 10 | 100,000 |
| **Total** | **100** | **1,000,000** |

Genome fragmentation simulated four quality tiers: high (10--50 contigs, N50 100--500 kb), medium (50--200 contigs, N50 20--100 kb), low (200--500 contigs, N50 5--20 kb), and highly fragmented (500--2,000 contigs, N50 1--5 kb). Three bias mechanisms were applied: coverage-based dropout, GC-biased loss, and repeat region exclusion. Contamination levels were distributed among contaminant genomes using Dirichlet allocation.

---

## Table S8: Neural network training configuration

The MAGICCModelV3 architecture was developed through three training iterations (V1, V2, V3), with progressive improvements to architecture, regularization, and output activation.

### Table S8a: Architecture details

| Component | Configuration |
|-----------|---------------|
| **K-mer branch** | |
| Input | 9,249 k-mer counts |
| Layer 1 | Dense(4,096) -> BN -> SiLU -> Dropout(0.4) -> SE block |
| Layer 2 | Dense(1,024) -> BN -> SiLU -> Dropout(0.2) -> SE block |
| Layer 3 | Dense(256) -> BN -> SiLU |
| SE block | Squeeze-and-excitation attention (reduction ratio 16) |
| **Assembly branch** | |
| Input | 26 assembly statistics |
| Layer 1 | Dense(128) -> BN -> SiLU -> Dropout(0.2) |
| Layer 2 | Dense(64) -> BN -> SiLU |
| **Fusion head** | |
| Cross-attention | 4 heads; assembly embedding as query, k-mer embedding reshaped to 16 tokens |
| Gated residual | Learnable gate on cross-attention output |
| Output layers | Dense(128) -> BN -> SiLU -> Dropout(0.1) -> Dense(64) -> SiLU -> Dense(2) |
| **Output activation** | |
| Completeness | sigmoid(x) * 50 + 50, bounded [50, 100]% |
| Contamination | sigmoid(x) * 100, bounded [0, 100]% |
| **Total parameters** | 44,851,010 |

### Table S8b: Training hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1 x 10^-3 |
| Weight decay | 5 x 10^-4 |
| LR schedule | Cosine annealing warm restarts (T_0=10, T_mult=2) |
| Loss function | Weighted MSE: 2.0 x MSE(completeness) + 1.0 x MSE(contamination) |
| Batch size | 512 |
| Max epochs | 150 (early stopping patience=20) |
| Gradient clipping | max_norm = 1.0 |
| Mixed precision | FP16 with gradient checkpointing |
| Data augmentation | 2% random k-mer masking + Gaussian noise (sigma=0.01) |
| GPU | Quadro P2200 (5.1 GB VRAM) |
| ONNX export | Opset 17, FP32, dynamic batch sizes |

### Table S8c: Training results across model versions

| Metric | V1 | V2 | V3 (final) |
|--------|:--:|:--:|:----------:|
| Architecture | Dense + concat fusion | Dense + concat fusion | SE attention + cross-attention fusion |
| Parameters | 42,417,538 | 42,417,538 | 44,851,010 |
| Assembly features | 20 | 26 | 26 |
| Completeness output | sigmoid * 100 | sigmoid * 50 + 50 | sigmoid * 50 + 50 |
| Dropout (layer 1) | 0.3 | 0.4 | 0.4 |
| Weight decay | 1 x 10^-4 | 5 x 10^-4 | 5 x 10^-4 |
| Loss weights (comp:cont) | 1.0 : 1.5 | 2.0 : 1.0 | 2.0 : 1.0 |
| Total epochs (early stop) | 87 | 87 | 84 |
| Best epoch | 67 | 67 | 64 |
| Training time | 4.7 hours | 4.5 hours | ~4 hours |
| Val completeness MAE | 4.77% | 3.93% | **3.77%** |
| Val contamination MAE | 4.51% | 4.49% | **4.35%** |
| Val completeness R^2 | 0.755 | 0.828 | **0.833** |
| Val contamination R^2 | 0.937 | 0.945 | **0.946** |

Key changes from V1 to V2: fixed "complete" sample type (was generating 86% mean completeness instead of 100%), changed output activation for completeness from sigmoid * 100 to sigmoid * 50 + 50 (using full sigmoid range), added 6 k-mer summary features (expanding assembly features from 20 to 26), increased regularization, and rebalanced loss weights to prioritize completeness. Key change from V2 to V3: added squeeze-and-excitation attention blocks and cross-attention fusion mechanism.

---

## Supplementary Figures

**Figure S1. Per-tool predicted versus true quality scores for benchmark Set A (completeness gradient).** (a) Predicted versus true completeness for MAGICC, CheckM2, CoCoPyE, and DeepCheck. (b) Predicted versus true contamination. Set A contains 1,000 genomes with varying completeness (50--100%) and 0% contamination, derived from NCBI-finished reference genomes.

**Figure S2. Per-tool predicted versus true quality scores for benchmark Set B (contamination gradient).** (a) Predicted versus true completeness for MAGICC, CheckM2, CoCoPyE, and DeepCheck. (b) Predicted versus true contamination. Set B contains 1,000 genomes at 100% completeness with varying cross-phylum contamination (0--80%).

**Figure S3. Per-tool predicted versus true quality scores for benchmark Set C (Patescibacteria).** (a) Predicted versus true completeness for MAGICC, CheckM2, CoCoPyE, and DeepCheck. (b) Predicted versus true contamination. Set C contains 1,000 Patescibacteria genomes with uniform completeness (50--100%) and contamination (0--100%), testing performance on reduced-genome organisms.

**Figure S4. Per-tool predicted versus true quality scores for benchmark Set D (Archaea).** (a) Predicted versus true completeness for MAGICC, CheckM2, CoCoPyE, and DeepCheck. (b) Predicted versus true contamination. Set D contains 1,000 archaeal genomes with uniform completeness (50--100%) and contamination (0--100%), testing performance on underrepresented taxonomic lineages.

**Figure S5. Per-tool predicted versus true quality scores for benchmark Set E (realistic mixed).** (a) Predicted versus true completeness for MAGICC, CheckM2, CoCoPyE, and DeepCheck. (b) Predicted versus true contamination. Set E contains 1,000 genomes with varied completeness (50--100%), contamination (0--100%), and fragmentation levels, representative of the full training distribution.
