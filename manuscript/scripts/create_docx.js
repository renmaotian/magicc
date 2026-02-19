const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun,
  Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, PageNumber,
  BorderStyle, WidthType, ShadingType, VerticalAlign, PageBreak,
  FootnoteReferenceRun, Footnote
} = require("docx");

const figDir = path.join(__dirname, "..", "figures");
const outPath = path.join(__dirname, "..", "manuscript.docx");

// Helper: read image and get dimensions
function readImg(name) {
  return fs.readFileSync(path.join(figDir, name));
}

// Helper: create a normal paragraph
function p(texts, opts = {}) {
  const children = [];
  for (const t of texts) {
    if (typeof t === "string") children.push(new TextRun(t));
    else children.push(new TextRun(t));
  }
  return new Paragraph({ children, spacing: { after: 120 }, ...opts });
}

// Helper: bold text run
function b(text) { return { text, bold: true }; }
// Helper: italic text run
function i(text) { return { text, italics: true }; }
// Helper: superscript text run
function sup(text) { return { text, superScript: true }; }

// Helper: create a figure with caption
function figure(imgFile, imgType, width, height, captionRuns) {
  return [
    new Paragraph({ children: [new PageBreak()] }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 240, after: 120 },
      children: [
        new ImageRun({
          type: imgType,
          data: readImg(imgFile),
          transformation: { width, height },
          altText: { title: imgFile, description: imgFile, name: imgFile }
        })
      ]
    }),
    new Paragraph({
      spacing: { before: 120, after: 240 },
      children: captionRuns
    })
  ];
}

// Build document content
const children = [];

// ===== TITLE =====
children.push(new Paragraph({
  heading: HeadingLevel.TITLE,
  spacing: { before: 600, after: 240 },
  children: [new TextRun({ text: "MAGICC: ultra-fast genome quality assessment using core gene k-mer profiles and deep learning", bold: true, size: 32 })]
}));

// ===== ABSTRACT =====
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Abstract")] }));

children.push(p([
  "Metagenome-assembled genomes (MAGs) are central to culture-independent microbiology, yet quality assessment remains a critical bottleneck. Existing tools such as CheckM2, CoCoPyE, and DeepCheck achieve reasonable completeness estimation but severely underestimate contamination\u2014particularly when contaminant sequences originate from phylogenetically distant organisms. Using NCBI-finished reference genomes with known ground truth, we show that at true contamination levels above 20%, these tools produce mean absolute errors (MAEs) of 26\u201339%, rendering contamination estimates unreliable for the majority of real-world MAGs. Here we present MAGICC (Metagenome-Assembled Genome Inference of Completeness and Contamination), which replaces gene annotation with core gene k-mer profiling to capture the compositional shift that foreign DNA introduces. MAGICC combines 9,249 canonical 9-mer counts derived from bacterial and archaeal core genes with 26 assembly statistics in a dual-branch neural network featuring squeeze-and-excitation attention and cross-attention fusion. Trained on one million synthetic genomes spanning 110 phyla, MAGICC achieves 2.74% completeness MAE and 3.60% contamination MAE across five benchmark sets\u2014a 6.4-fold improvement in contamination accuracy over CheckM2. MAGICC processes genomes at 1,451 genomes per minute per thread (~1,700\u00d7 faster than CheckM2), requires only 0.66 GB of memory (28\u00d7 less than CheckM2), and is available as a single-command Python tool."
]));

// ===== INTRODUCTION =====
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Introduction")] }));

children.push(p([
  "The reconstruction of genomes from metagenomic sequencing data has transformed microbiology, enabling the discovery of thousands of previously unknown bacterial and archaeal lineages",
  sup("1,2"),
  ". These metagenome-assembled genomes (MAGs) are generated through computational binning of assembled contigs, a process that is inherently imperfect. The resulting genomes exhibit varying degrees of completeness\u2014the fraction of the true genome that was recovered\u2014and contamination\u2014the inclusion of sequences from other organisms. Accurate quality assessment is essential before downstream analyses, and the Minimum Information about a Metagenome-Assembled Genome (MIMAG) standards classify MAGs into high-quality (\u226590% complete, <5% contaminated), medium-quality (\u226550% complete, <10% contaminated), and low-quality categories",
  sup("3"),
  "."
]));

children.push(p([
  "The current standard for MAG quality assessment is CheckM2",
  sup("4"),
  ", which predicts completeness and contamination using gradient boosted decision trees and neural networks trained on KEGG ortholog (KO) annotation features. CheckM2 represented a major advance over its predecessor CheckM",
  sup("5"),
  ", which relied on lineage-specific marker gene sets and phylogenetic placement. However, CheckM2's pipeline requires gene prediction with Prodigal",
  sup("6"),
  ", protein alignment with DIAMOND",
  sup("7"),
  " against the UniRef100 database, and subsequent KO annotation\u2014a computationally expensive process that yields approximately 0.8 genomes per minute per thread and demands ~19 GB of peak memory. For the rapidly growing repositories of MAGs\u2014the gcMeta database now contains over 2.7 million MAGs",
  sup("8"),
  "\u2014this translates to processing times measured in months rather than hours."
]));

children.push(p([
  "More fundamentally, functional annotation features have an intrinsic blind spot for contamination detection. CheckM2's contamination model learns from the pattern of KO annotations in a genome; when foreign DNA from a different phylum is present, the contaminant contributes its own set of KO annotations that may not be recognizable as anomalous. Unlike the traditional marker-duplication approach of CheckM (which detects contamination through duplicate single-copy genes), functional annotations lack an inherent mechanism for detecting the ",
  i("presence"),
  " of non-self sequences\u2014they only capture ",
  i("what functions"),
  " are encoded, not ",
  i("which organism"),
  " they came from. When contaminant genes encode functions already present in the host genome or belong to broadly conserved pathways, the functional profile appears normal despite substantial contamination. This is particularly problematic for cross-phylum contamination, where contaminant and host genomes share few orthologous genes."
]));

children.push(p([
  "Other tools face similar limitations. CoCoPyE",
  sup("9"),
  " uses protein domain profiles (Pfam) for feature extraction, inheriting the same functional-annotation blind spot. DeepCheck",
  sup("10"),
  " applies a ResNet architecture to CheckM2's intermediate feature vectors, remaining dependent on CheckM2's annotation pipeline and thus inheriting both its computational cost and its feature limitations. GUNC",
  sup("11"),
  " takes a fundamentally different approach by analyzing the taxonomic consistency of individual contigs, but operates as a contamination detector (binary flag) rather than providing quantitative estimates."
]));

children.push(p([
  "To quantify this gap, we evaluated CheckM2, CoCoPyE, and DeepCheck on synthetic genomes derived exclusively from NCBI-finished reference genomes, where the true completeness and contamination are known by construction. While all three tools achieved reasonable completeness prediction (MAE 2.5\u20134.2%), their contamination estimates deteriorated dramatically with increasing contamination levels: at true contamination above 20%, MAEs ranged from 26% to 39% (",
  b("Fig. 1b"),
  "). This systematic underestimation means that a genome with 60% true contamination might be reported as having only 20\u201330%\u2014a difference that could fundamentally alter downstream biological conclusions."
]));

children.push(p([
  "We reasoned that k-mer composition could address this gap. Genomic k-mer profiles carry rich taxonomic signal\u2014a principle exploited by metagenomic binning tools such as MetaBAT2",
  sup("12"),
  " and CONCOCT",
  sup("13"),
  ", which cluster contigs by tetranucleotide frequency. When foreign DNA is introduced into a genome, the k-mer profile shifts measurably because different organisms have distinct nucleotide composition patterns shaped by GC content, codon usage bias, and evolutionary divergence. Critically, k-mer counting requires only a single pass through the sequence data, without gene prediction, alignment, or database searches."
]));

children.push(p([
  "Here we present MAGICC (Metagenome-Assembled Genome Inference of Completeness and Contamination), which combines core gene k-mer profiles with assembly statistics in a dual-branch neural network to predict genome quality. By selecting 9,249 canonical 9-mers from bacterial and archaeal core genes and pairing them with 26 assembly statistics that capture structural signatures of contamination (e.g., GC bimodality, GC outlier fraction), MAGICC achieves accurate predictions across the full range of completeness (50\u2013100%) and contamination (0\u2013100%), processes genomes three orders of magnitude faster than existing tools, and requires minimal computational resources."
]));

// ===== RESULTS =====
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Results")] }));

// --- Results subsection 1 ---
children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Existing tools systematically underestimate contamination from foreign organisms")] }));

children.push(p([
  "To establish the need for a new approach, we first benchmarked three current tools\u2014CheckM2 v1.0.1, CoCoPyE v0.5.0, and DeepCheck\u2014on controlled synthetic genomes derived from 1,810 NCBI-finished reference genomes (assembly level \u201cComplete Genome\u201d or \u201cChromosome\u201d) from 35 phyla (",
  b("Fig. 1"),
  "). Using finished genomes as the ground truth eliminates any circularity in quality assessment, as these genomes have independently verified assemblies."
]));

children.push(p([
  "In a completeness gradient experiment (1,000 genomes, 0% contamination, completeness spanning 50\u2013100%), all tools performed reasonably: CheckM2 achieved 2.54% MAE, CoCoPyE 3.47%, and DeepCheck 4.15% (",
  b("Fig. 1a"),
  "). However, in a contamination gradient experiment (1,000 genomes, 100% complete, contamination spanning 0\u201380% from cross-phylum sources), performance collapsed: CheckM2's contamination MAE rose to 17.08%, CoCoPyE to 19.14%, and DeepCheck to 25.70% (",
  b("Fig. 1b"),
  "). At contamination levels above 20%, the tools produced MAEs of 25.8%, 30.6%, and 38.6%, respectively. In a realistic mixed set of 1,000 genomes combining varied completeness and contamination levels, contamination MAE remained high across all tools (18.1\u201324.4%) (",
  b("Fig. 1c"),
  ")."
]));

children.push(p([
  "The speed of existing tools further limits their applicability. All three tools processed fewer than 1 genome per minute per thread (",
  b("Fig. 1d"),
  "), meaning that assessing 100,000 genomes\u2014a routine scale for modern metagenomic studies",
  sup("8"),
  "\u2014would require weeks of computation on standard hardware."
]));

// --- FIGURE 1 ---
children.push(...figure("figure1_motivating.png", "png", 650, 500, [
  new TextRun({ text: "Figure 1. Existing tools systematically underestimate contamination from foreign organisms. ", bold: true }),
  new TextRun("("),
  new TextRun({ text: "a", bold: true }),
  new TextRun(") Completeness prediction accuracy on 1,000 synthetic genomes from NCBI-finished references with 0% contamination at six completeness levels. All tools achieve reasonable accuracy (MAE 2.5\u20134.2%). ("),
  new TextRun({ text: "b", bold: true }),
  new TextRun(") Contamination prediction on 1,000 genomes with 100% completeness at five contamination levels (0\u201380%, cross-phylum). Predicted values (y-axis) dramatically diverge from true values (x-axis) at higher contamination levels. ("),
  new TextRun({ text: "c", bold: true }),
  new TextRun(") Realistic mixed benchmark (1,000 genomes). Scatter plots of predicted versus true contamination for each tool. ("),
  new TextRun({ text: "d", bold: true }),
  new TextRun(") Processing speed of existing tools in genomes per minute per thread. All tools process fewer than 1 genome/min/thread.")
]));

// --- Results subsection 2 ---
children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("MAGICC overview")] }));

children.push(p([
  "MAGICC takes a FASTA genome file as input and produces completeness and contamination predictions through four steps (",
  b("Fig. 2"),
  "). First, the genome sequence is scanned in a single pass using a Numba-accelerated rolling hash to count 9,249 pre-selected canonical 9-mers derived from bacterial core genes (85 single-copy marker families) and archaeal core genes (128 marker families). These k-mers were selected by prevalence across 1,000 representative bacterial and 1,000 representative archaeal genomes from the GTDB",
  sup("14"),
  " training set, capturing k-mers present in the broadest range of organisms. Second, 26 assembly statistics are computed, including 11 contig length metrics (N50, L50, contig count, etc.), 4 GC composition features (mean, standard deviation, interquartile range, bimodality index), 4 distributional features (GC outlier fraction, largest contig fraction, top-10% concentration, N50/mean ratio), 1 total k-mer count feature, and 6 k-mer summary statistics (unique k-mer count, duplicate ratio, Shannon entropy, etc.). Third, features are normalized using pre-computed statistics (Z-score for k-mers, log-transform for lengths, min-max for proportions, robust scaling for counts). Fourth, the normalized features are passed through a dual-branch neural network: the k-mer branch (9,249\u21924,096\u21921,024\u2192256 with squeeze-and-excitation attention blocks after each dense layer) and the assembly branch (26\u2192128\u219264) are fused via cross-attention (assembly embedding queries k-mer tokens) into a final prediction head (320\u2192128\u219264\u21922), outputting completeness [50\u2013100%] and contamination [0\u2013100%]."
]));

children.push(p([
  "The model was trained on one million synthetic genomes generated from 100,000 high-quality reference genomes (\u226598% CheckM2 completeness, \u22642% contamination, <100 contigs, N50 >20 kbp) spanning 110 phyla from the GTDB. Training genomes were synthesized with realistic fragmentation patterns (four quality tiers with coverage dropout, GC-biased loss, and repeat exclusion), contamination from within-phylum (1\u20133 genomes) and cross-phylum (1\u20135 genomes) sources distributed via Dirichlet allocation, and completeness ranging from 50% to 100%. The model was trained with mixed-precision FP16, AdamW optimizer (learning rate 1\u00d710",
  sup("\u22123"),
  ", weight decay 5\u00d710",
  sup("\u22124"),
  "), cosine annealing warm restarts, and a weighted MSE loss (2\u00d7 completeness weight) for 87 epochs with early stopping."
]));

// --- FIGURE 2 ---
children.push(...figure("figure2_architecture.png", "png", 650, 310, [
  new TextRun({ text: "Figure 2. MAGICC architecture overview. ", bold: true }),
  new TextRun("Input FASTA files are processed through a k-mer counting module (9,249 canonical 9-mers from core genes via Numba-accelerated rolling hash) and an assembly statistics module (26 features). These feed into a dual-branch neural network: the k-mer branch with squeeze-and-excitation attention and the assembly branch, fused via cross-attention where assembly features query k-mer tokens. Output predictions: completeness [50\u2013100%] and contamination [0\u2013100%].")
]));

// --- Results subsection 3 ---
children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("MAGICC achieves accurate predictions across diverse benchmarks")] }));

children.push(p([
  "We evaluated MAGICC on five benchmark sets totaling 5,000 genomes, all derived from finished reference genomes not used in training (",
  b("Fig. 3"),
  ", ",
  b("Table 1"),
  "). Set A (1,000 genomes, completeness gradient, 0% contamination) tested completeness prediction in isolation: MAGICC achieved 1.99% MAE, comparable to CheckM2 (2.54%). Set B (1,000 genomes, 100% complete, contamination gradient 0\u201380%) tested contamination prediction: MAGICC achieved 2.78% MAE, a dramatic improvement over CheckM2 (17.66%), CoCoPyE (19.14%), and DeepCheck (25.70%). Set C (1,000 Patescibacteria genomes) and Set D (1,000 Archaea genomes) tested challenging lineages with reduced genomes and underrepresented taxa: MAGICC maintained strong performance (Set C: 2.96%/5.05% completeness/contamination MAE; Set D: 4.06%/5.40%), while competitor tools produced contamination MAEs of 36\u201345%. Set E (1,000 mixed genomes with realistic composition) confirmed MAGICC's robustness (3.92%/4.32% MAE)."
]));

children.push(p([
  "Across all five benchmark sets, MAGICC achieved overall completeness MAE of 2.74% (R\u00b2=0.90) and contamination MAE of 3.60% (R\u00b2=0.96), compared to CheckM2's 6.00%/23.14%, CoCoPyE's 6.62%/20.26%, and DeepCheck's 10.10%/27.57% (",
  b("Table 1"),
  "). The improvement was most striking for contamination, where MAGICC's MAE was 6.4\u00d7 lower than CheckM2's. All improvements were statistically significant (paired Wilcoxon signed-rank test, p<0.001; 1,000-replicate bootstrap 95% confidence intervals non-overlapping)."
]));

children.push(p([
  "For MIMAG quality classification on Sets C and D, MAGICC achieved macro-averaged F1 scores of 0.89, compared to 0.28\u20130.37 for CheckM2, 0.42\u20130.71 for CoCoPyE, and 0.21\u20130.27 for DeepCheck. This demonstrates that MAGICC's improved contamination estimation translates directly to more reliable quality tier assignments."
]));

// --- FIGURE 3 ---
children.push(...figure("figure3_benchmark.png", "png", 650, 420, [
  new TextRun({ text: "Figure 3. MAGICC benchmark performance across five evaluation sets. ", bold: true }),
  new TextRun("("),
  new TextRun({ text: "a", bold: true }),
  new TextRun(","),
  new TextRun({ text: "b", bold: true }),
  new TextRun(") MAGICC predicted versus true values for completeness and contamination across all benchmark sets (n=5,000), colored by set. ("),
  new TextRun({ text: "c", bold: true }),
  new TextRun(") CheckM2 predicted versus true contamination for comparison. ("),
  new TextRun({ text: "d", bold: true }),
  new TextRun(","),
  new TextRun({ text: "e", bold: true }),
  new TextRun(") Per-set MAE comparison across four tools for completeness and contamination. ("),
  new TextRun({ text: "f", bold: true }),
  new TextRun(") Overall MAE with 95% bootstrap confidence intervals.")
]));

// --- Results subsection 4 ---
children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Why k-mer features detect foreign contamination")] }));

children.push(p([
  "The fundamental advantage of k-mer features over functional annotation features lies in their ability to capture compositional identity rather than functional identity. When a contaminant genome from a different phylum is introduced, it brings a distinct nucleotide composition\u2014reflected in its k-mer profile\u2014regardless of whether its genes encode functions already present in the host. Consider a Pseudomonadota genome contaminated with Bacillota sequences: the contaminant contributes k-mers characteristic of its own GC content, codon usage, and taxonomic signature, shifting the aggregate k-mer profile away from what a pure Pseudomonadota genome would produce. The neural network learns to associate such profile distortions with contamination."
]));

children.push(p([
  "This is complemented by the assembly statistics branch, which captures structural signatures of contamination. GC bimodality increases when sequences from organisms with different GC content are mixed; the GC outlier fraction rises when contaminant contigs have unusual composition relative to the dominant genome; and k-mer summary statistics (entropy, unique ratio) shift because contamination introduces k-mers not present in the host's core gene set. The cross-attention fusion mechanism enables the model to use assembly-level context (e.g., GC distribution) to modulate which k-mer patterns are most informative for a given genome."
]));

children.push(p([
  "By contrast, CheckM2's KEGG-based features capture the ",
  i("functional repertoire"),
  " of a genome. Because KEGG orthologs are defined by function rather than taxonomic origin, a contaminant gene encoding a broadly conserved function (e.g., ribosomal protein, DNA polymerase) produces the same feature value as the native gene. The contamination model must therefore rely on detecting ",
  i("excess"),
  " functions or unusual functional profiles\u2014a much weaker signal than the compositional shift captured by k-mers. CheckM2's training data also primarily simulated contamination through duplicate contig sampling (where contamination manifests as duplicate markers), rather than through addition of foreign genomic material. This training regime means the model was never explicitly taught to recognize cross-phylum contamination patterns at high levels."
]));

// --- Results subsection 5 ---
children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("MAGICC is orders of magnitude faster and lighter")] }));

children.push(p([
  "MAGICC's design eliminates the gene prediction, protein alignment, and database search steps that dominate the runtime of existing tools. On the benchmark Set E (1,000 genomes), MAGICC processed genomes at 1,451 genomes per minute per thread\u2014approximately 1,700\u00d7 faster than CheckM2 (0.82 genomes/min/thread) and 2,100\u00d7 faster than CoCoPyE (0.70 genomes/min/thread) (",
  b("Fig. 4a"),
  "). This speed advantage is achieved through Numba-accelerated k-mer counting with a rolling hash (17.5 ms per 5 Mb genome) and ONNX Runtime model inference (0.18 ms per sample at batch 1,024). Multi-threaded execution (43 threads) achieved 7,559 genomes per minute."
]));

children.push(p([
  "Peak memory usage was 0.66 GB for MAGICC versus 18.76 GB for CheckM2 and 15.93 GB for CoCoPyE (",
  b("Fig. 4b"),
  ")\u2014a 28\u00d7 reduction. CheckM2's memory is dominated by the DIAMOND alignment against the UniRef100 database. For processing 100,000 genomes, MAGICC would require approximately 69 minutes on a single thread or 13 minutes with 43 threads, compared to approximately 87 hours for CheckM2 (32 threads) or 60 hours for CoCoPyE (48 threads) (",
  b("Fig. 4c"),
  ")."
]));

// --- FIGURE 4 ---
children.push(...figure("figure4_speed.png", "png", 650, 260, [
  new TextRun({ text: "Figure 4. Computational resource comparison. ", bold: true }),
  new TextRun("("),
  new TextRun({ text: "a", bold: true }),
  new TextRun(") Processing speed in genomes per minute per thread (log scale). MAGICC: 1,451; CheckM2: 0.82; CoCoPyE: 0.70. ("),
  new TextRun({ text: "b", bold: true }),
  new TextRun(") Peak memory usage in GB. MAGICC: 0.66; CheckM2: 18.76; CoCoPyE: 15.93. ("),
  new TextRun({ text: "c", bold: true }),
  new TextRun(") Projected wall-clock time for processing 100,000 genomes under typical configurations.")
]));

// --- Results subsection 6 ---
children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Large-scale validation on 100,000 synthetic genomes")] }));

children.push(p([
  "To assess MAGICC's robustness at scale, we evaluated it on 100,000 held-out test genomes spanning six sample types: pure genomes (15,000), complete genomes with contamination (15,000), cross-phylum contamination (30,000), within-phylum contamination (30,000), archaeal genomes (5,000), and reduced-genome organisms (5,000) (",
  b("Fig. 5"),
  "). Overall metrics were 3.81% completeness MAE (R\u00b2=0.83) and 4.37% contamination MAE (R\u00b2=0.95). Performance was strongest on pure genomes (completeness MAE 2.21%, contamination MAE 0.36%) and complete genomes (completeness MAE 0.97%, contamination MAE 3.82%), and weakest on within-phylum contamination (completeness MAE 5.39%, contamination MAE 6.36%) and reduced-genome organisms (completeness MAE 6.31%, contamination MAE 6.69%). The higher error for within-phylum contamination is expected, as contaminant genomes from the same phylum have more similar k-mer profiles to the host, making compositional shifts harder to detect. Inference completed in 146 seconds for all 100,000 genomes (41,022 genomes/min)."
]));

// --- FIGURE 5 ---
children.push(...figure("figure5_100k.png", "png", 650, 500, [
  new TextRun({ text: "Figure 5. Large-scale validation on 100,000 test genomes. ", bold: true }),
  new TextRun("("),
  new TextRun({ text: "a", bold: true }),
  new TextRun(","),
  new TextRun({ text: "b", bold: true }),
  new TextRun(") Predicted versus true values for completeness and contamination, colored by sample type. ("),
  new TextRun({ text: "c", bold: true }),
  new TextRun(") Per-sample-type MAE for both metrics. ("),
  new TextRun({ text: "d", bold: true }),
  new TextRun(") Per-sample-type R\u00b2 values. Within-phylum contamination and reduced-genome organisms show highest error, as expected from their inherently more challenging compositional signals.")
]));

// ===== DISCUSSION =====
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Discussion")] }));

children.push(p([
  "We have presented MAGICC, a tool that addresses two critical limitations of existing MAG quality assessment methods: inaccurate contamination estimation and prohibitive computational cost. By replacing functional annotation with core gene k-mer profiling, MAGICC achieves a 6.4-fold reduction in contamination MAE relative to CheckM2 while processing genomes ~1,700\u00d7 faster with 28\u00d7 less memory."
]));

children.push(p([
  "The key insight underlying MAGICC is that contamination fundamentally alters the nucleotide composition of a genome, and this compositional shift is readily captured by k-mer profiles. Existing tools that rely on functional annotations\u2014whether KEGG orthologs (CheckM2, DeepCheck) or Pfam domains (CoCoPyE)\u2014operate in a feature space where taxonomically distant organisms may appear similar, because function is more conserved than composition across the tree of life. K-mer profiles, by contrast, carry inherent taxonomic signal: organisms within the same phylum share more similar k-mer distributions than organisms across phyla, a principle well-established in metagenomic binning",
  sup("12,13"),
  " and taxonomic classification",
  sup("15"),
  "."
]));

children.push(p([
  "The choice of 9-mers from core genes rather than whole-genome k-mers was deliberate. Core genes are present across virtually all bacteria and archaea, providing a universal feature space that does not require prior knowledge of the query genome's taxonomy. By selecting the 9,249 most prevalent k-mers across a phylogenetically diverse panel of 2,000 reference genomes, we ensure broad applicability while keeping the feature vector compact enough for efficient neural network inference. The use of raw counts rather than frequencies is also important: raw counts reflect both completeness (fewer counts when genome is incomplete) and contamination (additional counts from foreign k-mers), whereas frequencies are confounded by genome size."
]));

children.push(p([
  "MAGICC has several limitations. First, within-phylum contamination is harder to detect than cross-phylum contamination (6.36% vs 4.07% contamination MAE on the 100K test set), because closely related organisms share more similar k-mer profiles. This is an inherent limitation of any composition-based approach. Second, the model was trained on synthetic genomes, which may not capture all the complexities of real metagenomic binning artifacts, such as chimeric contigs or strain-level mixing. Third, completeness prediction accuracy (R\u00b2=0.83 on the 100K test set) has room for improvement; this likely reflects the weak individual correlation of k-mer features with completeness (max |r|=0.17), suggesting that fundamentally different features (e.g., gene presence/absence) may be needed for further gains. Fourth, MAGICC currently supports bacterial and archaeal genomes; extension to other domains (e.g., viral, eukaryotic) would require domain-specific core gene sets and retraining."
]));

children.push(p([
  "Despite these limitations, MAGICC's combination of accuracy, speed, and resource efficiency makes it immediately practical for large-scale genomic surveillance, population-level microbiome studies, and real-time quality assessment pipelines. As metagenomic databases continue to grow\u2014gcMeta now contains over 2.7 million MAGs",
  sup("8"),
  " and GTDB over 730,000 genomes",
  sup("14"),
  "\u2014the ability to assess genome quality in seconds rather than hours becomes not merely convenient but essential."
]));

// ===== METHODS =====
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Methods")] }));

// Methods subsections
children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Reference genome curation")] }));
children.push(p([
  "We downloaded metadata for all 732,475 genomes in GTDB (715,230 bacterial, 17,245 archaeal) and applied strict quality filters: CheckM2 completeness \u226598%, contamination \u22642%, contig count <100, N50 >20 kbp, and longest contig >100 kbp. This yielded 277,183 high-quality reference genomes (275,207 bacterial, 1,976 archaeal) across 110 phyla. From these, 100,000 genomes were selected using square-root proportional stratified sampling across phyla to balance representation, with all available genomes included for underrepresented lineages (Patescibacteriota: 1,609; DPANN archaea: 24; candidate phyla: 586). Genomes were downloaded using the NCBI datasets CLI v18.16.0, achieving 99.96% success (99,957 of 100,000). The dataset was split into training (79,948), validation (10,010), and test (9,999) sets with stratified 80/10/10 allocation by phylum and no overlapping genomes."
]));

children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("K-mer feature selection")] }));
children.push(p([
  "From the training set, 1,000 representative bacterial genomes (from 97 phyla) and 1,000 representative archaeal genomes (from 13 phyla) were selected by stratified sampling. For each genome, genes were predicted using Prodigal v2.6.3 and searched against 85 bacterial core gene HMM profiles (single-copy genes present in \u226595% of bacteria) and 128 archaeal core gene HMM profiles using HMMER 3.4 with trusted cutoff thresholds. Core gene DNA sequences were extracted and canonical 9-mers were counted using KMC 3.2.2. The 9,000 most prevalent bacterial k-mers (present in 529\u2013992 of 1,000 genomes) and 1,000 most prevalent archaeal k-mers (present in 791\u2013998 of 1,000 genomes) were merged into a final set of 9,249 unique canonical 9-mers (751 shared between domains)."
]));

children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Synthetic genome generation")] }));
children.push(p([
  "One million synthetic genomes (800,000 training, 100,000 validation, 100,000 test) were generated with the following composition per batch of 10,000: 1,500 pure genomes (0% contamination), 1,500 complete genomes (100% completeness with varying contamination), 3,000 within-phylum contamination, 3,000 cross-phylum contamination, 500 reduced-genome organisms, and 500 archaeal genomes."
]));
children.push(p([
  "Genome fragmentation simulated four quality tiers: high (10\u201350 contigs, N50 100\u2013500 kb), medium (50\u2013200 contigs, N50 20\u2013100 kb), low (200\u2013500 contigs, N50 5\u201320 kb), and highly fragmented (500\u20132,000 contigs, N50 1\u20135 kb). Contig lengths were drawn from log-normal distributions (\u03bc=log(N50), \u03c3\u2208[0.8, 1.2]). Three bias mechanisms were applied before completeness filtering: coverage-based dropout (log-normal coverage model, threshold 5\u00d7), GC-biased loss (probability proportional to |z-score| of contig GC content), and repeat exclusion (targeting high-homopolymer contigs). Target completeness was achieved by greedily accumulating contigs from smallest to largest."
]));
children.push(p([
  "Contamination was introduced by fragmenting 1\u20135 contaminant genomes independently and appending their contigs. Contamination rate was defined as (total contaminant bp / dominant genome full reference length) \u00d7 100, ensuring independence from completeness. Target contamination was allocated among contaminant genomes using Dirichlet distribution. Multi-copy support allowed contamination up to 100% of the reference genome size."
]));

children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Feature extraction and normalization")] }));
children.push(p([
  "For each synthetic or benchmark genome, features were extracted in two parallel streams. K-mer counts were obtained using a Numba JIT-compiled rolling hash over the genome sequence, encoding each 9-mer as a bit-packed integer and using canonical form (minimum of forward and reverse complement codes) with a lookup table for the 9,249 selected k-mers. Assembly statistics (26 features) were computed using Numba-accelerated functions for GC content, Nx/Lx metrics, and distributional properties."
]));
children.push(p([
  "Features were normalized using streaming statistics computed from the training set: k-mers were log(count+1)-transformed and Z-score standardized; length-based assembly features were log\u2081\u2080-transformed; proportion-based features were min-max scaled; and count-based features were robustly scaled using median and interquartile range from reservoir sampling."
]));

children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Neural network architecture and training")] }));
children.push(p([
  "The MAGICCModelV3 architecture comprises three components. The k-mer branch processes the 9,249-dimensional input through Dense(4,096)\u2192BN\u2192SiLU\u2192Dropout(0.4)\u2192SE\u2192Dense(1,024)\u2192BN\u2192SiLU\u2192Dropout(0.2)\u2192SE\u2192Dense(256)\u2192BN\u2192SiLU, where SE denotes squeeze-and-excitation attention blocks (reduction ratio 16). The assembly branch processes 26 features through Dense(128)\u2192BN\u2192SiLU\u2192Dropout(0.2)\u2192Dense(64)\u2192BN\u2192SiLU. The fusion head applies cross-attention (4 heads, assembly embedding as query, k-mer embedding reshaped into 16 tokens) with a gated residual connection, followed by Dense(128)\u2192BN\u2192SiLU\u2192Dropout(0.1)\u2192Dense(64)\u2192SiLU\u2192Dense(2). Outputs are bounded via sigmoid: completeness = \u03c3(x)\u00d750+50 \u2208 [50,100], contamination = \u03c3(x)\u00d7100 \u2208 [0,100]. Total parameters: 44,851,010."
]));
children.push(p([
  "Training used mixed-precision FP16 with gradient checkpointing on a Quadro P2200 GPU (5.1 GB VRAM). The optimizer was AdamW (lr=1\u00d710",
  sup("\u22123"),
  ", weight_decay=5\u00d710",
  sup("\u22124"),
  ") with cosine annealing warm restarts (T\u2080=10, T_mult=2). The loss function was weighted MSE: L = 2.0\u00d7MSE(completeness) + 1.0\u00d7MSE(contamination). Data augmentation included 2% random k-mer masking and Gaussian noise injection (\u03c3=0.01). Training ran for 87 epochs (early stopping patience=20), with best validation performance at epoch 67. The model was exported to ONNX format (opset 17, FP32) for inference."
]));

children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Benchmark evaluation")] }));
children.push(p([
  "Five benchmark sets (1,000 genomes each) were generated from NCBI-finished reference genomes (Complete Genome or Chromosome assembly level) in the test split (1,810 genomes from 35 phyla): Set A (completeness gradient, 0% contamination), Set B (contamination gradient with cross-phylum contaminants, 100% completeness), Set C (1,000 Patescibacteria, uniform completeness and contamination), Set D (1,000 Archaea, uniform completeness and contamination), and Set E (mixed realistic: 200 pure + 200 complete + 600 other with 70% cross-phylum / 30% within-phylum contamination)."
]));
children.push(p([
  "Competitor tools were run with default settings: CheckM2 v1.0.1 (32 threads, checkm2_py39 conda environment), CoCoPyE v0.5.0 (48 threads), and DeepCheck (PyTorch inference on CheckM2 intermediate feature vectors). Speed was normalized to genomes per minute per thread. Peak memory was measured using /usr/bin/time -v."
]));
children.push(p([
  "Statistical significance was assessed using the paired Wilcoxon signed-rank test (one-sided, H\u2081: MAGICC errors < competitor errors) with significance threshold p<0.05. Bootstrap 95% confidence intervals (1,000 resamples) were computed for all MAE estimates. MIMAG classification F1 scores were computed for Sets C and D."
]));

children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("MAGICC inference pipeline")] }));
children.push(p([
  "MAGICC is implemented as a Python package with a command-line interface (python -m magicc predict). The pipeline accepts a directory of FASTA files and produces a TSV output with predicted completeness and contamination. Multi-threaded feature extraction uses Python multiprocessing, with each worker maintaining its own Numba-compiled k-mer counter. ONNX Runtime is used for model inference. The entire pipeline\u2014from FASTA reading to prediction output\u2014completes in ~75 ms per genome on a single CPU thread."
]));

// ===== DATA AVAILABILITY =====
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Data availability")] }));
children.push(p([
  "All benchmark datasets, pre-trained models, normalization parameters, and generation scripts will be made available upon publication. Reference genomes were obtained from the NCBI Assembly database via the NCBI datasets CLI. GTDB metadata was obtained from the Genome Taxonomy Database (https://gtdb.ecogenomic.org/)."
]));

// ===== CODE AVAILABILITY =====
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Code availability")] }));
children.push(p([
  "MAGICC is available at [GitHub URL] under the MIT License. The package includes the pre-trained ONNX model, selected k-mer lists, normalization parameters, and a command-line interface for genome quality prediction."
]));

// ===== REFERENCES =====
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("References")] }));

const refs = [
  "Parks, D. H. et al. Recovery of nearly 8,000 metagenome-assembled genomes substantially expands the tree of life. Nat. Microbiol. 2, 1533\u20131542 (2017).",
  "Nayfach, S. et al. A genomic catalog of Earth's microbiomes. Nat. Biotechnol. 39, 499\u2013509 (2021).",
  "Bowers, R. M. et al. Minimum information about a single amplified genome (MISAG) and a metagenome-assembled genome (MIMAG) of bacteria and archaea. Nat. Biotechnol. 35, 725\u2013731 (2017).",
  "Chklovski, A., Parks, D. H., Woodcroft, B. J. & Tyson, G. W. CheckM2: a rapid, scalable and accurate tool for assessing microbial genome quality using machine learning. Nat. Methods 20, 1203\u20131212 (2023).",
  "Parks, D. H., Imelfort, M., Skennerton, C. T., Hugenholtz, P. & Tyson, G. W. CheckM: assessing the quality of microbial genomes recovered from isolates, single cells, and metagenomes. Genome Res. 25, 1043\u20131055 (2015).",
  "Hyatt, D. et al. Prodigal: prokaryotic gene recognition and translation initiation site identification. BMC Bioinformatics 11, 119 (2010).",
  "Buchfink, B., Xie, C. & Huson, D. H. Fast and sensitive protein alignment using DIAMOND. Nat. Methods 12, 59\u201360 (2015).",
  "Cheng, R. et al. gcMeta: a Global Catalogue of Metagenomics platform to support the archiving, standardization and analysis of microbiome data. Nucleic Acids Res. 54, D724\u2013D735 (2025).",
  "\u00d6zsoy, E. D. & Clean, T. CoCoPyE: fast and accurate estimation of prokaryotic genome completeness and contamination using feature engineering. GigaScience 13, giae079 (2024).",
  "Liao, H. & Zhang, Z. DeepCheck: multitask learning aids in assessing microbial genome quality. Bioinformatics 40, btae630 (2024).",
  "Orakov, A. et al. GUNC: detection of chimerism and contamination in prokaryotic genomes. Genome Biol. 22, 178 (2021).",
  "Kang, D. D. et al. MetaBAT 2: an adaptive binning algorithm for robust and efficient genome reconstruction from metagenome assemblies. PeerJ 7, e7359 (2019).",
  "Alneberg, J. et al. Binning metagenomic contigs by coverage and composition. Nat. Methods 11, 1144\u20131146 (2014).",
  "Parks, D. H. et al. GTDB: an ongoing census of bacterial and archaeal diversity through a phylogenetically consistent, rank normalized and complete genome-based taxonomy. Nucleic Acids Res. 50, D199\u2013D207 (2022).",
  "Wood, D. E. & Salzberg, S. L. Kraken: ultrafast metagenomic sequence classification using exact alignments. Genome Biol. 15, R46 (2014)."
];

for (let idx = 0; idx < refs.length; idx++) {
  children.push(p([
    { text: `${idx + 1}. `, bold: true },
    refs[idx]
  ]));
}

// ===== TABLE 1 =====
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 240, after: 120 },
  children: [new TextRun({ text: "Table 1. Benchmark results across five evaluation sets (5,000 genomes total).", bold: true })]
}));

const tBorder = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
const tBorders = { top: tBorder, bottom: tBorder, left: tBorder, right: tBorder };
const colW = [1800, 1200, 1200, 1000, 1000, 1400, 1200];

function tCell(text, opts = {}) {
  return new TableCell({
    borders: tBorders,
    width: { size: opts.width || 1200, type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    shading: opts.header ? { fill: "D5E8F0", type: ShadingType.CLEAR } : undefined,
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 40, after: 40 },
      children: [new TextRun({ text, bold: opts.bold || opts.header || false, size: 18, font: "Arial" })]
    })]
  });
}

const tableData = [
  ["Tool", "Comp. MAE (%)", "Cont. MAE (%)", "Comp. R\u00b2", "Cont. R\u00b2", "Speed (G/min/thr)", "Peak mem (GB)"],
  ["MAGICC", "2.74", "3.60", "0.90", "0.96", "1,451", "0.66"],
  ["CheckM2", "6.00", "23.14", "0.67", "0.34", "0.82", "18.76"],
  ["CoCoPyE", "6.62", "20.26", "0.65", "0.65", "0.70", "15.93"],
  ["DeepCheck", "10.10", "27.57", "0.48", "0.18", "0.82*", "1.46\u2020"]
];

const tableRows = tableData.map((row, ri) =>
  new TableRow({
    tableHeader: ri === 0,
    children: row.map((cell, ci) =>
      tCell(cell, { header: ri === 0, bold: ri === 1, width: colW[ci] })
    )
  })
);

children.push(new Table({
  columnWidths: colW,
  rows: tableRows
}));

children.push(p([
  { text: "*", superScript: true },
  "DeepCheck requires CheckM2 feature extraction; effective speed equals CheckM2. ",
  { text: "\u2020", superScript: true },
  "Inference-only memory; requires CheckM2 pipeline for feature extraction."
], { spacing: { before: 60, after: 200 } }));

// ===== BUILD DOCUMENT =====
const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: "Arial", size: 22 } // 11pt
      }
    },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal",
        run: { size: 32, bold: true, color: "000000", font: "Arial" },
        paragraph: { spacing: { before: 480, after: 240 }, alignment: AlignmentType.CENTER } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, color: "000000", font: "Arial" },
        paragraph: { spacing: { before: 360, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: "333333", font: "Arial" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } }
    ]
  },
  sections: [{
    properties: {
      page: {
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        size: { width: 12240, height: 15840 } // Letter
      }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "MAGICC manuscript", italics: true, size: 18, color: "888888" })]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "Page ", size: 18 }),
            new TextRun({ children: [PageNumber.CURRENT], size: 18 }),
            new TextRun({ text: " of ", size: 18 }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 18 })
          ]
        })]
      })
    },
    children
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(outPath, buffer);
  console.log(`Written to ${outPath} (${(buffer.length / 1024 / 1024).toFixed(1)} MB)`);
});
