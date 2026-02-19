const fs = require("fs");
const path = require("path");
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  Table,
  TableRow,
  TableCell,
  WidthType,
  AlignmentType,
  BorderStyle,
  ShadingType,
  ImageRun,
  PageBreak,
  PageNumber,
  PageNumberSeparator,
  Header,
  Footer,
  HeadingLevel,
  TableLayoutType,
} = require("docx");

// ─── helpers ────────────────────────────────────────────────────────────────

const DIR = __dirname;
const FIGURES_DIR = path.join(DIR, "figures");

const FONT_DEFAULT = "Times New Roman";
const FONT_HEADING = "Arial";

const BLUE_SHADING = { type: ShadingType.CLEAR, fill: "D5E8F0", color: "auto" };
const BORDER_THIN = {
  style: BorderStyle.SINGLE,
  size: 1,
  color: "CCCCCC",
};
const CELL_BORDERS = {
  top: BORDER_THIN,
  bottom: BORDER_THIN,
  left: BORDER_THIN,
  right: BORDER_THIN,
};

/** Shorthand for a text run */
function tr(text, opts = {}) {
  return new TextRun({
    text,
    font: opts.font || FONT_DEFAULT,
    size: opts.size || 22, // 11pt
    bold: opts.bold || false,
    italics: opts.italics || false,
    superScript: opts.superScript || false,
  });
}

/** Paragraph with default font */
function para(children, opts = {}) {
  if (typeof children === "string") {
    children = [tr(children, opts)];
  }
  return new Paragraph({
    children,
    alignment: opts.alignment || AlignmentType.LEFT,
    spacing: opts.spacing || { after: 120 },
    indent: opts.indent,
  });
}

/** Section heading */
function sectionHeading(text) {
  return new Paragraph({
    children: [
      new TextRun({
        text,
        font: FONT_HEADING,
        size: 26, // 13pt
        bold: true,
      }),
    ],
    spacing: { before: 360, after: 120 },
  });
}

/** Sub-table heading e.g. "Table S1a: ..." */
function subHeading(text) {
  return new Paragraph({
    children: [
      new TextRun({
        text,
        font: FONT_HEADING,
        size: 22, // 11pt
        bold: true,
      }),
    ],
    spacing: { before: 240, after: 80 },
  });
}

/** Footnote paragraph */
function footnote(text) {
  return new Paragraph({
    children: [
      new TextRun({
        text,
        font: FONT_DEFAULT,
        size: 18, // 9pt
        italics: true,
      }),
    ],
    spacing: { after: 80 },
  });
}

/** Parse a markdown-style cell that may contain bold markers */
function parseCellText(text) {
  text = text.trim();
  const runs = [];
  // Split on **...**
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  for (const part of parts) {
    if (part.startsWith("**") && part.endsWith("**")) {
      runs.push(
        new TextRun({
          text: part.slice(2, -2),
          font: FONT_DEFAULT,
          size: 18,
          bold: true,
        })
      );
    } else if (part.length > 0) {
      runs.push(
        new TextRun({
          text: part,
          font: FONT_DEFAULT,
          size: 18,
        })
      );
    }
  }
  return runs;
}

/** Build a table cell */
function cell(text, opts = {}) {
  const isHeader = opts.header || false;
  const widthDxa = opts.width;
  const alignment = opts.alignment || AlignmentType.CENTER;

  let children;
  if (isHeader) {
    children = [
      new TextRun({
        text: text.trim(),
        font: FONT_DEFAULT,
        size: 18,
        bold: true,
      }),
    ];
  } else {
    children = parseCellText(text);
  }

  const cellOpts = {
    children: [
      new Paragraph({
        children,
        alignment,
        spacing: { before: 30, after: 30 },
      }),
    ],
    borders: CELL_BORDERS,
    width: widthDxa ? { size: widthDxa, type: WidthType.DXA } : undefined,
  };

  if (isHeader) {
    cellOpts.shading = BLUE_SHADING;
  }

  return new TableCell(cellOpts);
}

/** Build a Word table from header array and data arrays */
function buildTable(headers, rows, columnWidthsDxa) {
  const totalWidth = columnWidthsDxa.reduce((a, b) => a + b, 0);

  const headerRow = new TableRow({
    children: headers.map((h, i) =>
      cell(h, { header: true, width: columnWidthsDxa[i] })
    ),
    tableHeader: true,
  });

  const dataRows = rows.map(
    (row) =>
      new TableRow({
        children: row.map((c, i) =>
          cell(c, { width: columnWidthsDxa[i] })
        ),
      })
  );

  return new Table({
    rows: [headerRow, ...dataRows],
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: columnWidthsDxa,
    layout: TableLayoutType.FIXED,
  });
}

/** A horizontal line separator */
function separator() {
  return new Paragraph({
    children: [],
    spacing: { before: 120, after: 120 },
    border: {
      bottom: { style: BorderStyle.SINGLE, size: 6, color: "999999" },
    },
  });
}

/** Spacer paragraph */
function spacer(pts = 200) {
  return new Paragraph({ children: [], spacing: { after: pts } });
}

/** Page break inside a paragraph */
function pageBreakPara() {
  return new Paragraph({ children: [new PageBreak()] });
}

// ─── Figure helper ──────────────────────────────────────────────────────────

function figureImage(filename, widthPx, heightPx) {
  const imgPath = path.join(FIGURES_DIR, filename);
  const imgData = fs.readFileSync(imgPath);
  return new Paragraph({
    children: [
      new ImageRun({
        data: imgData,
        transformation: { width: widthPx, height: heightPx },
        type: "png",
        altText: {
          title: filename,
          description: `Supplementary figure ${filename}`,
          name: filename,
        },
      }),
    ],
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 80 },
  });
}

/** Figure caption with bold label and panel labels */
function figureCaption(label, captionText) {
  // Bold the "Figure SN." part, then render rest with bold (a) (b) labels
  const runs = [
    new TextRun({
      text: label,
      font: FONT_DEFAULT,
      size: 22,
      bold: true,
    }),
  ];

  // Process caption text for bold (a) and (b)
  const parts = captionText.split(/(\([a-z]\))/g);
  for (const part of parts) {
    if (/^\([a-z]\)$/.test(part)) {
      runs.push(
        new TextRun({
          text: " " + part,
          font: FONT_DEFAULT,
          size: 22,
          bold: true,
        })
      );
    } else if (part.length > 0) {
      runs.push(
        new TextRun({
          text: part,
          font: FONT_DEFAULT,
          size: 22,
        })
      );
    }
  }

  return new Paragraph({
    children: runs,
    spacing: { after: 240 },
    alignment: AlignmentType.LEFT,
  });
}

// ─── Document content ───────────────────────────────────────────────────────

function buildDocument() {
  const children = [];

  // ──── Title ────
  children.push(
    new Paragraph({
      children: [
        new TextRun({
          text: "Supplementary Information",
          font: FONT_DEFAULT,
          size: 32, // 16pt
          bold: true,
        }),
      ],
      alignment: AlignmentType.CENTER,
      spacing: { after: 120 },
    })
  );

  // ──── Subtitle ────
  children.push(
    new Paragraph({
      children: [
        new TextRun({
          text: "MAGICC: ultra-fast genome quality assessment using core gene k-mer profiles and deep learning",
          font: FONT_DEFAULT,
          size: 24, // 12pt
          italics: true,
        }),
      ],
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
    })
  );

  children.push(separator());

  // ──── TABLE S1 ────
  children.push(sectionHeading("Table S1: Motivating analysis results"));
  children.push(
    para(
      "Results from the motivating analysis demonstrating limitations of existing tools. These experiments used exclusively NCBI-finished reference genomes as dominant genomes. MAGICC was not included in the motivating analysis because these experiments were designed to characterize the gap in existing tools that motivated MAGICC\u2019s development."
    )
  );

  // S1a
  children.push(
    subHeading(
      "Table S1a: Set A \u2014 Controlled completeness (1,000 genomes, 0% contamination)"
    )
  );
  children.push(
    buildTable(
      ["Tool", "Comp MAE", "Cont MAE"],
      [
        ["CheckM2", "2.54%", "0.28%"],
        ["CoCoPyE", "3.47%", "0.73%"],
        ["DeepCheck", "4.15%", "0.43%"],
      ],
      [3200, 2800, 2800]
    )
  );

  // S1b
  children.push(
    subHeading(
      "Table S1b: Set B \u2014 Controlled contamination (1,000 genomes, 100% completeness, cross-phylum)"
    )
  );
  children.push(
    buildTable(
      ["Tool", "Comp MAE", "Cont MAE", "Cont MAE (>20% true)"],
      [
        ["CheckM2", "0.42%", "17.08%", "25.8%"],
        ["CoCoPyE", "2.57%", "19.14%", "30.6%"],
        ["DeepCheck", "6.74%", "25.70%", "38.6%"],
      ],
      [2200, 2000, 2000, 2600]
    )
  );

  // S1c
  children.push(
    subHeading(
      "Table S1c: Set C \u2014 Realistic mixed (1,000 genomes: 200 pure + 200 complete + 600 other)"
    )
  );
  children.push(
    buildTable(
      [
        "Tool",
        "Comp MAE",
        "Cont MAE",
        "Comp RMSE",
        "Cont RMSE",
        "Comp R\u00B2",
        "Cont R\u00B2",
      ],
      [
        [
          "CheckM2",
          "9.13%",
          "18.13%",
          "14.26%",
          "26.69%",
          "0.4962",
          "0.5330",
        ],
        [
          "CoCoPyE",
          "4.99%",
          "20.32%",
          "7.27%",
          "29.37%",
          "0.8170",
          "0.7379",
        ],
        [
          "DeepCheck",
          "11.96%",
          "24.37%",
          "17.19%",
          "35.46%",
          "0.4295",
          "0.3307",
        ],
      ],
      [1400, 1200, 1200, 1300, 1300, 1200, 1200]
    )
  );

  children.push(spacer(80));
  children.push(
    para(
      "At true contamination levels above 20%, all existing tools produce MAEs of 25.8\u201338.6%, meaning a genome with 60% true contamination might be reported as having only 20\u201330% contamination. This systematic underestimation motivated the development of MAGICC\u2019s k-mer-based approach."
    )
  );

  children.push(separator());

  // ──── TABLE S2 ────
  children.push(
    sectionHeading("Table S2: Detailed benchmark results per set per tool")
  );
  children.push(
    para(
      "All benchmark sets used NCBI-finished reference genomes (assembly level \u201CComplete Genome\u201D or \u201CChromosome\u201D) as dominant genomes, providing clean ground truth with independently verified assemblies. Sets A_v2 and B_v2 are controlled gradient experiments; Sets C and D test challenging lineages; Set E tests realistic mixed conditions."
    )
  );

  // S2a
  children.push(subHeading("Table S2a: Completeness prediction (MAE and R\u00B2)"));
  children.push(
    buildTable(
      [
        "Set",
        "MAGICC MAE",
        "MAGICC R\u00B2",
        "CheckM2 MAE",
        "CoCoPyE MAE",
        "DeepCheck MAE",
      ],
      [
        [
          "A_v2 (n=1,000)",
          "**1.99%**",
          "0.9585",
          "2.54%",
          "3.63%",
          "4.26%",
        ],
        [
          "B_v2 (n=1,000)",
          "**0.77%**",
          "\u2014",
          "**0.45%**",
          "2.61%",
          "6.61%",
        ],
        [
          "C (n=1,000)",
          "**2.96%**",
          "0.9064",
          "7.99%",
          "15.73%",
          "17.97%",
        ],
        [
          "D (n=1,000)",
          "**4.06%**",
          "0.8098",
          "9.89%",
          "6.14%",
          "9.97%",
        ],
        [
          "E (n=1,000)",
          "**3.92%**",
          "0.8241",
          "9.14%",
          "5.02%",
          "11.69%",
        ],
      ],
      [1700, 1400, 1400, 1500, 1500, 1500]
    )
  );

  // S2b
  children.push(
    subHeading("Table S2b: Contamination prediction (MAE and R\u00B2)")
  );
  children.push(
    buildTable(
      [
        "Set",
        "MAGICC MAE",
        "MAGICC R\u00B2",
        "CheckM2 MAE",
        "CoCoPyE MAE",
        "DeepCheck MAE",
      ],
      [
        [
          "A_v2 (n=1,000)",
          "0.46%",
          "\u2014",
          "**0.27%**",
          "0.77%",
          "0.40%",
        ],
        [
          "B_v2 (n=1,000)",
          "**2.78%**",
          "0.9763",
          "17.66%",
          "19.14%",
          "25.70%",
        ],
        [
          "C (n=1,000)",
          "**5.05%**",
          "0.9390",
          "42.37%",
          "34.07%",
          "44.56%",
        ],
        [
          "D (n=1,000)",
          "**5.40%**",
          "0.9241",
          "36.92%",
          "25.73%",
          "41.77%",
        ],
        [
          "E (n=1,000)",
          "**4.32%**",
          "0.9490",
          "18.47%",
          "21.60%",
          "25.43%",
        ],
      ],
      [1700, 1400, 1400, 1500, 1500, 1500]
    )
  );

  // S2c
  children.push(
    subHeading("Table S2c: Overall results (5,000 genomes across all 5 sets)")
  );
  children.push(
    buildTable(
      ["Metric", "MAGICC", "CheckM2", "CoCoPyE", "DeepCheck"],
      [
        ["Comp MAE", "**2.74%**", "6.00%", "6.62%", "10.10%"],
        ["Cont MAE", "**3.60%**", "23.14%", "20.26%", "27.57%"],
        ["Comp R\u00B2", "**0.90**", "0.67", "0.65", "0.48"],
        ["Cont R\u00B2", "**0.96**", "0.34", "0.65", "0.18"],
      ],
      [1800, 1800, 1800, 1800, 1800]
    )
  );

  children.push(spacer(80));
  children.push(
    para([
      tr(
        "Benchmark set descriptions: Set A_v2 \u2014 completeness gradient (50\u2013100%), 0% contamination, 1,000 genomes from finished references. Set B_v2 \u2014 contamination gradient (0\u201380%, cross-phylum), 100% completeness, 1,000 genomes. Set C \u2014 1,000 Patescibacteria genomes, uniform completeness (50\u2013100%) and contamination (0\u2013100%). Set D \u2014 1,000 Archaea genomes, uniform completeness (50\u2013100%) and contamination (0\u2013100%). Set E \u2014 1,000 mixed genomes (200 pure + 200 complete + 600 other with 70% cross-phylum / 30% within-phylum contamination). Bold indicates best performance per metric. All improvements of MAGICC over competitor tools on contamination were statistically significant (paired Wilcoxon signed-rank test, p < 0.001).",
        { size: 20 }
      ),
    ])
  );

  children.push(separator());

  // ──── TABLE S3 ────
  children.push(sectionHeading("Table S3: MIMAG classification F1 scores"));
  children.push(
    para(
      "MIMAG quality classification performance on benchmark Sets C (Patescibacteria) and D (Archaea), which span the full range of completeness and contamination. F1 scores are macro-averaged across three MIMAG quality categories: high-quality (\u226590% complete, <5% contaminated), medium-quality (\u226550% complete, <10% contaminated), and low-quality (all others)."
    )
  );
  children.push(
    buildTable(
      ["Tool", "MIMAG F1 (macro-averaged)"],
      [
        ["**MAGICC**", "**0.89**"],
        ["CoCoPyE", "0.42\u20130.71"],
        ["CheckM2", "0.28\u20130.37"],
        ["DeepCheck", "0.21\u20130.27"],
      ],
      [4400, 4400]
    )
  );
  children.push(spacer(80));
  children.push(
    para(
      "MAGICC\u2019s substantially higher F1 score demonstrates that its improved contamination estimation translates directly to more reliable quality tier assignments. The low F1 scores of competitor tools are driven primarily by their systematic underestimation of contamination, which causes heavily contaminated genomes to be misclassified as high- or medium-quality."
    )
  );

  children.push(separator());

  // ──── TABLE S4 ────
  children.push(
    sectionHeading("Table S4: Computational resource comparison")
  );
  children.push(
    para(
      "All tools were benchmarked on Set E (1,000 mixed genomes) using /usr/bin/time -v for peak memory measurement. Speed is reported as wall-clock time under each tool\u2019s typical multi-threaded configuration."
    )
  );
  children.push(
    buildTable(
      [
        "Tool",
        "Threads",
        "Peak memory (GB)",
        "Wall-clock time (Set E)",
        "Genomes/min/thread",
      ],
      [
        ["**MAGICC**", "**1**", "**0.66**", "**97.5 s**", "**1,451**"],
        ["CheckM2", "32", "18.76", "86 min 37 s", "0.82"],
        ["CoCoPyE", "48", "15.93", "60 min 32 s", "0.70"],
        ["DeepCheck", "1*", "1.46*", "12 min 5 s*", "0.82**"],
      ],
      [1600, 1400, 1800, 2200, 1800]
    )
  );

  children.push(
    footnote(
      "* DeepCheck memory and wall-clock time reflect inference only; DeepCheck requires CheckM2\u2019s full pipeline (gene prediction, DIAMOND alignment, KEGG annotation) for feature extraction, which must be run first."
    )
  );
  children.push(
    footnote(
      "** DeepCheck\u2019s effective end-to-end speed equals CheckM2\u2019s speed since feature extraction dominates runtime."
    )
  );

  children.push(spacer(80));

  // Derived metrics
  children.push(
    buildTable(
      ["Derived metric", "MAGICC", "CheckM2", "CoCoPyE"],
      [
        ["Memory ratio vs MAGICC", "1x", "28x", "24x"],
        ["Speed ratio vs MAGICC (per thread)", "1x", "1/1,700x", "1/2,100x"],
        [
          "Projected time for 100K genomes (1 thread)",
          "~69 min",
          "~87 hours",
          "~60+ hours",
        ],
      ],
      [2800, 2000, 2000, 2000]
    )
  );

  children.push(spacer(80));
  children.push(
    para(
      "MAGICC\u2019s speed advantage stems from eliminating gene prediction, protein alignment, and database search steps. K-mer counting uses a Numba-accelerated rolling hash (17.5 ms per 5 Mb genome), and model inference uses ONNX Runtime (0.18 ms per sample at batch size 1,024)."
    )
  );

  children.push(separator());

  // ──── TABLE S5 ────
  children.push(
    sectionHeading("Table S5: Reference genome filtering statistics")
  );
  children.push(
    para(
      "Genome metadata was downloaded from the Genome Taxonomy Database (GTDB) and filtered using strict quality criteria to obtain high-quality reference genomes for training data synthesis. All filters were applied conjunctively (all must pass):"
    )
  );
  // Filter criteria bullets
  const filters = [
    "CheckM2 completeness >= 98%",
    "CheckM2 contamination <= 2%",
    "Contig count < 100",
    "N50 > 20 kbp",
    "Longest contig > 100 kbp",
  ];
  for (const f of filters) {
    children.push(
      new Paragraph({
        children: [tr("\u2022  " + f)],
        spacing: { after: 40 },
        indent: { left: 360 },
      })
    );
  }
  children.push(spacer(80));

  children.push(
    buildTable(
      ["Stage", "Bacterial", "Archaeal", "Total"],
      [
        ["Starting genomes (GTDB)", "715,230", "17,245", "732,475"],
        [
          "**After all filters**",
          "**275,207 (38.5%)**",
          "**1,976 (11.5%)**",
          "**277,183**",
        ],
        ["Phyla represented", "", "", "110"],
        ["Genome size (median)", "", "", "3.8 Mbp"],
        ["Genome size (range)", "", "", "0.3\u201313.6 Mbp"],
        ["N50 (median)", "", "", "270 kbp"],
        ["Contig count (median)", "", "", "38"],
        ["**Selected for project**", "98,024", "1,976", "100,000"],
        [
          "Selection method",
          "Square-root proportional stratified sampling",
          "All included",
          "",
        ],
        ["Successfully downloaded", "", "", "99,957 (99.96%)"],
        ["**Train / Val / Test split**", "", "", ""],
        ["Training set", "", "", "79,948"],
        ["Validation set", "", "", "10,010"],
        ["Test set", "", "", "9,999"],
      ],
      [2800, 2400, 1600, 2000]
    )
  );

  children.push(spacer(80));
  children.push(
    para(
      "Selection ensured all available genomes from underrepresented lineages were included: Patescibacteriota (1,609), DPANN archaea (24), and candidate phyla (586). Genomes were split 80/10/10 by phylum with no overlap between splits."
    )
  );

  children.push(separator());

  // ──── TABLE S6 ────
  children.push(
    sectionHeading("Table S6: K-mer feature selection statistics")
  );
  children.push(
    para(
      "K-mer features were derived from canonical 9-mers counted in single-copy core gene DNA sequences. Representative genomes were selected from the training set only (no data leakage). Core genes were identified using Prodigal v2.6.3 for gene prediction and HMMER 3.4 with trusted cutoff thresholds for HMM profile searching."
    )
  );

  children.push(
    buildTable(
      ["Parameter", "Bacterial", "Archaeal"],
      [
        ["Representative genomes", "1,000", "1,000"],
        ["Phyla represented", "97", "13"],
        ["Core gene HMM profiles", "85 (TIGR/JCVI)", "128 (Pfam)"],
        ["Core genes per genome (mean)", "83.3", "127.9"],
        ["Core genes per genome (median)", "84", "129"],
        ["Core genes per genome (range)", "50\u201389", "80\u2013138"],
        ["Total unique canonical 9-mers observed", "131,072", "131,072"],
        ["K-mers per genome (mean)", "40,517", "51,320"],
        ["K-mers per genome (median)", "41,638", "52,755"],
        ["K-mers per genome (range)", "23,069\u201349,624", "32,392\u201367,234"],
        ["**Selected k-mers**", "**9,000**", "**1,000**"],
        ["Selection criterion", "Top prevalence", "Top prevalence"],
        ["Prevalence range (of 1,000 genomes)", "529\u2013992", "791\u2013998"],
        ["Overlap between bacterial and archaeal sets", "751", "751"],
        ["**Final merged k-mer set**", "**9,249 unique canonical 9-mers**", ""],
        ["Bacteria-only k-mers", "8,249", "\u2014"],
        ["Archaea-only k-mers", "\u2014", "249"],
        ["Shared k-mers", "751", "751"],
      ],
      [3200, 2800, 2800]
    )
  );

  children.push(spacer(80));
  children.push(
    footnote(
      "Note: 131,072 = 4\u2079 / 2 represents all possible canonical 9-mers (since 9 is odd, each k-mer has a distinct reverse complement)."
    )
  );

  children.push(separator());

  // ──── TABLE S7 ────
  children.push(
    sectionHeading("Table S7: Synthetic training data composition")
  );
  children.push(
    para(
      "One million synthetic genomes were generated from 99,957 high-quality reference genomes. Each batch of 10,000 genomes followed the composition below, maintaining consistent ratios across training, validation, and test splits."
    )
  );

  children.push(
    buildTable(
      ["Sample type", "Per batch (of 10,000)", "Percentage", "Description"],
      [
        [
          "Pure genomes",
          "1,500",
          "15%",
          "0% contamination, 50\u2013100% completeness",
        ],
        [
          "Complete genomes",
          "1,500",
          "15%",
          "100% completeness (original contigs), 0\u2013100% contamination",
        ],
        [
          "Within-phylum contamination",
          "3,000",
          "30%",
          "1\u20133 contaminant genomes from same phylum",
        ],
        [
          "Cross-phylum contamination",
          "3,000",
          "30%",
          "1\u20135 contaminant genomes from different phyla",
        ],
        [
          "Reduced-genome organisms",
          "500",
          "5%",
          "Small-genome taxa (e.g., Patescibacteriota)",
        ],
        ["Archaeal genomes", "500", "5%", "Archaeal dominant genomes"],
        ["**Total per batch**", "**10,000**", "**100%**", ""],
      ],
      [2400, 2000, 1400, 3000]
    )
  );

  children.push(spacer(120));

  children.push(
    buildTable(
      ["Dataset split", "Batches", "Total genomes"],
      [
        ["Training", "80", "800,000"],
        ["Validation", "10", "100,000"],
        ["Test", "10", "100,000"],
        ["**Total**", "**100**", "**1,000,000**"],
      ],
      [3000, 2800, 2800]
    )
  );

  children.push(spacer(80));
  children.push(
    para(
      "Genome fragmentation simulated four quality tiers: high (10\u201350 contigs, N50 100\u2013500 kb), medium (50\u2013200 contigs, N50 20\u2013100 kb), low (200\u2013500 contigs, N50 5\u201320 kb), and highly fragmented (500\u20132,000 contigs, N50 1\u20135 kb). Three bias mechanisms were applied: coverage-based dropout, GC-biased loss, and repeat region exclusion. Contamination levels were distributed among contaminant genomes using Dirichlet allocation."
    )
  );

  children.push(separator());

  // ──── TABLE S8 ────
  children.push(
    sectionHeading("Table S8: Neural network training configuration")
  );
  children.push(
    para(
      "The MAGICCModelV3 architecture was developed through three training iterations (V1, V2, V3), with progressive improvements to architecture, regularization, and output activation."
    )
  );

  // S8a
  children.push(subHeading("Table S8a: Architecture details"));
  children.push(
    buildTable(
      ["Component", "Configuration"],
      [
        ["**K-mer branch**", ""],
        ["Input", "9,249 k-mer counts"],
        [
          "Layer 1",
          "Dense(4,096) \u2192 BN \u2192 SiLU \u2192 Dropout(0.4) \u2192 SE block",
        ],
        [
          "Layer 2",
          "Dense(1,024) \u2192 BN \u2192 SiLU \u2192 Dropout(0.2) \u2192 SE block",
        ],
        ["Layer 3", "Dense(256) \u2192 BN \u2192 SiLU"],
        ["SE block", "Squeeze-and-excitation attention (reduction ratio 16)"],
        ["**Assembly branch**", ""],
        ["Input", "26 assembly statistics"],
        [
          "Layer 1",
          "Dense(128) \u2192 BN \u2192 SiLU \u2192 Dropout(0.2)",
        ],
        ["Layer 2", "Dense(64) \u2192 BN \u2192 SiLU"],
        ["**Fusion head**", ""],
        [
          "Cross-attention",
          "4 heads; assembly embedding as query, k-mer embedding reshaped to 16 tokens",
        ],
        ["Gated residual", "Learnable gate on cross-attention output"],
        [
          "Output layers",
          "Dense(128) \u2192 BN \u2192 SiLU \u2192 Dropout(0.1) \u2192 Dense(64) \u2192 SiLU \u2192 Dense(2)",
        ],
        ["**Output activation**", ""],
        ["Completeness", "sigmoid(x) \u00D7 50 + 50, bounded [50, 100]%"],
        ["Contamination", "sigmoid(x) \u00D7 100, bounded [0, 100]%"],
        ["**Total parameters**", "**44,851,010**"],
      ],
      [3400, 5400]
    )
  );

  children.push(spacer(120));

  // S8b
  children.push(subHeading("Table S8b: Training hyperparameters"));
  children.push(
    buildTable(
      ["Parameter", "Value"],
      [
        ["Optimizer", "AdamW"],
        ["Learning rate", "1 \u00D7 10\u207B\u00B3"],
        ["Weight decay", "5 \u00D7 10\u207B\u2074"],
        [
          "LR schedule",
          "Cosine annealing warm restarts (T\u2080=10, T_mult=2)",
        ],
        [
          "Loss function",
          "Weighted MSE: 2.0 \u00D7 MSE(completeness) + 1.0 \u00D7 MSE(contamination)",
        ],
        ["Batch size", "512"],
        ["Max epochs", "150 (early stopping patience=20)"],
        ["Gradient clipping", "max_norm = 1.0"],
        ["Mixed precision", "FP16 with gradient checkpointing"],
        [
          "Data augmentation",
          "2% random k-mer masking + Gaussian noise (\u03C3=0.01)",
        ],
        ["GPU", "Quadro P2200 (5.1 GB VRAM)"],
        ["ONNX export", "Opset 17, FP32, dynamic batch sizes"],
      ],
      [3400, 5400]
    )
  );

  children.push(spacer(120));

  // S8c
  children.push(
    subHeading("Table S8c: Training results across model versions")
  );
  children.push(
    buildTable(
      ["Metric", "V1", "V2", "V3 (final)"],
      [
        [
          "Architecture",
          "Dense + concat fusion",
          "Dense + concat fusion",
          "SE attention + cross-attention fusion",
        ],
        ["Parameters", "42,417,538", "42,417,538", "44,851,010"],
        ["Assembly features", "20", "26", "26"],
        [
          "Completeness output",
          "sigmoid \u00D7 100",
          "sigmoid \u00D7 50 + 50",
          "sigmoid \u00D7 50 + 50",
        ],
        ["Dropout (layer 1)", "0.3", "0.4", "0.4"],
        [
          "Weight decay",
          "1 \u00D7 10\u207B\u2074",
          "5 \u00D7 10\u207B\u2074",
          "5 \u00D7 10\u207B\u2074",
        ],
        ["Loss weights (comp:cont)", "1.0 : 1.5", "2.0 : 1.0", "2.0 : 1.0"],
        ["Total epochs (early stop)", "87", "87", "84"],
        ["Best epoch", "67", "67", "64"],
        ["Training time", "4.7 hours", "4.5 hours", "~4 hours"],
        ["Val completeness MAE", "4.77%", "3.93%", "**3.77%**"],
        ["Val contamination MAE", "4.51%", "4.49%", "**4.35%**"],
        ["Val completeness R\u00B2", "0.755", "0.828", "**0.833**"],
        ["Val contamination R\u00B2", "0.937", "0.945", "**0.946**"],
      ],
      [2600, 2000, 2000, 2200]
    )
  );

  children.push(spacer(80));
  children.push(
    para(
      "Key changes from V1 to V2: fixed \u201Ccomplete\u201D sample type (was generating 86% mean completeness instead of 100%), changed output activation for completeness from sigmoid \u00D7 100 to sigmoid \u00D7 50 + 50 (using full sigmoid range), added 6 k-mer summary features (expanding assembly features from 20 to 26), increased regularization, and rebalanced loss weights to prioritize completeness. Key change from V2 to V3: added squeeze-and-excitation attention blocks and cross-attention fusion mechanism.",
      { size: 20 }
    )
  );

  children.push(separator());
  children.push(pageBreakPara());

  // ──── SUPPLEMENTARY FIGURES ────
  children.push(sectionHeading("Supplementary Figures"));

  // Figure S1
  children.push(figureImage("figS1_setA.png", 580, 200));
  children.push(
    figureCaption(
      "Figure S1. ",
      "Per-tool predicted versus true quality scores for benchmark Set A (completeness gradient). (a) Predicted versus true completeness for MAGICC, CheckM2, CoCoPyE, and DeepCheck. (b) Predicted versus true contamination. Set A contains 1,000 genomes with varying completeness (50\u2013100%) and 0% contamination, derived from NCBI-finished reference genomes."
    )
  );

  children.push(spacer(200));

  // Figure S2
  children.push(figureImage("figS2_setB.png", 580, 200));
  children.push(
    figureCaption(
      "Figure S2. ",
      "Per-tool predicted versus true quality scores for benchmark Set B (contamination gradient). (a) Predicted versus true completeness for MAGICC, CheckM2, CoCoPyE, and DeepCheck. (b) Predicted versus true contamination. Set B contains 1,000 genomes at 100% completeness with varying cross-phylum contamination (0\u201380%)."
    )
  );

  children.push(pageBreakPara());

  // Figure S3
  children.push(figureImage("figS3_setC.png", 580, 290));
  children.push(
    figureCaption(
      "Figure S3. ",
      "Per-tool predicted versus true quality scores for benchmark Set C (Patescibacteria). (a) Predicted versus true completeness for MAGICC, CheckM2, CoCoPyE, and DeepCheck. (b) Predicted versus true contamination. Set C contains 1,000 Patescibacteria genomes with uniform completeness (50\u2013100%) and contamination (0\u2013100%), testing performance on reduced-genome organisms."
    )
  );

  children.push(spacer(200));

  // Figure S4
  children.push(figureImage("figS4_setD.png", 580, 290));
  children.push(
    figureCaption(
      "Figure S4. ",
      "Per-tool predicted versus true quality scores for benchmark Set D (Archaea). (a) Predicted versus true completeness for MAGICC, CheckM2, CoCoPyE, and DeepCheck. (b) Predicted versus true contamination. Set D contains 1,000 archaeal genomes with uniform completeness (50\u2013100%) and contamination (0\u2013100%), testing performance on underrepresented taxonomic lineages."
    )
  );

  children.push(pageBreakPara());

  // Figure S5
  children.push(figureImage("figS5_setE.png", 580, 290));
  children.push(
    figureCaption(
      "Figure S5. ",
      "Per-tool predicted versus true quality scores for benchmark Set E (realistic mixed). (a) Predicted versus true completeness for MAGICC, CheckM2, CoCoPyE, and DeepCheck. (b) Predicted versus true contamination. Set E contains 1,000 genomes with varied completeness (50\u2013100%), contamination (0\u2013100%), and fragmentation levels, representative of the full training distribution."
    )
  );

  // ──── Build document ────

  const doc = new Document({
    styles: {
      default: {
        document: {
          run: {
            font: FONT_DEFAULT,
            size: 22,
          },
        },
      },
    },
    sections: [
      {
        properties: {
          page: {
            size: {
              width: 12240, // 8.5" letter
              height: 15840, // 11" letter
            },
            margin: {
              top: 1440,
              right: 1440,
              bottom: 1440,
              left: 1440,
            },
          },
        },
        headers: {
          default: new Header({ children: [] }),
        },
        footers: {
          default: new Footer({
            children: [
              new Paragraph({
                children: [
                  new TextRun({
                    children: [PageNumber.CURRENT],
                    font: FONT_DEFAULT,
                    size: 20,
                  }),
                ],
                alignment: AlignmentType.CENTER,
              }),
            ],
          }),
        },
        children,
      },
    ],
  });

  return doc;
}

// ─── Main ───────────────────────────────────────────────────────────────────

async function main() {
  const doc = buildDocument();
  const buffer = await Packer.toBuffer(doc);
  const outPath = path.join(DIR, "supplementary.docx");
  fs.writeFileSync(outPath, buffer);
  console.log(`Wrote ${outPath} (${(buffer.length / 1024).toFixed(1)} KB)`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
