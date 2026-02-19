const fs = require("fs");
const path = require("path");
const docx = require("docx");

const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  ImageRun,
  Table,
  TableRow,
  TableCell,
  HeadingLevel,
  AlignmentType,
  WidthType,
  BorderStyle,
  ShadingType,
  PageNumber,
  Footer,
  Header,
  PageBreak,
  ExternalHyperlink,
  TabStopType,
  TabStopPosition,
  convertInchesToTwip,
  LevelFormat,
} = docx;

// ── Paths ──────────────────────────────────────────────────────────────────────
const BASE = "/home/tianrm/projects/magicc2/manuscript";
const FIG_DIR = path.join(BASE, "figures");
const OUTPUT = path.join(BASE, "manuscript.docx");

// ── Figure metadata ────────────────────────────────────────────────────────────
const figures = {
  1: {
    file: path.join(FIG_DIR, "figure1_motivating.png"),
    width: 2156,
    height: 1452,
  },
  2: {
    file: path.join(FIG_DIR, "figure2_workflow.png"),
    width: 1686,
    height: 723,
  },
  3: {
    file: path.join(FIG_DIR, "figure3_benchmark.png"),
    width: 2180,
    height: 1452,
  },
};

// Scale to 6.5 inches wide (EMU units: 1 inch = 914400 EMU)
const PAGE_WIDTH_EMU = 6.5 * 914400;
function scaledDimensions(fig) {
  const ratio = fig.height / fig.width;
  const widthEMU = PAGE_WIDTH_EMU;
  const heightEMU = Math.round(widthEMU * ratio);
  return { width: widthEMU, height: heightEMU };
}

// ── Figure legends (extracted from the markdown) ─────────────────────────────
const figureLegends = {
  1: {
    title:
      "Existing tools systematically underestimate contamination from foreign organisms.",
    text: '(a) Schematic of the three motivating dataset profiles: Set A (completeness gradient, 0% contamination), Set B (contamination gradient, 100% completeness), and Set C (realistic mixed genomes with varying completeness and contamination). (b) Completeness MAE for CheckM2, CoCoPyE, and DeepCheck across Sets A, B, and C. Foreign contamination in Set C significantly inflates completeness errors compared to Sets A and B. (c) Contamination MAE across Sets A, B, and C. All tools severely underestimate contamination in Sets B and C. (d) Scatter of predicted versus true contamination for Set B (contamination gradient), showing systematic underestimation by all three tools as contamination increases. (e) Scatter of predicted versus true contamination for Set C (realistic mixed), confirming the underestimation pattern in genomes with combined incompleteness and contamination. Only CheckM2, CoCoPyE, and DeepCheck are shown; MAGICC is excluded to present the motivating gap before introducing our solution.',
  },
  2: {
    title:
      "MAGICC workflow from data curation to training and inference.",
    text: "The workflow begins with reference genome curation from GTDB (277,183 high-quality genomes across 110 phyla), followed by k-mer feature selection from core genes of 2,000 representative genomes, yielding 9,249 canonical 9-mers. One million synthetic genomes are generated with realistic fragmentation, contamination, and bias patterns. Features are extracted via two parallel streams: Numba-accelerated k-mer counting (9,249 features) and assembly statistics (26 features). These feed into a dual-branch neural network: the k-mer branch with squeeze-and-excitation attention and the assembly branch, fused via cross-attention where assembly features query k-mer tokens. Output predictions: completeness [50\u2013100%] and contamination [0\u2013100%]. At inference, MAGICC takes a FASTA file and produces quality predictions in ~75 ms per genome.",
  },
  3: {
    title:
      "MAGICC benchmark performance across five evaluation sets.",
    text: "(a) Schematic of the five benchmark dataset profiles: Set A (completeness gradient), Set B (contamination gradient), Set C (Patescibacteria), Set D (Archaea), and Set E (realistic mixed). (b) Completeness MAE comparison across four tools (MAGICC, CheckM2, CoCoPyE, DeepCheck) for each benchmark set. (c) Contamination MAE comparison across four tools for each benchmark set, highlighting MAGICC\u2019s consistent advantage. (d) Scatter of predicted versus true contamination for Set B (contamination gradient) for all four tools. (e) Scatter of predicted versus true contamination for Set C (Patescibacteria). (f) Scatter of predicted versus true contamination for Set D (Archaea). (g) Scatter of predicted versus true contamination for Set E (realistic mixed genomes). Across all scatter plots, MAGICC predictions (shown in red) closely follow the diagonal, while competitor tools systematically underestimate contamination.",
  },
};

// ── Helpers for inline formatting ──────────────────────────────────────────────

/**
 * Parse a line of text into an array of TextRun objects, handling:
 *   **bold**, *italic*, ^superscript^
 */
function parseInlineFormatting(text, baseOptions = {}) {
  const runs = [];
  // Regex to match **bold**, *italic*, or ^superscript^
  const pattern = /(\*\*(.+?)\*\*|\*(.+?)\*|\^(.+?)\^)/g;
  let lastIndex = 0;
  let match;

  while ((match = pattern.exec(text)) !== null) {
    // Text before the match
    if (match.index > lastIndex) {
      const before = text.slice(lastIndex, match.index);
      if (before) {
        runs.push(new TextRun({ text: before, ...baseOptions }));
      }
    }

    if (match[2]) {
      // **bold**
      runs.push(
        new TextRun({ text: match[2], bold: true, ...baseOptions })
      );
    } else if (match[3]) {
      // *italic*
      runs.push(
        new TextRun({ text: match[3], italics: true, ...baseOptions })
      );
    } else if (match[4]) {
      // ^superscript^
      runs.push(
        new TextRun({
          text: match[4],
          superScript: true,
          ...baseOptions,
        })
      );
    }

    lastIndex = match.index + match[0].length;
  }

  // Remaining text
  if (lastIndex < text.length) {
    runs.push(
      new TextRun({ text: text.slice(lastIndex), ...baseOptions })
    );
  }

  if (runs.length === 0) {
    runs.push(new TextRun({ text: text, ...baseOptions }));
  }

  return runs;
}

/**
 * Create a body paragraph with inline formatting.
 */
function bodyParagraph(text, extraOptions = {}) {
  return new Paragraph({
    spacing: { after: 120, line: 276 },
    ...extraOptions,
    children: parseInlineFormatting(text, {
      font: "Arial",
      size: 22, // 11pt
    }),
  });
}

/**
 * Create heading paragraphs
 */
function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 200 },
    children: [
      new TextRun({
        text: text,
        bold: true,
        font: "Arial",
        size: 28, // 14pt
      }),
    ],
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 160 },
    children: [
      new TextRun({
        text: text,
        bold: true,
        font: "Arial",
        size: 24, // 12pt
      }),
    ],
  });
}

/**
 * Create a figure with caption paragraph(s)
 */
function createFigure(figNum) {
  const fig = figures[figNum];
  const legend = figureLegends[figNum];
  const dim = scaledDimensions(fig);
  const imgData = fs.readFileSync(fig.file);

  const elements = [];

  // The image paragraph
  elements.push(
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 240, after: 120 },
      children: [
        new ImageRun({
          data: imgData,
          transformation: {
            width: Math.round(dim.width / 914400 * 72), // convert EMU to points... actually docx uses EMU
            height: Math.round(dim.height / 914400 * 72),
          },
          type: "png",
          altText: {
            title: `Figure ${figNum}`,
            description: legend.title,
            name: `figure${figNum}`,
          },
        }),
      ],
    })
  );

  // Caption paragraph: "Figure N. Title text" in bold, then legend text in normal
  const captionRuns = [
    new TextRun({
      text: `Figure ${figNum}. `,
      bold: true,
      font: "Arial",
      size: 20, // 10pt
    }),
    new TextRun({
      text: legend.title + " ",
      bold: true,
      font: "Arial",
      size: 20,
    }),
  ];

  // Parse the legend text for inline formatting (it may contain bold/italic refs)
  const legendRuns = parseInlineFormatting(legend.text, {
    font: "Arial",
    size: 20, // 10pt
  });
  captionRuns.push(...legendRuns);

  elements.push(
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 240 },
      children: captionRuns,
    })
  );

  return elements;
}

// ── Rebuild createFigure with correct pixel-based sizing ──────────────────────
// docx ImageRun transformation expects pixels, not points or EMU
// 6.5 inches at 96 DPI = 624 pixels
function createFigureCorrect(figNum) {
  const fig = figures[figNum];
  const legend = figureLegends[figNum];
  const imgData = fs.readFileSync(fig.file);

  // Scale to 6.5 inches. The transformation width/height are in pixels at 96 DPI
  const targetWidthPx = 624; // 6.5 inches * 96 DPI
  const ratio = fig.height / fig.width;
  const targetHeightPx = Math.round(targetWidthPx * ratio);

  const elements = [];

  // The image paragraph
  elements.push(
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 240, after: 120 },
      children: [
        new ImageRun({
          data: imgData,
          transformation: {
            width: targetWidthPx,
            height: targetHeightPx,
          },
          type: "png",
          altText: {
            title: `Figure ${figNum}`,
            description: legend.title,
            name: `figure${figNum}`,
          },
        }),
      ],
    })
  );

  // Caption paragraph
  const captionRuns = [
    new TextRun({
      text: `Figure ${figNum}. `,
      bold: true,
      font: "Arial",
      size: 20,
    }),
    new TextRun({
      text: legend.title + " ",
      bold: true,
      font: "Arial",
      size: 20,
    }),
  ];

  const legendRuns = parseInlineFormatting(legend.text, {
    font: "Arial",
    size: 20,
  });
  captionRuns.push(...legendRuns);

  elements.push(
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 240 },
      children: captionRuns,
    })
  );

  return elements;
}

// ── Table 1 ────────────────────────────────────────────────────────────────────

function createTable1() {
  const headerStyle = {
    font: "Arial",
    size: 20,
    bold: true,
    color: "FFFFFF",
  };
  const cellStyle = { font: "Arial", size: 20 };
  const boldCellStyle = { font: "Arial", size: 20, bold: true };

  const headerShading = {
    type: ShadingType.CLEAR,
    fill: "2E74B5",
    color: "auto",
  };

  const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const thinBorder = { style: BorderStyle.SINGLE, size: 1, color: "B0B0B0" };

  const colWidths = [1800, 1400, 1400, 1100, 1100, 1800, 1700];
  const totalWidth = colWidths.reduce((a, b) => a + b, 0);

  const headers = [
    "Tool",
    "Comp. MAE (%)",
    "Cont. MAE (%)",
    "Comp. R\u00B2",
    "Cont. R\u00B2",
    "Speed (G/min/thread)",
    "Peak memory (GB)",
  ];

  const rows = [
    {
      cells: ["MAGICC", "2.74", "3.60", "0.90", "0.96", "1,451", "0.66"],
      bold: true,
    },
    {
      cells: ["CheckM2", "6.00", "23.14", "0.67", "0.34", "0.82", "18.76"],
      bold: false,
    },
    {
      cells: ["CoCoPyE", "6.62", "20.26", "0.65", "0.65", "0.70", "15.93"],
      bold: false,
    },
    {
      cells: [
        "DeepCheck",
        "10.10",
        "27.57",
        "0.48",
        "0.18",
        "0.82*",
        "1.46\u2020",
      ],
      bold: false,
    },
  ];

  function makeCell(text, style, isHeader, colIdx) {
    return new TableCell({
      width: { size: colWidths[colIdx], type: WidthType.DXA },
      shading: isHeader ? headerShading : undefined,
      borders: {
        top: thinBorder,
        bottom: thinBorder,
        left: thinBorder,
        right: thinBorder,
      },
      children: [
        new Paragraph({
          alignment: colIdx === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
          spacing: { before: 40, after: 40 },
          children: [new TextRun({ text, ...style })],
        }),
      ],
    });
  }

  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => makeCell(h, headerStyle, true, i)),
  });

  const dataRows = rows.map(
    (row) =>
      new TableRow({
        children: row.cells.map((c, i) =>
          makeCell(c, i === 0 && row.bold ? boldCellStyle : cellStyle, false, i)
        ),
      })
  );

  const table = new Table({
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows],
    width: { size: totalWidth, type: WidthType.DXA },
  });

  return table;
}

// ── References ─────────────────────────────────────────────────────────────────
const references = [
  {
    num: 1,
    text: "Parks, D. H. et al. Recovery of nearly 8,000 metagenome-assembled genomes substantially expands the tree of life. ",
    journal: "Nat. Microbiol.",
    rest: " 2, 1533\u20131542 (2017).",
  },
  {
    num: 2,
    text: "Nayfach, S. et al. A genomic catalog of Earth\u2019s microbiomes. ",
    journal: "Nat. Biotechnol.",
    rest: " 39, 499\u2013509 (2021).",
  },
  {
    num: 3,
    text: "Bowers, R. M. et al. Minimum information about a single amplified genome (MISAG) and a metagenome-assembled genome (MIMAG) of bacteria and archaea. ",
    journal: "Nat. Biotechnol.",
    rest: " 35, 725\u2013731 (2017).",
  },
  {
    num: 4,
    text: "Chklovski, A., Parks, D. H., Woodcroft, B. J. & Tyson, G. W. CheckM2: a rapid, scalable and accurate tool for assessing microbial genome quality using machine learning. ",
    journal: "Nat. Methods",
    rest: " 20, 1203\u20131212 (2023).",
  },
  {
    num: 5,
    text: "Parks, D. H., Imelfort, M., Skennerton, C. T., Hugenholtz, P. & Tyson, G. W. CheckM: assessing the quality of microbial genomes recovered from isolates, single cells, and metagenomes. ",
    journal: "Genome Res.",
    rest: " 25, 1043\u20131055 (2015).",
  },
  {
    num: 6,
    text: "Hyatt, D. et al. Prodigal: prokaryotic gene recognition and translation initiation site identification. ",
    journal: "BMC Bioinformatics",
    rest: " 11, 119 (2010).",
  },
  {
    num: 7,
    text: "Buchfink, B., Xie, C. & Huson, D. H. Fast and sensitive protein alignment using DIAMOND. ",
    journal: "Nat. Methods",
    rest: " 12, 59\u201360 (2015).",
  },
  {
    num: 8,
    text: "Cheng, R. et al. gcMeta: a Global Catalogue of Metagenomics platform to support the archiving, standardization and analysis of microbiome data. ",
    journal: "Nucleic Acids Res.",
    rest: " 54, D724\u2013D735 (2025).",
  },
  {
    num: 9,
    text: "\u00D6zsoy, E. D. & Clean, T. CoCoPyE: fast and accurate estimation of prokaryotic genome completeness and contamination using feature engineering. ",
    journal: "GigaScience",
    rest: " 13, giae079 (2024).",
  },
  {
    num: 10,
    text: "Liao, H. & Zhang, Z. DeepCheck: multitask learning aids in assessing microbial genome quality. ",
    journal: "Bioinformatics",
    rest: " 40, btae630 (2024).",
  },
  {
    num: 11,
    text: "Orakov, A. et al. GUNC: detection of chimerism and contamination in prokaryotic genomes. ",
    journal: "Genome Biol.",
    rest: " 22, 178 (2021).",
  },
  {
    num: 12,
    text: "Kang, D. D. et al. MetaBAT 2: an adaptive binning algorithm for robust and efficient genome reconstruction from metagenome assemblies. ",
    journal: "PeerJ",
    rest: " 7, e7359 (2019).",
  },
  {
    num: 13,
    text: "Alneberg, J. et al. Binning metagenomic contigs by coverage and composition. ",
    journal: "Nat. Methods",
    rest: " 11, 1144\u20131146 (2014).",
  },
  {
    num: 14,
    text: "Parks, D. H. et al. GTDB: an ongoing census of bacterial and archaeal diversity through a phylogenetically consistent, rank normalized and complete genome-based taxonomy. ",
    journal: "Nucleic Acids Res.",
    rest: " 50, D199\u2013D207 (2022).",
  },
  {
    num: 15,
    text: "Wood, D. E. & Salzberg, S. L. Kraken: ultrafast metagenomic sequence classification using exact alignments. ",
    journal: "Genome Biol.",
    rest: " 15, R46 (2014).",
  },
];

function createReferenceParagraph(ref) {
  return new Paragraph({
    spacing: { after: 80, line: 276 },
    indent: { left: 360, hanging: 360 },
    children: [
      new TextRun({
        text: `${ref.num}. `,
        font: "Arial",
        size: 20,
      }),
      new TextRun({
        text: ref.text,
        font: "Arial",
        size: 20,
      }),
      new TextRun({
        text: ref.journal,
        font: "Arial",
        size: 20,
        italics: true,
      }),
      new TextRun({
        text: ref.rest,
        font: "Arial",
        size: 20,
      }),
    ],
  });
}

// ── Section-level paragraph builders ──────────────────────────────────────────
// Read the markdown and split into paragraphs for each section.

const mdContent = fs.readFileSync(
  path.join(BASE, "manuscript.md"),
  "utf-8"
);

// Helper: split text by blank lines into paragraph strings
function splitParagraphs(text) {
  return text
    .split(/\n\n+/)
    .map((p) => p.replace(/\n/g, " ").trim())
    .filter((p) => p.length > 0);
}

// ── Build document children ───────────────────────────────────────────────────
const children = [];

// ── Title ─────────────────────────────────────────────────────────────────────
children.push(
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 600, after: 400 },
    children: [
      new TextRun({
        text: "MAGICC: ultra-fast genome quality assessment using core gene k-mer profiles and deep learning",
        bold: true,
        font: "Arial",
        size: 32, // 16pt
      }),
    ],
  })
);

// ── Abstract ──────────────────────────────────────────────────────────────────
children.push(heading1("Abstract"));

const abstractText =
  "Metagenome-assembled genomes (MAGs) and single-cell amplified genomes (SAGs) are central to culture-independent microbiology, yet quality assessment remains a critical bottleneck. Existing tools such as CheckM2, CoCoPyE, and DeepCheck achieve reasonable completeness estimation but severely underestimate contamination\u2014particularly when contaminant sequences originate from phylogenetically distant organisms. Using NCBI-finished reference genomes with known ground truth, we show that at true contamination levels above 20%, these tools produce mean absolute errors (MAEs) of 26\u201339%, rendering contamination estimates unreliable for real-world MAGs and SAGs. Here we present MAGICC (Metagenome-Assembled Genome Inference of Completeness and Contamination), which replaces gene annotation with core gene k-mer profiling to capture the compositional shift that foreign DNA introduces. MAGICC combines 9,249 canonical 9-mer counts derived from bacterial and archaeal single-copy core genes with 26 assembly statistics in a dual-branch neural network featuring squeeze-and-excitation attention and cross-attention fusion. Trained on one million synthetic genomes spanning 110 phyla, MAGICC achieves 2.74% completeness MAE and 3.60% contamination MAE across five benchmark sets\u2014a 6.4-fold improvement in contamination accuracy over CheckM2. MAGICC processes genomes at 1,451 genomes per minute per thread (~1,700\u00D7 faster than CheckM2), requires only 0.66 GB of memory (28\u00D7 less than CheckM2), and is available as a single-command Python tool and web server.";

children.push(bodyParagraph(abstractText));

// ── Introduction ──────────────────────────────────────────────────────────────
children.push(heading1("Introduction"));

// Extract intro text from markdown (lines 9-20 roughly)
const introSection = mdContent.split("## Introduction")[1].split("## Results")[0].trim();
const introParagraphs = splitParagraphs(introSection);

for (const para of introParagraphs) {
  children.push(bodyParagraph(para));
}

// ── Results ───────────────────────────────────────────────────────────────────
children.push(heading1("Results"));

// Subsection: Existing tools systematically underestimate contamination
children.push(
  heading2(
    "Existing tools systematically underestimate contamination from foreign organisms"
  )
);

const existingToolsSection = mdContent
  .split("### Existing tools systematically underestimate contamination from foreign organisms")[1]
  .split("### MAGICC overview")[0]
  .trim();
const existingToolsParagraphs = splitParagraphs(existingToolsSection);

for (const para of existingToolsParagraphs) {
  children.push(bodyParagraph(para));
}

// ── Figure 1 (after "Existing tools systematically underestimate contamination") ──
children.push(...createFigureCorrect(1));

// Subsection: MAGICC overview
children.push(heading2("MAGICC overview"));

const overviewSection = mdContent
  .split("### MAGICC overview")[1]
  .split("### MAGICC achieves accurate predictions")[0]
  .trim();
const overviewParagraphs = splitParagraphs(overviewSection);

for (const para of overviewParagraphs) {
  children.push(bodyParagraph(para));
}

// ── Figure 2 (after "MAGICC overview") ──
children.push(...createFigureCorrect(2));

// Subsection: MAGICC achieves accurate predictions
children.push(
  heading2("MAGICC achieves accurate predictions across diverse benchmarks")
);

const benchSection = mdContent
  .split("### MAGICC achieves accurate predictions across diverse benchmarks")[1]
  .split("### Why k-mer features detect foreign contamination")[0]
  .trim();
const benchParagraphs = splitParagraphs(benchSection);

for (const para of benchParagraphs) {
  children.push(bodyParagraph(para));
}

// ── Figure 3 (after "MAGICC achieves accurate predictions") ──
children.push(...createFigureCorrect(3));

// Subsection: Why k-mer features detect foreign contamination
children.push(
  heading2("Why k-mer features detect foreign contamination")
);

const kmerSection = mdContent
  .split("### Why k-mer features detect foreign contamination")[1]
  .split("## Discussion")[0]
  .trim();
const kmerParagraphs = splitParagraphs(kmerSection);

for (const para of kmerParagraphs) {
  children.push(bodyParagraph(para));
}

// ── Discussion ────────────────────────────────────────────────────────────────
children.push(heading1("Discussion"));

const discussionSection = mdContent
  .split("## Discussion")[1]
  .split("## Methods")[0]
  .trim();
const discussionParagraphs = splitParagraphs(discussionSection);

for (const para of discussionParagraphs) {
  children.push(bodyParagraph(para));
}

// ── Methods ───────────────────────────────────────────────────────────────────
children.push(heading1("Methods"));

const methodSubsections = [
  "Reference genome curation",
  "K-mer feature selection",
  "Synthetic genome generation",
  "Feature extraction and normalization",
  "Neural network architecture and training",
  "Benchmark evaluation",
  "MAGICC inference pipeline",
];

for (let i = 0; i < methodSubsections.length; i++) {
  const subName = methodSubsections[i];
  children.push(heading2(subName));

  const startMarker = `### ${subName}`;
  let endMarker;
  if (i < methodSubsections.length - 1) {
    endMarker = `### ${methodSubsections[i + 1]}`;
  } else {
    endMarker = "## Data availability";
  }

  const sectionText = mdContent
    .split(startMarker)[1]
    .split(endMarker)[0]
    .trim();
  const paragraphs = splitParagraphs(sectionText);

  for (const para of paragraphs) {
    children.push(bodyParagraph(para));
  }
}

// ── Data availability ─────────────────────────────────────────────────────────
children.push(heading1("Data availability"));

const dataAvailSection = mdContent
  .split("## Data availability")[1]
  .split("## Code availability")[0]
  .trim();
const dataAvailParagraphs = splitParagraphs(dataAvailSection);

for (const para of dataAvailParagraphs) {
  children.push(bodyParagraph(para));
}

// ── Code availability ─────────────────────────────────────────────────────────
children.push(heading1("Code availability"));

const codeAvailSection = mdContent
  .split("## Code availability")[1]
  .split("## References")[0]
  .trim();
const codeAvailParagraphs = splitParagraphs(codeAvailSection);

for (const para of codeAvailParagraphs) {
  children.push(bodyParagraph(para));
}

// ── References ────────────────────────────────────────────────────────────────
children.push(heading1("References"));

for (const ref of references) {
  children.push(createReferenceParagraph(ref));
}

// ── Table 1 ───────────────────────────────────────────────────────────────────
// Page break before table
children.push(
  new Paragraph({
    children: [new PageBreak()],
  })
);

children.push(
  new Paragraph({
    spacing: { before: 200, after: 200 },
    children: [
      new TextRun({
        text: "Table 1. ",
        bold: true,
        font: "Arial",
        size: 22,
      }),
      new TextRun({
        text: "Benchmark results across five evaluation sets (5,000 genomes total).",
        bold: true,
        font: "Arial",
        size: 22,
      }),
    ],
  })
);

children.push(createTable1());

// Table footnotes
children.push(
  new Paragraph({
    spacing: { before: 80, after: 40 },
    children: [
      new TextRun({
        text: "*DeepCheck requires CheckM2 feature extraction; effective speed equals CheckM2.",
        font: "Arial",
        size: 18, // 9pt
        italics: true,
      }),
    ],
  })
);

children.push(
  new Paragraph({
    spacing: { after: 200 },
    children: [
      new TextRun({
        text: "\u2020Inference-only memory; requires CheckM2 pipeline for feature extraction.",
        font: "Arial",
        size: 18,
        italics: true,
      }),
    ],
  })
);

// ── Build Document ────────────────────────────────────────────────────────────

const doc = new Document({
  styles: {
    default: {
      document: {
        run: {
          font: "Arial",
          size: 22, // 11pt
        },
      },
    },
  },
  numbering: {
    config: [],
  },
  sections: [
    {
      properties: {
        page: {
          margin: {
            top: convertInchesToTwip(1),
            bottom: convertInchesToTwip(1),
            left: convertInchesToTwip(1),
            right: convertInchesToTwip(1),
          },
        },
      },
      headers: {},
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [
                new TextRun({
                  children: [PageNumber.CURRENT],
                  font: "Arial",
                  size: 18,
                }),
              ],
            }),
          ],
        }),
      },
      children: children,
    },
  ],
});

// ── Write ─────────────────────────────────────────────────────────────────────
Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(OUTPUT, buffer);
  console.log(`Manuscript written to ${OUTPUT}`);
  console.log(`File size: ${(buffer.length / 1024 / 1024).toFixed(2)} MB`);
});
