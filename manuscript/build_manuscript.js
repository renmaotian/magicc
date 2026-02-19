const fs = require("fs");
const path = require("path");
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  ImageRun,
  Table,
  TableRow,
  TableCell,
  TableBorders,
  AlignmentType,
  HeadingLevel,
  PageNumber,
  Footer,
  BorderStyle,
  ShadingType,
  WidthType,
  LevelFormat,
  convertInchesToTwip,
  PageBreak,
  VerticalAlign,
} = require("docx");

// ---------------------------------------------------------------------------
// Read manuscript markdown
// ---------------------------------------------------------------------------
const mdPath = path.join(__dirname, "manuscript.md");
const mdText = fs.readFileSync(mdPath, "utf-8");

// Read figure images
const fig1 = fs.readFileSync(path.join(__dirname, "figures/figure1_motivating.png"));
const fig2 = fs.readFileSync(path.join(__dirname, "figures/figure2_workflow.png"));
const fig3 = fs.readFileSync(path.join(__dirname, "figures/figure3_benchmark.png"));

// ---------------------------------------------------------------------------
// Parse the markdown into sections
// ---------------------------------------------------------------------------
const lines = mdText.split("\n");

function extractSections(lines) {
  const sections = [];
  let current = null;
  for (const line of lines) {
    const h1 = line.match(/^# (.+)/);
    const h2 = line.match(/^## (.+)/);
    const h3 = line.match(/^### (.+)/);
    if (h1 && !line.startsWith("##")) {
      current = { level: 1, title: h1[1].trim(), content: [] };
      sections.push(current);
    } else if (h2 && !line.startsWith("###")) {
      current = { level: 2, title: h2[1].trim(), content: [] };
      sections.push(current);
    } else if (h3) {
      current = { level: 3, title: h3[1].trim(), content: [] };
      sections.push(current);
    } else if (current) {
      current.content.push(line);
    }
  }
  return sections;
}

const sections = extractSections(lines);

function findSection(title) {
  return sections.find((s) => s.title === title);
}

function getSectionText(title) {
  const s = findSection(title);
  if (!s) return "";
  return s.content.join("\n").trim();
}

// ---------------------------------------------------------------------------
// Inline formatting parser
// ---------------------------------------------------------------------------
function parseInlineFormatting(text) {
  if (!text || text.trim() === "") return [];

  const runs = [];
  // Regex to match: superscript ^...^, bold **...**, italic *...*, or plain text
  const regex = /(\*\*(.+?)\*\*)|(\*(.+?)\*)|(\^([^^]+?)\^)|([^*^]+)/g;
  let match;

  while ((match = regex.exec(text)) !== null) {
    if (match[1]) {
      // Bold **...**
      runs.push(
        new TextRun({
          text: match[2],
          bold: true,
          font: "Times New Roman",
          size: 22,
        })
      );
    } else if (match[3]) {
      // Italic *...*
      runs.push(
        new TextRun({
          text: match[4],
          italics: true,
          font: "Times New Roman",
          size: 22,
        })
      );
    } else if (match[5]) {
      // Superscript ^...^
      runs.push(
        new TextRun({
          text: match[6],
          superScript: true,
          font: "Times New Roman",
          size: 22,
        })
      );
    } else if (match[7]) {
      // Plain text
      runs.push(
        new TextRun({
          text: match[7],
          font: "Times New Roman",
          size: 22,
        })
      );
    }
  }
  return runs;
}

// Parse inline formatting but allow custom defaults (e.g., for smaller text)
function parseInlineFormattingWithDefaults(text, defaults) {
  if (!text || text.trim() === "") return [];

  const runs = [];
  const regex = /(\*\*(.+?)\*\*)|(\*(.+?)\*)|(\^([^^]+?)\^)|([^*^]+)/g;
  let match;

  while ((match = regex.exec(text)) !== null) {
    if (match[1]) {
      runs.push(new TextRun({ ...defaults, text: match[2], bold: true }));
    } else if (match[3]) {
      runs.push(new TextRun({ ...defaults, text: match[4], italics: true }));
    } else if (match[5]) {
      runs.push(new TextRun({ ...defaults, text: match[6], superScript: true }));
    } else if (match[7]) {
      runs.push(new TextRun({ ...defaults, text: match[7] }));
    }
  }
  return runs;
}

// ---------------------------------------------------------------------------
// Helper: create paragraphs from section text
// ---------------------------------------------------------------------------
function textToParagraphs(text) {
  const paragraphs = [];
  // Split by blank lines to get paragraphs
  const blocks = text.split(/\n\n+/);
  for (const block of blocks) {
    const trimmed = block.replace(/\n/g, " ").trim();
    if (trimmed === "") continue;
    // Skip table markdown
    if (trimmed.startsWith("|")) continue;
    // Skip footnote lines that start with *
    if (trimmed.startsWith("*DeepCheck") || trimmed.startsWith("†")) continue;
    paragraphs.push(
      new Paragraph({
        children: parseInlineFormatting(trimmed),
        spacing: { before: 120, after: 120, line: 276 },
      })
    );
  }
  return paragraphs;
}

// ---------------------------------------------------------------------------
// Heading helpers
// ---------------------------------------------------------------------------
function h1Paragraph(text) {
  return new Paragraph({
    children: [
      new TextRun({
        text: text,
        bold: true,
        font: "Arial",
        size: 28, // 14pt = 28 half-points
      }),
    ],
    spacing: { before: 240, after: 120, line: 276 },
  });
}

function h2Paragraph(text) {
  return new Paragraph({
    children: [
      new TextRun({
        text: text,
        bold: true,
        font: "Arial",
        size: 24, // 12pt
      }),
    ],
    spacing: { before: 240, after: 120, line: 276 },
  });
}

function h3Paragraph(text) {
  return new Paragraph({
    children: [
      new TextRun({
        text: text,
        bold: true,
        italics: true,
        font: "Arial",
        size: 22, // 11pt
      }),
    ],
    spacing: { before: 200, after: 120, line: 276 },
  });
}

// ---------------------------------------------------------------------------
// Figure helpers
// ---------------------------------------------------------------------------
function figureParagraph(imageBuffer, width, height, figNum) {
  return new Paragraph({
    children: [
      new ImageRun({
        data: imageBuffer,
        transformation: { width, height },
        type: "png",
        altText: {
          title: `Figure ${figNum}`,
          description: `Figure ${figNum} of the MAGICC manuscript`,
          name: `figure${figNum}`,
        },
      }),
    ],
    alignment: AlignmentType.CENTER,
    spacing: { before: 240, after: 120 },
  });
}

// ---------------------------------------------------------------------------
// Figure caption helpers
// ---------------------------------------------------------------------------
function figureCaptionParagraph(figNum, captionText) {
  // captionText is the full legend text from the manuscript
  // We need "Figure N." bold, then parse the rest for inline formatting
  // Also bold panel labels like (a), (b) etc.

  const children = [
    new TextRun({
      text: `Figure ${figNum}. `,
      bold: true,
      font: "Times New Roman",
      size: 22,
    }),
  ];

  // Parse the rest of the caption - need to bold panel labels (a), (b), etc.
  // Split on panel labels pattern
  const parts = captionText.split(/(\([a-g]\))/g);
  for (const part of parts) {
    if (/^\([a-g]\)$/.test(part)) {
      children.push(
        new TextRun({
          text: part,
          bold: true,
          font: "Times New Roman",
          size: 22,
        })
      );
    } else {
      // Parse for inline formatting (bold, italic, superscript)
      const runs = parseInlineFormatting(part);
      children.push(...runs);
    }
  }

  return new Paragraph({
    children,
    spacing: { before: 60, after: 240, line: 276 },
  });
}

// ---------------------------------------------------------------------------
// Extract figure legends from the markdown
// ---------------------------------------------------------------------------
function extractFigureLegend(figNum) {
  const legendSection = findSection("Figure legends");
  if (!legendSection) return "";
  const text = legendSection.content.join("\n");
  // Find the legend for figNum
  const pattern = new RegExp(
    `\\*\\*Figure ${figNum}\\.\\s*(.+?)\\*\\*\\s*(.+?)(?=\\*\\*Figure \\d|$)`,
    "s"
  );
  const m = text.match(pattern);
  if (m) {
    return m[1].trim() + " " + m[2].trim();
  }
  return "";
}

const fig1Legend = extractFigureLegend(1);
const fig2Legend = extractFigureLegend(2);
const fig3Legend = extractFigureLegend(3);

// ---------------------------------------------------------------------------
// Table 1 builder
// ---------------------------------------------------------------------------
function buildTable1() {
  const headerStyle = {
    font: "Times New Roman",
    size: 20,
    bold: true,
  };
  const cellStyle = {
    font: "Times New Roman",
    size: 20,
  };
  const boldCellStyle = {
    font: "Times New Roman",
    size: 20,
    bold: true,
  };

  const headers = [
    "Tool",
    "Comp. MAE (%)",
    "Cont. MAE (%)",
    "Comp. R\u00B2",
    "Cont. R\u00B2",
    "Speed (G/min/thread)",
    "Peak memory (GB)",
  ];

  const data = [
    {
      values: ["MAGICC", "2.74", "3.60", "0.90", "0.96", "1,451", "0.66"],
      bold: true,
    },
    {
      values: ["CheckM2", "6.00", "23.14", "0.67", "0.34", "0.82", "18.76"],
      bold: false,
    },
    {
      values: ["CoCoPyE", "6.62", "20.26", "0.65", "0.65", "0.70", "15.93"],
      bold: false,
    },
    {
      values: [
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

  // Column widths in DXA (total page width minus margins = 6.5 inches = 9360 DXA)
  const colWidths = [1400, 1300, 1300, 1100, 1100, 1660, 1500];
  const totalWidth = colWidths.reduce((a, b) => a + b, 0);

  function makeHeaderCell(text, width) {
    return new TableCell({
      children: [
        new Paragraph({
          children: [new TextRun({ ...headerStyle, text })],
          alignment: AlignmentType.CENTER,
          spacing: { before: 40, after: 40 },
        }),
      ],
      width: { size: width, type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: "D9E2F3" },
      verticalAlign: VerticalAlign.CENTER,
    });
  }

  function makeDataCell(text, width, isBold) {
    const style = isBold ? boldCellStyle : cellStyle;
    return new TableCell({
      children: [
        new Paragraph({
          children: [new TextRun({ ...style, text })],
          alignment: AlignmentType.CENTER,
          spacing: { before: 40, after: 40 },
        }),
      ],
      width: { size: width, type: WidthType.DXA },
      verticalAlign: VerticalAlign.CENTER,
    });
  }

  const headerRow = new TableRow({
    children: headers.map((h, i) => makeHeaderCell(h, colWidths[i])),
    tableHeader: true,
  });

  const dataRows = data.map(
    (row) =>
      new TableRow({
        children: row.values.map((v, i) =>
          makeDataCell(v, colWidths[i], row.bold)
        ),
      })
  );

  return new Table({
    rows: [headerRow, ...dataRows],
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: colWidths,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
      bottom: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
      left: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
      right: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
      insideVertical: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
    },
  });
}

// Table 1 caption
function table1Caption() {
  return new Paragraph({
    children: [
      new TextRun({
        text: "Table 1. ",
        bold: true,
        font: "Times New Roman",
        size: 22,
      }),
      new TextRun({
        text: "Benchmark results across five evaluation sets (5,000 genomes total).",
        font: "Times New Roman",
        size: 22,
      }),
    ],
    spacing: { before: 240, after: 120, line: 276 },
  });
}

// Table footnotes
function table1Footnotes() {
  return [
    new Paragraph({
      children: [
        new TextRun({
          text: "*DeepCheck requires CheckM2 feature extraction; effective speed equals CheckM2.",
          italics: true,
          font: "Times New Roman",
          size: 18,
        }),
      ],
      spacing: { before: 60, after: 0, line: 276 },
    }),
    new Paragraph({
      children: [
        new TextRun({
          text: "\u2020Inference-only memory; requires CheckM2 pipeline for feature extraction.",
          italics: true,
          font: "Times New Roman",
          size: 18,
        }),
      ],
      spacing: { before: 0, after: 240, line: 276 },
    }),
  ];
}

// ---------------------------------------------------------------------------
// References builder
// ---------------------------------------------------------------------------
function buildReferences() {
  const refSection = findSection("References");
  if (!refSection) return [];
  const text = refSection.content.join("\n").trim();
  const refLines = text.split("\n").filter((l) => l.trim() !== "");
  const paragraphs = [];

  for (const line of refLines) {
    const trimmed = line.trim();
    // Each line is like "1. Parks, D. H. et al. ..."
    const m = trimmed.match(/^(\d+)\.\s+(.+)$/);
    if (m) {
      const num = m[1];
      const refText = m[2];
      const children = [
        new TextRun({
          text: `${num}. `,
          font: "Times New Roman",
          size: 22,
        }),
      ];
      // Parse inline formatting for journal names in italic
      children.push(...parseInlineFormatting(refText));
      paragraphs.push(
        new Paragraph({
          children,
          spacing: { before: 60, after: 60, line: 276 },
          indent: { left: 360, hanging: 360 },
        })
      );
    }
  }
  return paragraphs;
}

// ---------------------------------------------------------------------------
// Build sub-section paragraphs with text content
// ---------------------------------------------------------------------------
function getSubsectionParagraphs(title) {
  const s = findSection(title);
  if (!s) return [];
  const text = s.content.join("\n").trim();
  return textToParagraphs(text);
}

// Special handler for "MAGICC achieves accurate predictions" section
// which includes Table 1 inline
function getResultsAccurateParagraphs() {
  const s = findSection("MAGICC achieves accurate predictions across diverse benchmarks");
  if (!s) return [];
  const text = s.content.join("\n").trim();
  const blocks = text.split(/\n\n+/);
  const result = [];

  for (const block of blocks) {
    const trimmed = block.replace(/\n/g, " ").trim();
    if (trimmed === "") continue;
    // Skip table markdown rows
    if (trimmed.startsWith("|")) continue;
    // Skip footnotes
    if (trimmed.startsWith("*DeepCheck") || trimmed.startsWith("\u2020")) continue;

    // Check if this block contains the table reference "**Table 1..."
    if (trimmed.startsWith("**Table 1.")) {
      // Insert table 1 here
      result.push(table1Caption());
      result.push(buildTable1());
      result.push(...table1Footnotes());
    } else {
      result.push(
        new Paragraph({
          children: parseInlineFormatting(trimmed),
          spacing: { before: 120, after: 120, line: 276 },
        })
      );
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Methods subsections
// ---------------------------------------------------------------------------
function getMethodsSubsections() {
  const methodsSubs = [
    "Reference genome curation",
    "K-mer feature selection",
    "Synthetic genome generation",
    "Feature extraction and normalization",
    "Neural network architecture and training",
    "Benchmark evaluation",
    "MAGICC inference pipeline",
  ];

  const result = [];
  for (const sub of methodsSubs) {
    const s = findSection(sub);
    if (s) {
      result.push(h3Paragraph(sub));
      result.push(...textToParagraphs(s.content.join("\n").trim()));
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Build the document
// ---------------------------------------------------------------------------
async function build() {
  const children = [];

  // === TITLE ===
  children.push(
    new Paragraph({
      children: [
        new TextRun({
          text: "MAGICC: ultra-fast genome quality assessment using core gene k-mer profiles and deep learning",
          bold: true,
          font: "Times New Roman",
          size: 28, // 14pt
        }),
      ],
      alignment: AlignmentType.CENTER,
      spacing: { before: 480, after: 480, line: 276 },
    })
  );

  // === ABSTRACT ===
  children.push(h1Paragraph("Abstract"));
  const abstractSection = findSection("Abstract");
  if (abstractSection) {
    const aText = abstractSection.content.join("\n").trim();
    children.push(...textToParagraphs(aText));
  }

  // === INTRODUCTION ===
  children.push(h1Paragraph("Introduction"));
  const introSection = findSection("Introduction");
  if (introSection) {
    children.push(...textToParagraphs(introSection.content.join("\n").trim()));
  }

  // === RESULTS ===
  children.push(h1Paragraph("Results"));

  // -- Subsection: Existing tools...
  children.push(
    h2Paragraph(
      "Existing tools systematically underestimate contamination"
    )
  );
  children.push(
    ...getSubsectionParagraphs(
      "Existing tools systematically underestimate contamination from foreign organisms"
    )
  );

  // -- INSERT Figure 1
  children.push(figureParagraph(fig1, 460, 690, 1));
  children.push(figureCaptionParagraph(1, fig1Legend));

  // -- Subsection: MAGICC overview
  children.push(h2Paragraph("MAGICC overview"));
  children.push(...getSubsectionParagraphs("MAGICC overview"));

  // -- INSERT Figure 2
  children.push(figureParagraph(fig2, 600, 200, 2));
  children.push(figureCaptionParagraph(2, fig2Legend));

  // -- Subsection: MAGICC achieves accurate predictions
  children.push(h2Paragraph("MAGICC achieves accurate predictions"));
  children.push(...getResultsAccurateParagraphs());

  // -- INSERT Figure 3
  children.push(figureParagraph(fig3, 460, 690, 3));
  children.push(figureCaptionParagraph(3, fig3Legend));

  // -- Subsection: MAGICC is computationally efficient
  children.push(h2Paragraph("MAGICC is computationally efficient"));
  children.push(
    ...getSubsectionParagraphs("MAGICC is computationally efficient")
  );

  // === DISCUSSION ===
  children.push(h1Paragraph("Discussion"));
  const discSection = findSection("Discussion");
  if (discSection) {
    children.push(...textToParagraphs(discSection.content.join("\n").trim()));
  }

  // === METHODS ===
  children.push(h1Paragraph("Methods"));
  children.push(...getMethodsSubsections());

  // === DATA AVAILABILITY ===
  children.push(h1Paragraph("Data availability"));
  const dataAvail = findSection("Data availability");
  if (dataAvail) {
    children.push(...textToParagraphs(dataAvail.content.join("\n").trim()));
  }

  // === CODE AVAILABILITY ===
  children.push(h1Paragraph("Code availability"));
  const codeAvail = findSection("Code availability");
  if (codeAvail) {
    children.push(...textToParagraphs(codeAvail.content.join("\n").trim()));
  }

  // === REFERENCES ===
  children.push(h1Paragraph("References"));
  children.push(...buildReferences());

  // ---------------------------------------------------------------------------
  // Create document
  // ---------------------------------------------------------------------------
  const doc = new Document({
    styles: {
      default: {
        document: {
          run: {
            font: "Times New Roman",
            size: 22,
          },
          paragraph: {
            spacing: { before: 120, after: 120, line: 276 },
          },
        },
      },
    },
    sections: [
      {
        properties: {
          page: {
            size: {
              width: convertInchesToTwip(8.5),
              height: convertInchesToTwip(11),
            },
            margin: {
              top: convertInchesToTwip(1),
              bottom: convertInchesToTwip(1),
              left: convertInchesToTwip(1),
              right: convertInchesToTwip(1),
            },
          },
        },
        footers: {
          default: new Footer({
            children: [
              new Paragraph({
                children: [
                  new TextRun({
                    children: [PageNumber.CURRENT],
                    font: "Times New Roman",
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

  // Pack and write
  const buffer = await Packer.toBuffer(doc);
  const outPath = path.join(__dirname, "manuscript.docx");
  fs.writeFileSync(outPath, buffer);
  console.log(`Manuscript written to ${outPath} (${buffer.length} bytes)`);
}

build().catch((err) => {
  console.error("Build failed:", err);
  process.exit(1);
});
