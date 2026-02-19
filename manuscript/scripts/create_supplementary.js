const fs = require("fs");
const path = require("path");
const docx = require(path.join(__dirname, "..", "node_modules", "docx"));

const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  Table,
  TableRow,
  TableCell,
  ImageRun,
  AlignmentType,
  HeadingLevel,
  BorderStyle,
  ShadingType,
  WidthType,
  PageNumber,
  Footer,
  PageBreak,
  TableLayoutType,
  convertInchesToTwip,
} = docx;

// ── Constants ──────────────────────────────────────────────────────────
const FONT = "Arial";
const BODY_SIZE = 22; // 11pt in half-points
const SMALL_SIZE = 20; // 10pt
const HEADING1_SIZE = 32; // 16pt
const HEADING2_SIZE = 26; // 13pt
const HEADING3_SIZE = 24; // 12pt
const TITLE_SIZE = 40; // 20pt
const SUBTITLE_SIZE = 24; // 12pt

const PAGE_WIDTH_DXA = 12240; // 8.5 inches
const MARGIN_DXA = 1440; // 1 inch
const CONTENT_WIDTH_DXA = PAGE_WIDTH_DXA - 2 * MARGIN_DXA; // 9360 DXA = 6.5 inches

const HEADER_SHADING = { type: ShadingType.CLEAR, color: "auto", fill: "D6E4F0" }; // light blue
const BORDER_STYLE = {
  style: BorderStyle.SINGLE,
  size: 1,
  color: "999999",
};
const TABLE_BORDERS = {
  top: BORDER_STYLE,
  bottom: BORDER_STYLE,
  left: BORDER_STYLE,
  right: BORDER_STYLE,
  insideHorizontal: BORDER_STYLE,
  insideVertical: BORDER_STYLE,
};

const FIG_DIR = path.join(__dirname, "..", "figures");

// ── Helper: parse text with bold (**), superscripts (^), and special chars ──
function parseInlineFormatting(text, baseOpts = {}) {
  const runs = [];
  // Split on **bold**, ^superscript patterns, and \* escaped asterisks
  // Process segments: bold (**...**), superscript (10^-3, R^2, etc.)
  let remaining = text;

  while (remaining.length > 0) {
    // Check for bold **...**
    const boldMatch = remaining.match(/^\*\*(.+?)\*\*/);
    if (boldMatch) {
      runs.push(...parseSuper(boldMatch[1], { ...baseOpts, bold: true }));
      remaining = remaining.slice(boldMatch[0].length);
      continue;
    }

    // Find the next ** marker
    const nextBold = remaining.indexOf("**");
    if (nextBold === -1) {
      // No more bold, process the rest for superscripts
      runs.push(...parseSuper(remaining, baseOpts));
      break;
    } else if (nextBold > 0) {
      // Process text before the bold marker
      runs.push(...parseSuper(remaining.slice(0, nextBold), baseOpts));
      remaining = remaining.slice(nextBold);
    } else {
      // nextBold === 0, already handled above
      // Malformed, just push as-is
      runs.push(new TextRun({ text: remaining, font: FONT, size: baseOpts.size || BODY_SIZE, ...baseOpts }));
      break;
    }
  }
  return runs;
}

// Parse superscript notation: 10^-3, R^2, etc.
function parseSuper(text, opts = {}) {
  const runs = [];
  // Match patterns like ^-3, ^2, ^-4 (with optional surrounding text)
  const superRegex = /\^(\{[^}]+\}|[-\d]+)/g;
  let lastIndex = 0;
  let match;

  while ((match = superRegex.exec(text)) !== null) {
    // Text before superscript
    if (match.index > lastIndex) {
      const before = text.slice(lastIndex, match.index);
      if (before) {
        runs.push(new TextRun({
          text: before,
          font: FONT,
          size: opts.size || BODY_SIZE,
          bold: opts.bold || false,
        }));
      }
    }
    // Superscript content
    let superText = match[1];
    if (superText.startsWith("{") && superText.endsWith("}")) {
      superText = superText.slice(1, -1);
    }
    runs.push(new TextRun({
      text: superText,
      font: FONT,
      size: opts.size || BODY_SIZE,
      bold: opts.bold || false,
      superScript: true,
    }));
    lastIndex = match.index + match[0].length;
  }

  // Remaining text after last superscript
  if (lastIndex < text.length) {
    const rest = text.slice(lastIndex);
    if (rest) {
      runs.push(new TextRun({
        text: rest,
        font: FONT,
        size: opts.size || BODY_SIZE,
        bold: opts.bold || false,
      }));
    }
  }

  if (runs.length === 0 && text.length > 0) {
    runs.push(new TextRun({
      text: text,
      font: FONT,
      size: opts.size || BODY_SIZE,
      bold: opts.bold || false,
    }));
  }

  return runs;
}

// ── Helper: make a paragraph ──
function makeParagraph(text, options = {}) {
  const {
    bold = false,
    size = BODY_SIZE,
    alignment = AlignmentType.LEFT,
    spacing = {},
    heading,
  } = options;

  const runs = parseInlineFormatting(text, { size, bold });

  return new Paragraph({
    children: runs,
    alignment,
    spacing: { after: 120, ...spacing },
    ...(heading ? { heading } : {}),
  });
}

// ── Helper: create table cell ──
function makeCell(text, options = {}) {
  const {
    bold = false,
    isHeader = false,
    alignment = AlignmentType.LEFT,
    width,
    shading,
  } = options;

  const cellBold = isHeader ? true : bold;
  const cellShading = isHeader ? HEADER_SHADING : shading;

  const textStr = String(text).replace(/\\\*/g, "*");
  const runs = parseInlineFormatting(textStr, {
    size: SMALL_SIZE,
    bold: cellBold,
  });

  const cellOpts = {
    children: [
      new Paragraph({
        children: runs,
        alignment: alignment,
        spacing: { before: 40, after: 40 },
      }),
    ],
    borders: TABLE_BORDERS,
  };

  if (width) {
    cellOpts.width = { size: width, type: WidthType.DXA };
  }
  if (cellShading) {
    cellOpts.shading = cellShading;
  }

  return new TableCell(cellOpts);
}

// ── Helper: detect alignment from markdown separator ──
function detectAlignment(sep) {
  const trimmed = sep.trim();
  if (trimmed.startsWith(":") && trimmed.endsWith(":")) return AlignmentType.CENTER;
  if (trimmed.endsWith(":")) return AlignmentType.RIGHT;
  return AlignmentType.LEFT;
}

// ── Helper: check if text looks numeric ──
function looksNumeric(text) {
  const cleaned = text.replace(/[*%,]/g, "").trim();
  return /^[\d.,\-~]+/.test(cleaned) || cleaned === "--" || cleaned === "";
}

// ── Parse a markdown table ──
function parseMarkdownTable(lines) {
  // lines: array of | ... | lines
  // First line is header, second is separator, rest are data
  if (lines.length < 3) return null;

  const parseRow = (line) => {
    return line
      .split("|")
      .slice(1, -1) // remove leading/trailing empty from |
      .map((cell) => cell.trim());
  };

  const headers = parseRow(lines[0]);
  const separators = parseRow(lines[1]);
  const alignments = separators.map(detectAlignment);
  const dataRows = lines.slice(2).map(parseRow);

  return { headers, alignments, dataRows };
}

// ── Build a Word table from parsed markdown table ──
function buildTable(parsed) {
  const { headers, alignments, dataRows } = parsed;
  const numCols = headers.length;
  const colWidth = Math.floor(CONTENT_WIDTH_DXA / numCols);
  const columnWidths = Array(numCols).fill(colWidth);
  // Adjust last column to absorb rounding
  columnWidths[numCols - 1] = CONTENT_WIDTH_DXA - colWidth * (numCols - 1);

  const headerRow = new TableRow({
    children: headers.map((h, i) =>
      makeCell(h, {
        isHeader: true,
        alignment: alignments[i] || AlignmentType.CENTER,
        width: columnWidths[i],
      })
    ),
    tableHeader: true,
  });

  const rows = dataRows.map((row) => {
    // Ensure row has correct number of cells
    while (row.length < numCols) row.push("");
    return new TableRow({
      children: row.map((cell, i) => {
        const isBoldRow = cell.startsWith("**") && cell.endsWith("**");
        // For header-like rows (e.g., "Filter criteria"), use light shading
        const isSubHeader =
          row[0].replace(/\*\*/g, "").trim().startsWith("Filter criteria") ||
          row[0].replace(/\*\*/g, "").trim().startsWith("After all filters") ||
          row[0].replace(/\*\*/g, "").trim().startsWith("Train / Val / Test split") ||
          row[0].replace(/\*\*/g, "").trim().startsWith("Total per batch") ||
          row[0].replace(/\*\*/g, "").trim().startsWith("Total") && row[0].includes("**") ||
          row[0].replace(/\*\*/g, "").trim().startsWith("Final merged") ||
          row[0].replace(/\*\*/g, "").trim().startsWith("Selected for project") ||
          row[0].replace(/\*\*/g, "").trim().startsWith("Selected k-mers");
        const shading = isSubHeader
          ? { type: ShadingType.CLEAR, color: "auto", fill: "E8EFF7" }
          : undefined;

        // Detect alignment: use column alignment for numeric data, left for text
        let cellAlign = alignments[i];
        // First column is always left-aligned if it's text
        if (i === 0) cellAlign = AlignmentType.LEFT;

        return makeCell(cell, {
          bold: false,
          alignment: cellAlign,
          width: columnWidths[i],
          shading,
        });
      }),
    });
  });

  return new Table({
    rows: [headerRow, ...rows],
    width: { size: CONTENT_WIDTH_DXA, type: WidthType.DXA },
    columnWidths,
    borders: TABLE_BORDERS,
    layout: TableLayoutType.FIXED,
  });
}

// ── Build section separator ──
function sectionSeparator() {
  return new Paragraph({
    children: [],
    spacing: { before: 200, after: 200 },
    border: {
      bottom: {
        style: BorderStyle.SINGLE,
        size: 1,
        color: "CCCCCC",
        space: 1,
      },
    },
  });
}

// ── Build a page break paragraph ──
function pageBreakParagraph() {
  return new Paragraph({
    children: [new PageBreak()],
  });
}

// ── Build figure with caption ──
function buildFigure(figPath, captionTitle, captionLegend, figNum) {
  const imgData = fs.readFileSync(figPath);

  // Original: 1995 x 915 px
  // Scale to fit 6.5 inches width (468 pt = 6.5 * 72), maintain aspect ratio
  // 468 pt width, height = 468 * (915/1995) = 214.7 pt
  const widthPt = 468;
  const heightPt = Math.round(468 * (915 / 1995));

  const elements = [];

  // Image paragraph
  elements.push(
    new Paragraph({
      children: [
        new ImageRun({
          data: imgData,
          transformation: {
            width: widthPt,
            height: heightPt,
          },
          type: "png",
          altText: {
            title: `Figure S${figNum}`,
            description: captionTitle,
            name: `figS${figNum}`,
          },
        }),
      ],
      alignment: AlignmentType.CENTER,
      spacing: { before: 240, after: 120 },
    })
  );

  // Caption paragraph: "Figure SN. Title" (bold) + legend (normal)
  elements.push(
    new Paragraph({
      children: [
        new TextRun({
          text: `Figure S${figNum}. `,
          font: FONT,
          size: SMALL_SIZE,
          bold: true,
        }),
        new TextRun({
          text: captionTitle + " ",
          font: FONT,
          size: SMALL_SIZE,
          bold: true,
        }),
        ...parseInlineFormatting(captionLegend, { size: SMALL_SIZE, bold: false }),
      ],
      alignment: AlignmentType.CENTER,
      spacing: { after: 240 },
    })
  );

  return elements;
}

// ── Parse the supplementary.md and build document ──
function buildDocument() {
  const mdPath = path.join(__dirname, "..", "supplementary.md");
  const mdContent = fs.readFileSync(mdPath, "utf-8");
  const lines = mdContent.split("\n");

  const children = [];

  // ── Title ──
  children.push(
    new Paragraph({
      children: [
        new TextRun({
          text: "Supplementary Information",
          font: FONT,
          size: TITLE_SIZE,
          bold: true,
        }),
      ],
      alignment: AlignmentType.CENTER,
      spacing: { after: 120 },
    })
  );

  // ── Subtitle ──
  children.push(
    new Paragraph({
      children: [
        new TextRun({
          text: "MAGICC: ultra-fast genome quality assessment using core gene k-mer profiles and deep learning",
          font: FONT,
          size: SUBTITLE_SIZE,
          italics: true,
        }),
      ],
      alignment: AlignmentType.CENTER,
      spacing: { after: 400 },
    })
  );

  children.push(sectionSeparator());

  // ── Process markdown content ──
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];

    // Skip the title and subtitle lines (already handled)
    if (line.startsWith("# Supplementary Information") || (line.startsWith("**MAGICC:") && line.endsWith("**"))) {
      i++;
      continue;
    }

    // Section separator
    if (line.trim() === "---") {
      children.push(sectionSeparator());
      i++;
      continue;
    }

    // Table heading ## Table S...
    if (line.startsWith("## Table S")) {
      // Add page break before each table section (except possibly the first)
      if (children.length > 4) {
        children.push(pageBreakParagraph());
      }
      const titleText = line.replace(/^## /, "");
      children.push(
        new Paragraph({
          children: parseInlineFormatting(titleText, { size: HEADING2_SIZE, bold: true }),
          alignment: AlignmentType.LEFT,
          spacing: { before: 240, after: 160 },
        })
      );
      i++;
      continue;
    }

    // Sub-table heading ### Table S...
    if (line.startsWith("### Table S")) {
      const titleText = line.replace(/^### /, "");
      children.push(
        new Paragraph({
          children: parseInlineFormatting(titleText, { size: HEADING3_SIZE, bold: true }),
          alignment: AlignmentType.LEFT,
          spacing: { before: 200, after: 120 },
        })
      );
      i++;
      continue;
    }

    // Supplementary Figures heading
    if (line.startsWith("## Supplementary Figures")) {
      children.push(pageBreakParagraph());
      children.push(
        new Paragraph({
          children: [
            new TextRun({
              text: "Supplementary Figures",
              font: FONT,
              size: HEADING1_SIZE,
              bold: true,
            }),
          ],
          alignment: AlignmentType.LEFT,
          spacing: { before: 240, after: 200 },
        })
      );
      i++;
      continue;
    }

    // Table block (starts with |)
    if (line.trim().startsWith("|")) {
      const tableLines = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        tableLines.push(lines[i]);
        i++;
      }
      const parsed = parseMarkdownTable(tableLines);
      if (parsed) {
        children.push(buildTable(parsed));
        children.push(
          new Paragraph({
            children: [],
            spacing: { after: 80 },
          })
        );
      }
      continue;
    }

    // Figure caption paragraphs
    if (line.startsWith("**Figure S")) {
      // Parse figure caption: **Figure SN. Title.** Legend text...
      const figMatch = line.match(/^\*\*Figure S(\d+)\.\s*(.+?)\*\*\s*(.*)/);
      if (figMatch) {
        const figNum = parseInt(figMatch[1]);
        const captionTitle = figMatch[2].replace(/\.\s*$/, "");
        const legend = figMatch[3] || "";

        // Figure files
        const figFiles = {
          1: "figS1_setA.png",
          2: "figS2_setB.png",
          3: "figS3_setC.png",
          4: "figS4_setD.png",
          5: "figS5_setE.png",
        };

        if (figFiles[figNum]) {
          const figPath = path.join(FIG_DIR, figFiles[figNum]);
          if (fs.existsSync(figPath)) {
            // Add page break before figures (except first one which already has one after heading)
            if (figNum > 1) {
              children.push(pageBreakParagraph());
            }
            const figElements = buildFigure(figPath, captionTitle, legend, figNum);
            children.push(...figElements);
          } else {
            console.warn(`Figure not found: ${figPath}`);
          }
        }
      }
      i++;
      continue;
    }

    // Regular paragraph text (non-empty, non-header, non-table)
    if (line.trim().length > 0 && !line.startsWith("#")) {
      // Handle footnotes starting with \*
      const cleanedLine = line.replace(/^\\\*\s*/, "* ").replace(/\\\*\*/g, "**");
      children.push(
        new Paragraph({
          children: parseInlineFormatting(cleanedLine, { size: SMALL_SIZE }),
          alignment: AlignmentType.LEFT,
          spacing: { before: 60, after: 120 },
        })
      );
      i++;
      continue;
    }

    // Empty line or unhandled
    i++;
  }

  // ── Create the Document ──
  const doc = new Document({
    styles: {
      default: {
        document: {
          run: {
            font: FONT,
            size: BODY_SIZE,
          },
        },
      },
    },
    sections: [
      {
        properties: {
          page: {
            margin: {
              top: MARGIN_DXA,
              right: MARGIN_DXA,
              bottom: MARGIN_DXA,
              left: MARGIN_DXA,
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
                    font: FONT,
                    size: SMALL_SIZE,
                  }),
                ],
                alignment: AlignmentType.CENTER,
              }),
            ],
          }),
        },
        children: children,
      },
    ],
  });

  return doc;
}

// ── Main ──
async function main() {
  console.log("Building supplementary document...");

  const doc = buildDocument();

  const outPath = path.join(__dirname, "..", "supplementary.docx");
  const buffer = await Packer.toBuffer(doc);
  fs.writeFileSync(outPath, buffer);

  console.log(`Successfully created: ${outPath}`);
  console.log(`File size: ${(buffer.length / 1024).toFixed(1)} KB`);
}

main().catch((err) => {
  console.error("Error creating document:", err);
  process.exit(1);
});
