// Markdown-subset -> DOCX builder for the final report.
// Usage: node build.js report.md appendix.md out.docx
const fs = require("fs");
const path = require("path");
const D = require("docx");
const { Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType,
  ImageRun, AlignmentType, BorderStyle, ShadingType, Footer, Header, PageNumber, LevelFormat,
  PageBreak, TableOfContents, ExternalHyperlink, SectionType, NumberFormat, TabStopType } = D;

const [mdMain, mdApp, out] = process.argv.slice(2);
const FONT = "Calibri";
const BODY = 24;          // half-points: 12 pt
const SMALL = 18;         // 9 pt for tables
const CAP = 20;           // 10 pt captions
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1247;   // A4, 2.2 cm margins
const TEXT_W = PAGE_W - 2 * MARGIN;                    // DXA
const BLUE = "1F3864";

function pngSize(buf) { return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) }; }

// inline: **bold**, *italic*, `code`, [text](url)
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g;
  let last = 0, m;
  const push = (t, o = {}) => { if (t) out.push(new TextRun({ text: t, font: FONT, ...base, ...o })); };
  while ((m = re.exec(text))) {
    push(text.slice(last, m.index));
    const s = m[0];
    if (s.startsWith("**")) push(s.slice(2, -2), { bold: true });
    else if (s.startsWith("`")) push(s.slice(1, -1), { font: "Consolas", size: base.size ? base.size - 2 : BODY - 4 });
    else if (s.startsWith("[")) {
      const mm = s.match(/\[([^\]]+)\]\(([^)]+)\)/);
      out.push(new ExternalHyperlink({ link: mm[2], children: [new TextRun({ text: mm[1], font: FONT, ...base, color: "1155CC", underline: {} })] }));
    } else push(s.slice(1, -1), { italics: true });
    last = m.index + s.length;
  }
  push(text.slice(last));
  return out;
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "A6A6A6" };
const borders = { top: border, bottom: border, left: border, right: border };

function table(rows, widthsSpec, size) {
  const head = rows[0];
  const n = head.length;
  rows = rows.map(r => r.length === n ? r : r.length > n ? [...r.slice(0, n - 1), r.slice(n - 1).join(" | ")] : [...r, ...Array(n - r.length).fill("")]);
  let w = widthsSpec && widthsSpec.length === n ? widthsSpec : null;
  if (!w) {   // width proportional to sqrt of max content length
    const len = head.map((_, j) => Math.max(...rows.map(r => (r[j] || "").length)));
    const s = len.map(l => Math.sqrt(Math.max(l, 3)));
    const tot = s.reduce((a, b) => a + b, 0);
    w = s.map(x => x / tot);
  }
  const cols = w.map(x => Math.floor(x * TEXT_W));
  cols[cols.length - 1] += TEXT_W - cols.reduce((a, b) => a + b, 0);
  const mk = (r, hdr) => new TableRow({
    tableHeader: hdr, cantSplit: true,
    children: r.map((c, j) => new TableCell({
      borders, width: { size: cols[j], type: WidthType.DXA },
      shading: hdr ? { fill: "D9E2F3", type: ShadingType.CLEAR, color: "auto" } : undefined,
      margins: { top: 30, bottom: 30, left: 70, right: 70 },
      children: [new Paragraph({ spacing: { before: 0, after: 0 }, children: runs(c, { size, bold: hdr || undefined }) })],
    })),
  });
  return new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: cols,
    rows: [mk(head, true), ...rows.slice(1).map(r => mk(r, false))] });
}

function parse(md, baseDir, opts = {}) {
  const lines = md.replace(/\r/g, "").split("\n");
  const kids = [];
  let i = 0, para = [], numInst = 0, prevNum = false;
  const size = opts.size || BODY;
  const flush = () => {
    if (para.length) {
      kids.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 100 }, children: runs(para.join(" "), { size }) }));
      para = [];
    }
  };
  while (i < lines.length) {
    const L = lines[i];
    if (/^\s*$/.test(L)) { flush(); i++; continue; }
    if (L.startsWith("<!--")) { while (i < lines.length && !lines[i].includes("-->")) i++; i++; continue; }
    if (L.trim().startsWith("```")) {   // fenced code block -> small monospace lines
      flush(); i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        kids.push(new Paragraph({ spacing: { before: 0, after: 0 }, indent: { left: 284 }, children: [new TextRun({ text: lines[i], font: "Consolas", size: 15 })] }));
        i++;
      }
      kids.push(new Paragraph({ spacing: { after: 80 }, children: [] })); i++; continue;
    }
    if (L.trim() === "\\pagebreak") { flush(); kids.push(new Paragraph({ children: [new PageBreak()] })); i++; continue; }
    let m;
    if ((m = L.match(/^(#{1,4}) (.*)$/))) {
      flush();
      const lvl = [HeadingLevel.HEADING_1, HeadingLevel.HEADING_2, HeadingLevel.HEADING_3, HeadingLevel.HEADING_4][m[1].length - 1];
      kids.push(new Paragraph({ heading: lvl, keepNext: true, children: runs(m[2]) }));
      i++; continue;
    }
    if ((m = L.match(/^!\[(.*)\]\((.*?)\)(?:\{(\d+)\})?$/))) {   // ![caption](path){percent}
      flush();
      const paths = m[2].split("|");
      const pct = (m[3] ? +m[3] : 100) / 100;
      const maxW = TEXT_W / 1440 * 96 * pct;   // px at 96 dpi
      const imgs = paths.map(p => {
        const buf = fs.readFileSync(path.resolve(baseDir, p.trim()));
        const s = pngSize(buf), wpx = maxW / paths.length - (paths.length > 1 ? 8 : 0);
        return new ImageRun({ type: "png", data: buf, transformation: { width: Math.round(wpx), height: Math.round(wpx * s.h / s.w) },
          altText: { title: m[1].slice(0, 60), description: m[1], name: path.basename(p) } });
      });
      kids.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 80, after: 40 }, children: imgs }));
      kids.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: runs(m[1], { size: CAP, italics: false }) }));
      i++; continue;
    }
    if (L.startsWith("::: ")) {   // table caption (placed above table)
      flush();
      kids.push(new Paragraph({ keepNext: true, spacing: { before: 120, after: 60 }, children: runs(L.slice(4), { size: CAP }) }));
      i++; continue;
    }
    if (L.startsWith("|")) {
      flush();
      const rows = [];
      let widths = null;
      while (i < lines.length && lines[i].startsWith("|")) {
        const cells = lines[i].trim().replace(/^\|/, "").replace(/\|$/, "").split(/(?<!\\)\|/).map(c => c.trim().replace(/\\\|/g, "|"));
        if (cells.every(c => /^:?-{2,}:?$/.test(c))) { i++; continue; }
        rows.push(cells);
        i++;
      }
      if (i < lines.length && (m = lines[i].match(/^\{widths: ([\d., ]+)\}$/))) { widths = m[1].split(",").map(Number); i++; }
      kids.push(table(rows, widths, opts.tableSize || SMALL));
      kids.push(new Paragraph({ spacing: { after: 80 }, children: [] }));
      continue;
    }
    if ((m = L.match(/^(\s*)[-*] (.*)$/))) {
      flush();
      const lvl = Math.min(Math.floor(m[1].length / 2), 1);
      kids.push(new Paragraph({ numbering: { reference: "bul", level: lvl }, alignment: AlignmentType.LEFT, spacing: { after: 40 }, children: runs(m[2], { size }) }));
      i++; continue;
    }
    if ((m = L.match(/^(\d+)\. (.*)$/))) {
      flush();
      if (m[1] === "1") numInst++;
      kids.push(new Paragraph({ numbering: { reference: "num", level: 0, instance: numInst }, spacing: { after: 40 }, children: runs(m[2], { size }) }));
      i++; continue;
    }
    if (L.startsWith("> ")) {   // reference list entry (hanging indent)
      flush();
      kids.push(new Paragraph({ indent: { left: 567, hanging: 567 }, spacing: { after: 80 }, children: runs(L.slice(2), { size }) }));
      i++; continue;
    }
    para.push(L.trim());
    i++;
  }
  flush();
  return kids;
}

const cfg = JSON.parse(fs.readFileSync(path.join(path.dirname(mdMain), "cover.json"), "utf8"));
const cover = [
  new Paragraph({ spacing: { before: 1200 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: cfg.milestone, font: FONT, size: 32, bold: true, color: "404040" })] }),
  new Paragraph({ spacing: { before: 400, after: 200 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: cfg.title, font: FONT, size: 44, bold: true, color: BLUE })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 500 }, children: [new TextRun({ text: cfg.subtitle, font: FONT, size: 26, color: "404040" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "National University of Singapore", font: FONT, size: 26, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: cfg.course, font: FONT, size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 500 }, children: [new TextRun({ text: cfg.semester, font: FONT, size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: cfg.team, font: FONT, size: 26, bold: true })] }),
  table([["Name", "Student number"], ...cfg.members], [0.5, 0.5], BODY),
  new Paragraph({ spacing: { before: 600 }, alignment: AlignmentType.CENTER, children: runs(cfg.links[0], { size: 22 }) }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: runs(cfg.links[1], { size: 22 }) }),
  new Paragraph({ spacing: { before: 500 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: cfg.date, font: FONT, size: 24, color: "595959" })] }),
];
const toc = [
  new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: "Table of Contents", font: FONT, size: 30, bold: true, color: BLUE })] }),
  new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
];

const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })] })] });
const header = new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: cfg.header, font: FONT, size: 18, color: "7F7F7F" })] })] });
const page = { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN, header: 600, footer: 600 } };

const doc = new Document({
  creator: cfg.team, title: cfg.title,
  styles: {
    default: { document: { run: { font: FONT, size: BODY } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: FONT, color: BLUE }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: FONT, color: BLUE }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, italics: true, font: FONT, color: "2F5496" }, paragraph: { spacing: { before: 140, after: 60 }, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [
      { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 280 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 900, hanging: 280 } } } } ] },
    { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 320 } } } }] },
  ] },
  sections: [
    { properties: { page }, children: cover },
    { properties: { page: { ...page, pageNumbers: { start: 1, formatType: NumberFormat.LOWER_ROMAN } } }, footers: { default: footer }, children: toc },
    { properties: { page: { ...page, pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } } }, headers: { default: header }, footers: { default: footer },
      children: parse(fs.readFileSync(mdMain, "utf8"), path.dirname(mdMain)) },
    { properties: { page, type: SectionType.NEXT_PAGE }, headers: { default: header }, footers: { default: footer },
      children: parse(fs.readFileSync(mdApp, "utf8"), path.dirname(mdApp), { size: 22, tableSize: 16 }) },
  ],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync(out, b); console.log("wrote", out); });
