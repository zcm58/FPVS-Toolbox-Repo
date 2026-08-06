/** Build the ACR BCA20 expert handoff workbook with @oai/artifact-tool.
 *
 * Usage:
 *   node build_bca20_analysis_workbook.mjs payload.json output.xlsx previews/
 *
 * Run this file from a writable directory whose adjacent node_modules contains
 * the bundled @oai/artifact-tool package.  No npm installation is required.
 */

import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [payloadPath, outputPath, previewDir] = process.argv.slice(2);
if (!payloadPath || !outputPath || !previewDir) {
  throw new Error("Expected: payload.json output.xlsx preview-directory");
}

const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
if (payload.schema_version !== 1 || !Array.isArray(payload.sheets)) {
  throw new Error("Unsupported or malformed workbook payload.");
}

function columnLetter(index) {
  let number = index + 1;
  let result = "";
  while (number > 0) {
    const remainder = (number - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    number = Math.floor((number - 1) / 26);
  }
  return result;
}

function tableName(sheetName, index) {
  const stem = sheetName.replace(/[^A-Za-z0-9]/g, "");
  const safeStem = /^[A-Za-z]/.test(stem) ? stem : `Sheet${stem}`;
  return `${safeStem}Table${index + 1}`;
}

function displayWidth(header, rows, index) {
  let width = String(header).length + 2;
  const sampleRows = rows.slice(0, 250);
  for (const row of sampleRows) {
    const value = row[index];
    const text = value === null || value === undefined ? "" : String(value);
    width = Math.max(width, Math.min(text.length + 2, 60));
  }
  if (/sha256/i.test(header)) return 22;
  if (/details|definition|note|source|reason|missing_value_rule/i.test(header)) {
    return Math.min(Math.max(width, 28), 60);
  }
  return Math.min(Math.max(width, 10), 32);
}

const workbook = Workbook.create();
for (const [sheetIndex, spec] of payload.sheets.entries()) {
  const sheet = workbook.worksheets.add(spec.name);
  sheet.showGridLines = false;
  const matrix = [spec.headers, ...spec.rows];
  const rowCount = matrix.length;
  const columnCount = spec.headers.length;
  if (rowCount < 2 || columnCount < 1) {
    throw new Error(`Sheet ${spec.name} must contain a header and at least one row.`);
  }
  const used = sheet.getRangeByIndexes(0, 0, rowCount, columnCount);
  used.values = matrix;
  const lastCell = `${columnLetter(columnCount - 1)}${rowCount}`;
  const table = sheet.tables.add(`A1:${lastCell}`, true, tableName(spec.name, sheetIndex));
  table.style = "TableStyleLight1";
  table.showFilterButton = true;
  table.showBandedRows = false;
  used.format = {
    fill: "#FFFFFF",
    font: { name: "Arial", size: 9, color: "#262626" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };

  if (rowCount > 1) {
    const data = sheet.getRangeByIndexes(1, 0, rowCount - 1, columnCount);
    data.conditionalFormats.addCustom("=MOD(ROW(),2)=0", {
      fill: "#F2F2F2",
    });
  }

  const header = sheet.getRangeByIndexes(0, 0, 1, columnCount);
  header.format = {
    fill: "#595959",
    font: { name: "Arial", size: 9, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "thin", color: "#A6A6A6" },
    rowHeight: 42,
  };

  for (let columnIndex = 0; columnIndex < columnCount; columnIndex += 1) {
    const headerName = spec.headers[columnIndex];
    const columnRange = sheet.getRangeByIndexes(0, columnIndex, rowCount, 1);
    columnRange.format.columnWidth = displayWidth(headerName, spec.rows, columnIndex);
    if ((spec.wrap_columns || []).includes(headerName)) {
      columnRange.format.wrapText = true;
    }
    if ((spec.left_align_columns || []).includes(headerName) && rowCount > 1) {
      sheet.getRangeByIndexes(1, columnIndex, rowCount - 1, 1).format.horizontalAlignment = "left";
    }
    const numberFormat = (spec.number_formats || {})[headerName];
    if (numberFormat && rowCount > 1) {
      sheet.getRangeByIndexes(1, columnIndex, rowCount - 1, 1).format.numberFormat = numberFormat;
    }
  }

  sheet.freezePanes.freezeRows(Number(spec.freeze_rows || 0));
  if (Number(spec.freeze_columns || 0) > 0) {
    sheet.freezePanes.freezeColumns(Number(spec.freeze_columns));
  }
}

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const inspection = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 12000,
  tableMaxRows: 3,
  tableMaxCols: 8,
  tableMaxCellChars: 80,
});
await fs.writeFile(
  path.join(previewDir, "workbook_inspection.ndjson"),
  `${inspection.ndjson || String(inspection)}\n`,
  "utf8",
);

for (const spec of payload.sheets) {
  const columnCount = spec.headers.length;
  const previewRows = Math.min(spec.rows.length + 1, 35);
  const previewRange = `A1:${columnLetter(columnCount - 1)}${previewRows}`;
  const preview = await workbook.render({
    sheetName: spec.name,
    range: previewRange,
    autoCrop: "all",
    scale: 1,
    format: "png",
  });
  const safeName = spec.name.replace(/[^A-Za-z0-9_-]/g, "_");
  await fs.writeFile(
    path.join(previewDir, `${String(payload.sheets.indexOf(spec) + 1).padStart(2, "0")}_${safeName}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

process.stdout.write(
  `${JSON.stringify({
    ok: true,
    output: path.resolve(outputPath),
    sheets: payload.sheets.map((sheet) => sheet.name),
    previews: path.resolve(previewDir),
  })}\n`,
);
