# -*- coding: utf-8 -*-
# Statistica 13.5 full metadata extractor via COM
# ------------------------------------------------
# - Scans a folder recursively for .stw
# - Opens each via COM (Statistica must be installed/licensed on this machine)
# - Extracts nodes, connections, settings XML, flattened params,
#   formulas, macros, SQL/regex/filepaths/connection strings
# - Writes MASTER CSVs + one consolidated Excel workbook
#
# Usage:
#   pip install pywin32 pandas openpyxl
#   python statistica_full_extract.py --root "D:\Workflows" --out "D:\Extracted" --excel "D:\Extracted\statistica_inventory.xlsx"

import os
import re
import csv
import sys
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import win32com.client as win32

# -----------------------------
# Heuristics & regex patterns
# -----------------------------
SQL_PATTERNS = [
    r"\bWITH\b[\s\S]*?\bSELECT\b[\s\S]+",    # CTEs
    r"\bSELECT\b[\s\S]+?\bFROM\b[\s\S]+",    # SELECT
    r"\bINSERT\b[\s\S]+?\bINTO\b[\s\S]+",    # INSERT
    r"\bUPDATE\b[\s\S]+?\bSET\b[\s\S]+",     # UPDATE
    r"\bDELETE\b[\s\S]+?\bFROM\b[\s\S]+",    # DELETE
]
SQL_RE = re.compile("(" + "|".join(SQL_PATTERNS) + ")", re.IGNORECASE)

REGEX_NAME_HINT = re.compile(r"(regex|reg[_ ]?exp|pattern|mask)", re.IGNORECASE)
REGEX_META = set(".^$*+?{}[]|()\\")
CONN_STR_HINT = re.compile(r"(Driver|Server|Host|UID|User\s*ID|PWD|Password|Database|DSN)=", re.IGNORECASE)
FILEPATH_HINT = re.compile(r"([A-Za-z]:\\[^:*?\"<>|\r\n]+|\b/[^ \t\r\n]+)", re.IGNORECASE)

# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sanitize(name: str, limit=150) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", str(name))[:limit].strip() or "unnamed"

def safe_get(obj, *names, default=None):
    for n in names:
        try:
            v = getattr(obj, n)
            _ = v  # force dispatch
            return v
        except Exception:
            continue
    return default

def flatten_xml(xml_text: str):
    """Flatten XML into [{'ParamPath': path, 'ParamValue': value}, ...]."""
    if not xml_text:
        return []
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(str(xml_text))
    except Exception:
        return []
    rows = []
    def walk(elem, path):
        tag = elem.tag.split('}')[-1]
        here = f"{path}/{tag}" if path else tag
        text = (elem.text or "").strip()
        if text:
            rows.append({"ParamPath": here, "ParamValue": text})
        for k, v in elem.attrib.items():
            rows.append({"ParamPath": f"{here}[@{k}]", "ParamValue": v})
        for ch in list(elem):
            walk(ch, here)
    walk(root, "")
    return rows

def detect_sql(text: str):
    if not text: return []
    return [m.group(0).strip() for m in SQL_RE.finditer(text)]

def likely_regex(value: str, field_name: str = ""):
    if not value: return False
    if field_name and REGEX_NAME_HINT.search(field_name):
        return True
    meta = sum(1 for ch in value if ch in REGEX_META)
    return meta >= 3 and len(value) <= 2000

def detect_filepaths(text: str):
    if not text: return []
    return [m.group(0).strip() for m in FILEPATH_HINT.finditer(text)]

def detect_connstrings(text: str):
    if not text: return []
    if CONN_STR_HINT.search(text):
        return [text.strip()]
    return []

def write_csv(path: Path, rows: list):
    if not rows: return
    ensure_dir(path.parent)
    cols = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

# -----------------------------
# Core extraction
# -----------------------------
def extract_workflow(app, wf_path: Path, out_root: Path):
    wf_name = wf_path.name
    print(f"  - Opening: {wf_name}")
    doc = app.Documents.Open(str(wf_path))

    # Node collections (object model differences across builds)
    workspace = safe_get(doc, "AsWorkspace")
    nodes_col = None
    if workspace is not None:
        nodes_col = safe_get(workspace, "Nodes")
    nodes_col = nodes_col or safe_get(doc, "Nodes") or safe_get(doc, "AnalysisItems") or safe_get(doc, "AnalysisNodes")

    # Create per-workflow folder
    wf_folder = out_root / sanitize(wf_path.stem)
    ensure_dir(wf_folder)

    nodes_rows, links_rows = [], []
    params_rows, sql_rows, regex_rows = [], [], []
    files_rows, conn_rows = [], []

    node_count = 0
    link_count = 0
    macros_saved = 0
    formulas_saved = 0

    if nodes_col is not None:
        # Try iterate, fall back to 1-based index
        try:
            _ = [n for n in nodes_col]  # test iter
            iterator = nodes_col
        except Exception:
            iterator = range(1, nodes_col.Count + 1)

        for it in iterator:
            node = it if not isinstance(it, int) else nodes_col(it)
            node_count += 1

            node_id   = safe_get(node, "ID", "Id", "Index", default=node_count)
            node_name = safe_get(node, "Name", "LongName", default=f"Node{node_count}")
            node_type = safe_get(node, "Type", "TypeName", "Procedure", default="")
            node_proc = safe_get(node, "Procedure", default="")
            node_desc = safe_get(node, "Description", default="")

            # Settings XML (often contains everything)
            settings_xml = safe_get(node, "SettingsXML", "SettingsXml", "XmlSettings", default="")
            xml_path = ""
            if settings_xml:
                xml_path = str(wf_folder / f"node_{node_id}_settings.xml")
                try:
                    with open(xml_path, "w", encoding="utf-8") as xf:
                        xf.write(str(settings_xml))
                except Exception:
                    xml_path = ""

            # Formula / expression (if present)
            formula_text = safe_get(node, "Formula", "Expression", default="")
            formula_path = ""
            if formula_text:
                formula_path = str(wf_folder / f"node_{node_id}_formula.txt")
                try:
                    with open(formula_path, "w", encoding="utf-8") as ff:
                        ff.write(str(formula_text))
                    formulas_saved += 1
                except Exception:
                    formula_path = ""

            # Macro (if attached)
            macro_path = ""
            try:
                macro_obj = safe_get(node, "Macro")
                if macro_obj:
                    mname = f"node_{node_id}_{sanitize(str(node_name))}.svb"
                    macro_path = str(wf_folder / mname)
                    macro_obj.SaveAs(macro_path)
                    macros_saved += 1
            except Exception:
                macro_path = ""

            # Node record
            nodes_rows.append({
                "Workflow": wf_name,
                "WorkflowPath": str(wf_path),
                "NodeID": node_id,
                "NodeName": str(node_name),
                "NodeType": str(node_type),
                "Procedure": str(node_proc),
                "Description": str(node_desc),
                "SettingsXMLPath": xml_path,
                "FormulaPath": formula_path,
                "MacroPath": macro_path
            })

            # Connections: try OutputLinks/Outputs and Document-level Links if needed
            out_links = safe_get(node, "OutputLinks", "Outputs")
            if out_links is not None:
                try:
                    _ = [l for l in out_links]
                    link_iter = out_links
                except Exception:
                    link_iter = range(1, out_links.Count + 1)
                for l in link_iter:
                    link = l if not isinstance(l, int) else out_links(l)
                    tgt = safe_get(link, "TargetNode", "Target", default=None)
                    src = safe_get(link, "SourceNode", "Source", default=node)
                    if tgt is None: continue
                    links_rows.append({
                        "Workflow": wf_name,
                        "FromNodeID": safe_get(src, "ID", "Id", "Index", default=""),
                        "FromNodeName": str(safe_get(src, "Name", "LongName", default="")),
                        "ToNodeID": safe_get(tgt, "ID", "Id", "Index", default=""),
                        "ToNodeName": str(safe_get(tgt, "Name", "LongName", default="")),
                    })
                    link_count += 1

            # Flatten params from XML
            flat_params = flatten_xml(settings_xml)
            for pr in flat_params:
                pr.update({
                    "Workflow": wf_name,
                    "NodeID": node_id,
                    "NodeName": str(node_name)
                })
                params_rows.append(pr)

                # Heuristics over each param value
                ppath, pval = pr["ParamPath"], str(pr["ParamValue"])

                # SQL
                for sql in detect_sql(pval):
                    sql_rows.append({
                        "Workflow": wf_name,
                        "NodeID": node_id,
                        "NodeName": str(node_name),
                        "ParamPath": ppath,
                        "SQL": sql
                    })

                # Regex
                if likely_regex(pval, field_name=ppath):
                    regex_rows.append({
                        "Workflow": wf_name,
                        "NodeID": node_id,
                        "NodeName": str(node_name),
                        "ParamPath": ppath,
                        "Regex": pval
                    })

                # File paths
                fps = detect_filepaths(pval)
                for fp in fps:
                    files_rows.append({
                        "Workflow": wf_name,
                        "NodeID": node_id,
                        "NodeName": str(node_name),
                        "ParamPath": ppath,
                        "FilePath": fp
                    })

                # Connection strings
                conns = detect_connstrings(pval)
                for cs in conns:
                    conn_rows.append({
                        "Workflow": wf_name,
                        "NodeID": node_id,
                        "NodeName": str(node_name),
                        "ParamPath": ppath,
                        "ConnectionString": cs
                    })

    # Fallback: some builds expose document-level Links/Connections
    doc_links = safe_get(doc, "Links", "Connections")
    if doc_links is not None and not links_rows:
        try:
            _ = [dl for dl in doc_links]
            d_iter = doc_links
        except Exception:
            d_iter = range(1, doc_links.Count + 1)
        for it in d_iter:
            lk = it if not isinstance(it, int) else doc_links(it)
            src = safe_get(lk, "SourceNode", "Source", default=None)
            tgt = safe_get(lk, "TargetNode", "Target", default=None)
            if src and tgt:
                links_rows.append({
                    "Workflow": wf_name,
                    "FromNodeID": safe_get(src, "ID", "Id", "Index", default=""),
                    "FromNodeName": str(safe_get(src, "Name", "LongName", default="")),
                    "ToNodeID": safe_get(tgt, "ID", "Id", "Index", default=""),
                    "ToNodeName": str(safe_get(tgt, "Name", "LongName", default="")),
                })
                link_count += 1

    # Save per-workflow CSVs
    write_csv(wf_folder / "nodes.csv", nodes_rows)
    write_csv(wf_folder / "connections.csv", links_rows)
    write_csv(wf_folder / "params.csv", params_rows)
    write_csv(wf_folder / "sql.csv", sql_rows)
    write_csv(wf_folder / "regex.csv", regex_rows)
    write_csv(wf_folder / "files.csv", files_rows)
    write_csv(wf_folder / "connections_strings.csv", conn_rows)

    # Summary / complexity
    complexity = node_count + link_count + (2 * macros_saved)
    summary = {
        "Workflow": wf_name,
        "WorkflowPath": str(wf_path),
        "Nodes": node_count,
        "Links": link_count,
        "MacrosSaved": macros_saved,
        "FormulasSaved": formulas_saved,
        "ComplexityScore": complexity
    }

    # Close doc
    try:
        doc.Close(False)
    except Exception:
        pass

    return summary, nodes_rows, links_rows, params_rows, sql_rows, regex_rows, files_rows, conn_rows


def main():
    parser = argparse.ArgumentParser(description="Statistica 13.5 full metadata extractor via COM")
    parser.add_argument("--root", required=True, help="Root folder to recursively scan for .stw")
    parser.add_argument("--out", required=True, help="Output folder for CSVs and per-workflow dumps")
    parser.add_argument("--excel", default="", help="Optional consolidated Excel workbook path")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_root = Path(args.out).resolve()
    ensure_dir(out_root)

    # Gather .stw
    wf_paths = [p for p in root.rglob("*.stw")]
    if not wf_paths:
        print("No .stw found under:", root)
        return 1

    start = datetime.now()
    print(f"Found {len(wf_paths)} workflows. Launching Statistica...")

    # Start Statistica
    app = win32.Dispatch("Statistica.Application")
    app.Visible = False

    all_summary, all_nodes, all_links = [], [], []
    all_params, all_sql, all_regex = [], [], []
    all_files, all_connstr = [], []

    for wf in wf_paths:
        try:
            (summary, nodes, links, params, sqls, regexs, files, connstr) = extract_workflow(app, wf, out_root)
            all_summary.append(summary)
            all_nodes.extend(nodes)
            all_links.extend(links)
            all_params.extend(params)
            all_sql.extend(sqls)
            all_regex.extend(regexs)
            all_files.extend(files)
            all_connstr.extend(connstr)
        except Exception as e:
            print(f"!! Error extracting {wf.name}: {e}")

    # Quit Statistica
    try:
        app.Quit()
    except Exception:
        pass

    # Write MASTER CSVs
    write_csv(out_root / "MASTER_workflows.csv", all_summary)
    write_csv(out_root / "MASTER_nodes.csv", all_nodes)
    write_csv(out_root / "MASTER_connections.csv", all_links)
    write_csv(out_root / "MASTER_params.csv", all_params)
    write_csv(out_root / "MASTER_sql.csv", all_sql)
    write_csv(out_root / "MASTER_regex.csv", all_regex)
    write_csv(out_root / "MASTER_files.csv", all_files)
    write_csv(out_root / "MASTER_connection_strings.csv", all_connstr)

    # Optional Excel
    if args.excel:
        try:
            with pd.ExcelWriter(args.excel, engine="openpyxl") as xw:
                if all_summary: pd.DataFrame(all_summary).to_excel(xw, index=False, sheet_name="Workflows")
                if all_nodes:   pd.DataFrame(all_nodes).to_excel(xw, index=False, sheet_name="Nodes")
                if all_links:   pd.DataFrame(all_links).to_excel(xw, index=False, sheet_name="Connections")
                if all_params:  pd.DataFrame(all_params).to_excel(xw, index=False, sheet_name="NodeParams")
                if all_sql:     pd.DataFrame(all_sql).to_excel(xw, index=False, sheet_name="SQL")
                if all_regex:   pd.DataFrame(all_regex).to_excel(xw, index=False, sheet_name="Regex")
                if all_files:   pd.DataFrame(all_files).to_excel(xw, index=False, sheet_name="Files")
                if all_connstr: pd.DataFrame(all_connstr).to_excel(xw, index=False, sheet_name="ConnStrings")
        except Exception as e:
            print("WARN: Failed to write Excel:", e)

    dur = datetime.now() - start
    print("\n=== Extraction complete ===")
    print(f"Workflows:   {len(all_summary)}")
    print(f"Nodes:       {len(all_nodes)}")
    print(f"Connections: {len(all_links)}")
    print(f"Params:      {len(all_params)}")
    print(f"SQL found:   {len(all_sql)}")
    print(f"Regex found: {len(all_regex)}")
    print(f"Files found: {len(all_files)}")
    print(f"ConnStrings: {len(all_connstr)}")
    print(f"Output dir:  {out_root}")
    if args.excel:
        print(f"Excel:       {args.excel}")
    print(f"Duration:    {dur}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
