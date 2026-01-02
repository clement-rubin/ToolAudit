import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from fpdf import FPDF

from services.storage import ensure_storage

AUDIT_LOG_PATH = Path("storage/audit_journal.jsonl")


def log_audit_event(step: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Ajoute une entree au journal d audit local (JSONL) avec horodatage UTC.
    """
    ensure_storage()
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "step": step,
        "details": details or {},
    }
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def read_audit_events() -> list[dict[str, Any]]:
    ensure_storage()
    if not AUDIT_LOG_PATH.exists():
        return []
    events: list[dict[str, Any]] = []
    with AUDIT_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def reset_audit_log() -> None:
    """
    Efface le journal d audit (utilise lors d une nouvelle importation de fichier).
    """
    ensure_storage()
    if AUDIT_LOG_PATH.exists():
        try:
            AUDIT_LOG_PATH.unlink()
        except OSError:
            AUDIT_LOG_PATH.write_text("", encoding="utf-8")


def _summarize(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    files: list[str] = []
    columns: set[str] = set()
    rules: set[str] = set()
    exports: list[str] = []

    for ev in events:
        det = ev.get("details", {}) or {}
        if not isinstance(det, dict):
            continue

        file_name = det.get("file") or det.get("source") or det.get("fichier")
        if file_name:
            files.append(str(file_name))

        for key in ("columns", "used_columns", "derived_columns", "model_features"):
            val = det.get(key)
            if isinstance(val, (list, tuple)):
                columns.update(str(v) for v in val)

        for key in ("rules", "controls"):
            val = det.get(key)
            if isinstance(val, (list, tuple)):
                rules.update(str(v) for v in val)

        export = det.get("export")
        if export:
            exports.append(str(export))

    return {
        "files": files,
        "columns": sorted(columns),
        "rules": sorted(rules),
        "exports": exports,
    }


def _format_details(details: Any) -> str:
    if details is None:
        return ""
    if isinstance(details, dict):
        try:
            return json.dumps(details, ensure_ascii=False, indent=2)
        except TypeError:
            return str(details)
    return str(details)


def build_html_report(
    events: list[dict[str, Any]],
    control_rules: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
) -> str:
    context = context or {}
    summary = _summarize(events)
    generated_at = datetime.utcnow().isoformat()

    actions_rows = "\n".join(
        f"<tr><td>{ev.get('timestamp','')}</td><td>{ev.get('step','')}</td>"
        f"<td><pre>{_format_details(ev.get('details', {}))}</pre></td></tr>"
        for ev in events
    )
    controls_rows = "\n".join(
        f"<li><strong>{r.get('id')}</strong> - {r.get('label')}: {r.get('criterion')}</li>"
        for r in control_rules
    )

    columns_html = ", ".join(summary["columns"]) if summary["columns"] else "Non renseigne"
    files_html = ", ".join(summary["files"]) if summary["files"] else "Non renseigne"
    exports_html = ", ".join(summary["exports"]) if summary["exports"] else "Aucun"

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <title>Rapport de traitement</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f7f7fa; color: #0f172a; }}
    h1 {{ font-size: 24px; }}
    h2 {{ margin-top: 28px; }}
    .card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; margin-top: 12px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #e2e8f0; padding: 8px; font-size: 12px; vertical-align: top; }}
    th {{ background: #e0e7ff; text-align: left; }}
    pre {{ white-space: pre-wrap; word-break: break-word; font-size: 11px; }}
  </style>
</head>
<body>
  <h1>Rapport de traitement et de controle</h1>
  <div class="card">
    <p><strong>Genere le :</strong> {generated_at} (UTC)</p>
    <p><strong>Source :</strong> {context.get('source','Non renseigne')}</p>
    <p><strong>Colonnes exploitees :</strong> {columns_html}</p>
    <p><strong>Fichiers sources :</strong> {files_html}</p>
    <p><strong>Exports produits :</strong> {exports_html}</p>
  </div>

  <h2>Journal des actions ({len(events)})</h2>
  <div class="card">
    <table>
      <thead>
        <tr><th>Horodatage</th><th>Etape</th><th>Details</th></tr>
      </thead>
      <tbody>
        {actions_rows}
      </tbody>
    </table>
  </div>

  <h2>Regles et controles appliques</h2>
  <div class="card">
    <ul>
      {controls_rows}
    </ul>
  </div>
</body>
</html>
"""


def _safe_pdf_text(text: str) -> str:
    return text.encode("latin-1", "replace").decode("latin-1")


def _compact_details(details: Any) -> str:
    if not isinstance(details, dict):
        return str(details)
    parts: list[str] = []
    for key, val in details.items():
        if isinstance(val, (list, tuple)):
            parts.append(f"{key}={len(val)} item(s)")
        elif isinstance(val, dict):
            parts.append(f"{key}={len(val)} champ(s)")
        else:
            parts.append(f"{key}={val}")
    return "; ".join(parts)


def _render_list(pdf: FPDF, label: str, items: list[str], empty_label: str) -> None:
    pdf.set_font("Arial", "B", 10)
    pdf.multi_cell(0, 6, _safe_pdf_text(label))
    pdf.set_font("Arial", "", 10)
    if not items:
        pdf.multi_cell(0, 6, _safe_pdf_text(f"- {empty_label}"))
    else:
        for item in items:
            txt = str(item)
            # Coupe les valeurs tres longues pour eviter les erreurs de largeur
            chunk_size = 80
            chunks = [txt[i : i + chunk_size] for i in range(0, len(txt), chunk_size)] or [txt]
            pdf.multi_cell(0, 6, _safe_pdf_text(f"- {chunks[0]}"))
            for ch in chunks[1:]:
                pdf.multi_cell(0, 6, _safe_pdf_text(f"  {ch}"))
    pdf.ln(1)


def build_pdf_report(
    events: list[dict[str, Any]],
    control_rules: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
) -> bytes:
    context = context or {}
    summary = _summarize(events)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, _safe_pdf_text("Rapport de traitement et controles"), ln=1)

    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, _safe_pdf_text(f"Source: {context.get('source', 'Non renseigne')}"))
    _render_list(pdf, "Colonnes exploitees :", summary["columns"], "Non renseigne")
    _render_list(pdf, "Fichiers sources :", summary["files"], "Non renseigne")
    _render_list(pdf, "Exports produits :", summary["exports"], "Aucun")
    pdf.ln(4)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, _safe_pdf_text("Journal des actions"), ln=1)
    pdf.set_font("Arial", "", 10)
    for ev in events:
        line = f"{ev.get('timestamp','')} | {ev.get('step','')} | {_compact_details(ev.get('details', {}))}"
        pdf.multi_cell(0, 6, _safe_pdf_text(line))
    pdf.ln(2)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, _safe_pdf_text("Regles et controles appliques"), ln=1)
    pdf.set_font("Arial", "", 10)
    for r in control_rules:
        pdf.multi_cell(
            0,
            6,
            _safe_pdf_text(f"{r.get('id')}: {r.get('criterion')}"),
        )

    return pdf.output(dest="S").encode("latin-1", "replace")


def generate_audit_report(
    events: list[dict[str, Any]],
    control_rules: list[dict[str, Any]],
    base_name: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Cree un rapport HTML et PDF dans storage/ et renvoie les buffers utiles
    pour un telechargement streamlit.
    """
    ensure_storage()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_base = base_name or f"audit_report_{ts}"

    html_content = build_html_report(events, control_rules, context=context)
    html_path = Path("storage") / f"{safe_base}.html"
    html_path.write_text(html_content, encoding="utf-8")

    pdf_bytes = build_pdf_report(events, control_rules, context=context)
    pdf_path = Path("storage") / f"{safe_base}.pdf"
    pdf_path.write_bytes(pdf_bytes)

    return {
        "html": html_content,
        "pdf_bytes": pdf_bytes,
        "html_path": html_path,
        "pdf_path": pdf_path,
    }
