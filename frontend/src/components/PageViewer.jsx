import { useState, useEffect } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

// Evidence panel: shows the scanned page image for the selected
// source. Header shows the stamped scan number (what the user sees
// printed on the page) plus the PDF index (debug aid). Prev/next walk
// pdf_page within the same document; the corresponding stamp for
// neighbouring pages is not known client-side, so navigation shows
// only the PDF index until a source chip is clicked again.
export default function PageViewer({ source, onClose }) {
  const [pdfPage, setPdfPage] = useState(source?.pdf_page ?? null);
  useEffect(() => setPdfPage(source?.pdf_page ?? null), [source]);

  if (!source || !source.image_url) {
    return (
      <aside className="page-viewer page-viewer--empty">
        <p>Ask a question — the cited contract page appears here.</p>
      </aside>
    );
  }

  const onCited = pdfPage === source.pdf_page;
  const pad = (n) => `p${String(n).padStart(4, "0")}.jpg`;
  const url = onCited
    ? `${API_URL}${source.image_url}`
    : `${API_URL}/pages/${source.document_id}/${pad(pdfPage)}`;

  return (
    <aside className="page-viewer">
      <div className="page-viewer__bar">
        <div className="page-viewer__meta">
          <strong>{source.document || source.document_id}</strong>
          <span>
            {onCited && source.clause ? `Clause ${source.clause} · ` : ""}
            {onCited && source.page ? `Scan p. ${source.page} · ` : ""}
            PDF p. {pdfPage}
          </span>
        </div>
        <div className="page-viewer__nav">
          <button onClick={() => setPdfPage((p) => Math.max(1, p - 1))}
            aria-label="Previous page">‹</button>
          <button onClick={() => setPdfPage((p) => p + 1)}
            aria-label="Next page">›</button>
          <button onClick={onClose} aria-label="Close viewer">×</button>
        </div>
      </div>
      <div className="page-viewer__img">
        <img src={url} alt={`Scanned contract page ${source.page ?? pdfPage}`}
          onError={(e) => { e.currentTarget.style.opacity = 0.25; }} />
      </div>
    </aside>
  );
}
