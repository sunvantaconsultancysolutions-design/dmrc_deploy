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
  const onPrev  = pdfPage === source.pdf_page - 1;
  const onNext  = pdfPage === source.pdf_page + 1;
  const pad = (n) => `p${String(n).padStart(4, "0")}.jpg`;

  // ISSUE 4 FIX: use API-supplied prev_image_url / next_image_url when
  // available, falling back to the manually-constructed path for any other
  // page (or when the API does not return those fields).
  //
  // Priority:
  //   cited page  -> source.image_url          (unchanged)
  //   prev page   -> source.prev_image_url      if truthy, else manual
  //   next page   -> source.next_image_url      if truthy, else manual
  //   other page  -> manual /pages/{doc}/{pad}  (unchanged)
  //
  // Backward compatibility: older server responses without prev/next_image_url
  // return undefined/null for those fields; the falsy check falls through to
  // the manual construction that already worked, so no regression is possible.
  const url = onCited
    ? `${API_URL}${source.image_url}`
    : onPrev && source.prev_image_url
      ? `${API_URL}${source.prev_image_url}`
      : onNext && source.next_image_url
        ? `${API_URL}${source.next_image_url}`
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
          <button
            onClick={() => setPdfPage((p) => {
              const maxPage = source.max_pdf_page;
              return maxPage ? Math.min(maxPage, p + 1) : p + 1;
            })}
            disabled={!!(source.max_pdf_page && pdfPage >= source.max_pdf_page)}
            style={source.max_pdf_page && pdfPage >= source.max_pdf_page
              ? {opacity: 0.3, cursor: 'not-allowed'} : {}}
            aria-label="Next page">›</button>
          <button onClick={onClose} aria-label="Close viewer">×</button>
        </div>
      </div>
      <div className="page-viewer__img">
        <img src={url} alt={`Scanned contract page ${source.page ?? pdfPage}`}
          onError={(e) => { e.currentTarget.style.opacity = 0.25; }} />
      </div>
      {/* FEATURE: figure_urls is populated backend-side (app.py's
          SourceItem model) but previously had zero frontend consumer.
          Renders only when the cited page actually has extracted
          figures -- currently empty for this corpus, so this stays
          invisible until scripts/extract_page_figures.py output ships
          with a build (see Dockerfile figure_images/ COPY). */}
      {Array.isArray(source.figure_urls) && source.figure_urls.length > 0 && (
        <div className="page-viewer__figures">
          <span className="page-viewer__figures-label">
            Figures on this page ({source.figure_urls.length})
          </span>
          <div className="page-viewer__figures-grid">
            {source.figure_urls.map((figUrl, i) => (
              <img
                key={figUrl ?? i}
                src={`${API_URL}${figUrl}`}
                alt={`Figure ${i + 1} from ${source.document || source.document_id}`}
                className="page-viewer__figure-thumb"
                onError={(e) => { e.currentTarget.style.display = "none"; }}
              />
            ))}
          </div>
        </div>
      )}
    </aside>
  );
}