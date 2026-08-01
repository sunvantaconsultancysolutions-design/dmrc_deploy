import { useState, useEffect } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const ZOOM_MIN = 0.5;
const ZOOM_MAX = 3;
const ZOOM_STEP = 0.15;

// PHASE 2 (Evidence Viewer) rewrite.
//
// Feature 1 -- Previous/Next now walk the retrieved EVIDENCE LIST for
// the current answer (the `sources` array, already ordered by
// retrieval/reranker score -- see app.py's rerank() + _build_sources())
// instead of the PDF's own physical page sequence. Concretely: this
// component receives the full `sources` array and the current
// `activeIndex` from App.jsx (which owns that state so it can also
// drive the highlighted SourceChip -- Feature 2), and simply renders
// `sources[activeIndex]`. Moving to the next/previous piece of
// evidence is exactly "activeIndex +/- 1", not "pdf_page +/- 1".
//
// Some evidence items have no rendered page image (e.g. BOQ ADDENDUM
// rows with no source_pdf -- a known, pre-existing data limitation,
// not something this phase changes). Previous/Next skip over those so
// the viewer never lands on a blank pane; the button disables itself
// when no viewable neighbour exists in that direction, satisfying
// Feature 4 ("the PDF viewer must automatically open the correct
// page" -- it always opens either a real page or nothing, never a
// wrong/blank one).
export default function PageViewer({ sources, activeIndex, onNavigate, onClose }) {
  const [zoom, setZoom] = useState(1);
  const [fitMode, setFitMode] = useState("fit-page"); // "fit-page" | "fit-width" | "custom"

  // Reset zoom whenever a different piece of evidence is opened, so a
  // zoomed-in state from one page never carries over confusingly to
  // the next.
  useEffect(() => {
    setZoom(1);
    setFitMode("fit-page");
  }, [activeIndex, sources]);

  const hasSources = Array.isArray(sources) && sources.length > 0;
  const source = hasSources && activeIndex != null ? sources[activeIndex] : null;

  if (!hasSources || !source || !source.image_url) {
    return (
      <aside className="page-viewer page-viewer--empty">
        <p>Ask a question — the cited contract page appears here.</p>
      </aside>
    );
  }

  // Nearest neighbour (in either direction) that actually has a
  // rendered page image, so Previous/Next always land on real evidence.
  const findViewableNeighbour = (fromIndex, step) => {
    let i = fromIndex + step;
    while (i >= 0 && i < sources.length) {
      if (sources[i].image_url) return i;
      i += step;
    }
    return null;
  };
  const prevIndex = findViewableNeighbour(activeIndex, -1);
  const nextIndex = findViewableNeighbour(activeIndex, +1);

  const url = `${API_URL}${source.image_url}`;

  function zoomIn() {
    setFitMode("custom");
    setZoom((z) => Math.min(ZOOM_MAX, +(z + ZOOM_STEP).toFixed(2)));
  }
  function zoomOut() {
    setFitMode("custom");
    setZoom((z) => Math.max(ZOOM_MIN, +(z - ZOOM_STEP).toFixed(2)));
  }
  function fitWidth() {
    setFitMode("fit-width");
    setZoom(1);
  }
  function fitPage() {
    setFitMode("fit-page");
    setZoom(1);
  }

  const imgClass =
    "page-viewer__page-img" +
    (fitMode === "fit-width" ? " page-viewer__page-img--fit-width" : "") +
    (fitMode === "fit-page" ? " page-viewer__page-img--fit-page" : "") +
    (fitMode === "custom" ? " page-viewer__page-img--custom" : "");
  const imgStyle = fitMode === "custom" ? { transform: `scale(${zoom})` } : undefined;

  return (
    <aside className="page-viewer">
      <div className="page-viewer__bar">
        <div className="page-viewer__meta">
          <strong>{source.document || source.document_id}</strong>
          <span>
            {source.clause ? `Clause ${source.clause} · ` : ""}
            {source.item_number && !source.clause ? `${source.item_number} · ` : ""}
            {source.page ? `Scan p. ${source.page} · ` : ""}
            Evidence {activeIndex + 1} of {sources.length}
          </span>
        </div>
        <div className="page-viewer__nav">
          <button
            onClick={() => prevIndex != null && onNavigate(prevIndex)}
            disabled={prevIndex == null}
            aria-label="Previous evidence"
            title="Previous evidence"
          >
            ‹
          </button>
          <button
            onClick={() => nextIndex != null && onNavigate(nextIndex)}
            disabled={nextIndex == null}
            aria-label="Next evidence"
            title="Next evidence"
          >
            ›
          </button>
          <button onClick={onClose} aria-label="Close viewer" title="Close">
            ×
          </button>
        </div>
      </div>

      <div className="page-viewer__toolbar">
        <button onClick={zoomOut} aria-label="Zoom out" title="Zoom out">−</button>
        <span className="page-viewer__zoom-level">
          {fitMode === "custom" ? `${Math.round(zoom * 100)}%` : fitMode === "fit-width" ? "Fit width" : "Fit page"}
        </span>
        <button onClick={zoomIn} aria-label="Zoom in" title="Zoom in">+</button>
        <span className="page-viewer__toolbar-divider" aria-hidden="true" />
        <button
          onClick={fitWidth}
          className={fitMode === "fit-width" ? "page-viewer__toolbar-btn--active" : ""}
          title="Fit to width"
        >
          Fit width
        </button>
        <button
          onClick={fitPage}
          className={fitMode === "fit-page" ? "page-viewer__toolbar-btn--active" : ""}
          title="Fit whole page"
        >
          Fit page
        </button>
      </div>

      <div className="page-viewer__img">
        <img
          src={url}
          alt={`Scanned contract page ${source.page ?? source.pdf_page}`}
          className={imgClass}
          style={imgStyle}
          onError={(e) => { e.currentTarget.style.opacity = 0.25; }}
        />
      </div>

      {/* Existing feature (unchanged): figure_urls is populated backend-side
          (app.py's SourceItem model). Renders only when the cited page
          actually has extracted figures. */}
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