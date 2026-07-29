// Talks to the FastAPI backend defined in src/app.py. Endpoints and field
// names below are copied 1:1 from that file's Pydantic models -- keep them
// in sync if app.py's contract changes.
//
//   POST /ask     { query } -> { answer, sources: SourceItem[], confidence }
//   GET  /status  -> { status, embedding_model, reranker_model,
//                      dense_model_loaded, reranker_model_loaded,
//                      chromadb_connected }
//
// SourceItem: { clause, page, pdf_page, document, document_id, image_url,
//               item_number, retrieval_source, reranker_score, chunk_id,
//               chunk_type } -- every field is optional, since not all
// chunk types (contract clause vs. BOQ row) carry all of them.
//
// PDF Evidence Viewer (SVS-DMRC-2026-03): `page` is now the number
// STAMPED on the scanned page (a string, e.g. "9"), not the PDF's own
// page index -- that index lives separately in `pdf_page` (an int),
// which is also the file-lookup key backing `image_url`
// (/pages/{document_id}/pNNNN.jpg) when a render exists for that page.

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

// QA FIX (Issue 7): askQuestion() previously had no timeout at all when
// called without an explicit `signal` (the only way App.jsx calls it
// today), so a hung backend request left the UI spinning indefinitely.
// 2 minutes is generous enough for a slow CPU-fallback generation or a
// GPU pod that finished its (separately health-checked) cold start but
// is still under load -- this is not meant to cover the documented
// 10-20 minute cold-start window itself, only a single /ask call.
const DEFAULT_ASK_TIMEOUT_MS = 120_000;

async function parseErrorDetail(res) {
  try {
    const body = await res.json();
    return body.detail || res.statusText;
  } catch {
    return res.statusText;
  }
}

export async function askQuestion(query, signal) {
  // Preserve the existing signature/behavior for any caller that already
  // supplies its own AbortSignal; only fall back to a default timeout
  // when none is given.
  const effectiveSignal = signal ?? AbortSignal.timeout(DEFAULT_ASK_TIMEOUT_MS);

  const res = await fetch(`${API_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
    signal: effectiveSignal,
  });

  if (!res.ok) {
    throw new Error(await parseErrorDetail(res));
  }
  return res.json(); // { answer, sources, confidence }
}

export async function getStatus() {
  const res = await fetch(`${API_URL}/status`);
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json();
}

export { API_URL };
