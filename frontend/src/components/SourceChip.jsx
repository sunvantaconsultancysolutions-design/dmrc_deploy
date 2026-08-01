// PHASE 2 (Evidence Viewer) -- Feature 2 (active highlight) + Feature 3
// (richer cards: clause title / BOQ description, "Unlabelled clause"
// only as a genuine last resort). All fields below already exist on
// the wire (app.py's SourceItem) -- clause/item_number/page/document/
// reranker_score/image_url are unchanged from before; heading and
// description are the two new additive fields from this phase.
export default function SourceChip({ source, active, onView }) {
  const {
    clause,
    heading,
    item_number,
    description,
    page,
    document,
    reranker_score,
    image_url,
  } = source;

  const isBoq = Boolean(item_number) && !clause;

  // Primary label: the identifier (clause number or BOQ item number).
  // Secondary label: the human-readable title/description, when the
  // backend actually transcribed one for this chunk. "Unlabelled
  // clause" is now only shown when NEITHER an identifier NOR a
  // title/description is available -- previously it appeared any time
  // `clause` was empty, even when a perfectly good heading existed.
  const primary = clause || item_number || null;
  const secondary = isBoq ? description : heading;
  const label = primary || secondary || "Unlabelled clause";

  const clickable = Boolean(image_url && onView);

  return (
    <div
      className={
        "source-chip" +
        (clickable ? " source-chip--link" : "") +
        (active ? " source-chip--active" : "")
      }
      role={clickable ? "button" : undefined}
      tabIndex={clickable ? 0 : undefined}
      aria-current={active ? "true" : undefined}
      onClick={clickable ? onView : undefined}
      onKeyDown={clickable ? (e) => e.key === "Enter" && onView() : undefined}
      title={clickable ? "View the scanned page" : undefined}
    >
      <span className="source-chip__marker" aria-hidden="true" />
      <div className="source-chip__body">
        <span className="source-chip__clause">{label}</span>
        {primary && secondary && (
          <span className="source-chip__title">{secondary}</span>
        )}
        <span className="source-chip__meta">
          {document ? document : "Scope of Work"}
          {page != null ? ` · p.${page}` : ""}
          {clickable ? " · view page" : ""}
        </span>
      </div>
      {reranker_score != null && (
        <span className="source-chip__score" title="Reranker relevance score">
          {(reranker_score * 100).toFixed(0)}%
        </span>
      )}
    </div>
  );
}