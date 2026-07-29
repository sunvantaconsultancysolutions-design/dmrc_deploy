export default function SourceChip({ source, onView }) {
  const { clause, page, document, item_number, reranker_score, image_url } = source;

  const label = clause || item_number || "Unlabelled clause";
  const clickable = Boolean(image_url && onView);

  return (
    <div
      className={"source-chip" + (clickable ? " source-chip--link" : "")}
      role={clickable ? "button" : undefined}
      tabIndex={clickable ? 0 : undefined}
      onClick={clickable ? () => onView(source) : undefined}
      onKeyDown={clickable ? (e) => e.key === "Enter" && onView(source) : undefined}
      title={clickable ? "View the scanned page" : undefined}
    >
      <span className="source-chip__marker" aria-hidden="true" />
      <div className="source-chip__body">
        <span className="source-chip__clause">{label}</span>
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
