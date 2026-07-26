import SourceChip from "./SourceChip.jsx";

// TASK 3 -- UI retrieval label fix.
//
// Previously this component always rendered "Grounded in N retrieved
// clause(s)", even when every retrieved chunk was actually a BOQ item
// (or a mix of both) -- misleading for any BOQ-only or mixed answer.
// The backend now sends each source's real `chunk_type` ("clause" or
// "boq", from metadata.get("chunk_type"); see src/app.py's SourceItem /
// _build_sources()). This derives the label from that metadata instead
// of hardcoding "clauses", per the task's instruction not to hardcode
// the wording.
function buildGroundedLabel(sources) {
  const count = sources.length;
  const types = new Set(sources.map((s) => s.chunk_type).filter(Boolean));

  if (types.size === 1) {
    const [onlyType] = types;
    if (onlyType === "boq") {
      return `Grounded in ${count} Retrieved BOQ ${count === 1 ? "Item" : "Items"}`;
    }
    if (onlyType === "clause") {
      return `Grounded in ${count} Retrieved ${count === 1 ? "Clause" : "Clauses"}`;
    }
  }

  // Mixed chunk types, or chunk_type missing on some/all sources (older
  // backend response shape) -- fall back to a neutral, always-accurate
  // label rather than guessing.
  return `Grounded in ${count} Retrieved ${count === 1 ? "Document" : "Documents"}`;
}

export default function Message({ role, content, sources, isError, isLoading }) {
  const isUser = role === "user";

  return (
    <div className={`msg-row ${isUser ? "msg-row--user" : "msg-row--assistant"}`}>
      <div className="msg-avatar" aria-hidden="true">
        {isUser ? "You" : "DM"}
      </div>

      <div className="msg-column">
        <div className={`msg-bubble ${isError ? "msg-bubble--error" : ""}`}>
          {isLoading ? (
            <span className="typing-dots" aria-label="Generating answer">
              <span />
              <span />
              <span />
            </span>
          ) : (
            <p>{content}</p>
          )}
        </div>

        {!isUser && sources && sources.length > 0 && (
          <div className="sources-rail">
            <span className="sources-rail__label">{buildGroundedLabel(sources)}</span>
            <div className="sources-rail__track">
              {sources.map((s, i) => (
                <SourceChip key={s.chunk_id || i} source={s} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}