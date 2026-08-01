// TASK 2 note (superseded by SVS-DMRC-2026-03, WP-9): source cards were
// removed from this component per an earlier manager request; the
// backend was left untouched throughout, still returning `sources` on
// every response (see src/app.py's SourceItem / _build_sources()).
// The PDF Evidence Viewer spec brings a source-chip strip back, now as
// the click-through into the scanned-page viewer rather than the old
// static card -- SourceChip.jsx below is the same component, updated
// for click behaviour.
import SourceChip from "./SourceChip.jsx";

export default function Message({ role, content, isError, isLoading, sources, activeChunkId, onViewSource }) {
  const isUser = role === "user";

  return (
    <div className={`msg-row ${isUser ? "msg-row--user" : "msg-row--assistant"}`}>
      <div className="msg-avatar" aria-hidden="true">
        {isUser ? "You" : "SV"}
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

        {!isUser && !isLoading && sources && sources.length > 0 && (
          <div className="source-chip-row">
            {sources.map((s, i) => (
              <SourceChip
                key={s.chunk_id || i}
                source={s}
                active={activeChunkId != null && s.chunk_id === activeChunkId}
                onView={onViewSource ? () => onViewSource(i) : undefined}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}