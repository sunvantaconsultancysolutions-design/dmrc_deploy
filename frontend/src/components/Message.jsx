// TASK 2 -- retrieved source cards removed from the UI (per manager
// request). The backend is untouched: /ask still runs full retrieval
// and still returns `sources`/`chunk_type` on every response exactly as
// before (see src/app.py's SourceItem / _build_sources()) -- this
// component simply no longer renders that data. SourceChip and the old
// buildGroundedLabel()/"Grounded in N Retrieved ..." formatting are no
// longer used here; SourceChip.jsx is left in place, unmodified, in
// case the source-card UI is reinstated later.

export default function Message({ role, content, isError, isLoading }) {
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
      </div>
    </div>
  );
}