import { useEffect, useRef, useState } from "react";
import Message from "./components/Message.jsx";
import ChatInput from "./components/ChatInput.jsx";
import StatusBadge from "./components/StatusBadge.jsx";
import PageViewer from "./components/PageViewer.jsx";
import { askQuestion } from "./api.js";

// TASK 1 -- Home-screen suggested questions, grouped by document type.
// Deliberately phrased with no clause numbers or BOQ item numbers (demo
// users don't know document identifiers) -- see task instructions.
const SUGGESTION_GROUPS = [
  {
    icon: "📄",
    label: "Contract Clauses",
    questions: [
      "What is the penalty for delay?",
      "What is the defects liability period?",
    ],
  },
  {
    icon: "📋",
    label: "Contract Formation",
    questions: [
      "What documents form the contract?",
      "Who is responsible for obtaining clearances?",
    ],
  },
  {
    icon: "🧮",
    label: "Bill of Quantities",
    questions: [
      "How much is Part-H worth?",
      "What discount was offered?",
    ],
  },
];

export default function App() {
  const [messages, setMessages] = useState([]);
  const [pending, setPending] = useState(false);
  // PHASE 2 (Evidence Viewer) -- Feature 1: Previous/Next now walk the
  // retrieved evidence list for the current answer (ordered by retrieval
  // score, i.e. `sources` as returned by the API) instead of the PDF's
  // own page sequence. `active` tracks *which* message's evidence list
  // is open and *which position* in it, rather than a single detached
  // source object -- that position is what both the PageViewer's
  // Previous/Next buttons and the highlighted SourceChip (Feature 2)
  // read from, so the two always stay in sync by construction.
  const [active, setActive] = useState(null); // { msgIndex, index } | null
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, pending]);

  async function handleSend(query) {
    const userMsg = { role: "user", content: query };
    setMessages((m) => [...m, userMsg]);
    setPending(true);

    try {
      const { answer, sources, confidence } = await askQuestion(query);
      const newMsgIndex = messages.length + 1; // this assistant message's future index
      const firstImageIndex = (sources || []).findIndex((s) => s.image_url);
      if (firstImageIndex !== -1) {
        setActive({ msgIndex: newMsgIndex, index: firstImageIndex });
      }
      setMessages((m) => [
        ...m,
        { role: "assistant", content: answer, sources, confidence },
      ]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: `Something went wrong reaching the model: ${err.message}. Check that the backend pod is running and ALLOWED_ORIGINS includes this site.`,
          isError: true,
        },
      ]);
    } finally {
      setPending(false);
    }
  }

  // The evidence list currently open in the PageViewer, and the active
  // source within it -- both derived from `active` + `messages` so
  // there is exactly one source of truth (no risk of the viewer and the
  // highlighted chip drifting apart, per Feature 2).
  const activeSources = active ? messages[active.msgIndex]?.sources || [] : null;
  const activeSource = activeSources ? activeSources[active.index] : null;

  function viewEvidence(msgIndex, index) {
    setActive({ msgIndex, index });
  }

  function navigateEvidence(newIndex) {
    if (!active) return;
    setActive({ msgIndex: active.msgIndex, index: newIndex });
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header__brand">
          <span className="app-header__mark">SV</span>
          <div>
            <h1>Sunvanta Consultancy Solutions</h1>
            <p>Customer Intelligent Chat</p>
          </div>
        </div>
        <StatusBadge />
      </header>

      <div className="app-columns">
      <div className="chat-column">
      <main className="chat-scroll" ref={scrollRef}>
        {messages.length === 0 ? (
          <div className="empty-state">
            <p className="empty-state__eyebrow">Contract Intelligence</p>
            <h2>
              Ask your contract.
              <br />
              Get the <span className="accent">clause</span>, not a guess.
            </h2>
            <p className="empty-state__sub">
              Query any clause, obligation, or scope item directly. Every answer is grounded
              in the retrieved contract text, with sources you can verify in seconds.
            </p>

            <div className="trust">
              <div className="trust-item">
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#1F7A6D" strokeWidth="2.2">
                  <path d="M12 2l8 4v6c0 5-3.5 8.5-8 10-4.5-1.5-8-5-8-10V6l8-4z" />
                </svg>
                <span>
                  <b>Grounded answers</b>
                  <small>Responses are retrieved from the indexed contract text, not guessed.</small>
                </span>
              </div>
              <div className="trust-item">
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#1F7A6D" strokeWidth="2.2">
                  <path d="M9 12l2 2 4-5" />
                  <circle cx="12" cy="12" r="10" />
                </svg>
                <span>
                  <b>Source-linked</b>
                  <small>Every answer can surface the clauses it was drawn from.</small>
                </span>
              </div>
              <div className="trust-item">
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#1F7A6D" strokeWidth="2.2">
                  <path d="M4 6h16M4 12h16M4 18h10" />
                </svg>
                <span>
                  <b>Hybrid retrieval</b>
                  <small>Dense embedding search combined with reranking for relevance.</small>
                </span>
              </div>
              <div className="trust-item">
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#1F7A6D" strokeWidth="2.2">
                  <path d="M12 8v5l3 2" />
                  <circle cx="12" cy="12" r="10" />
                </svg>
                <span>
                  <b>Verify before relying</b>
                  <small>Always cross-check answers against the source document.</small>
                </span>
              </div>
            </div>

            <p className="section-label">Try asking</p>
            {SUGGESTION_GROUPS.map((group) => (
              <div className="suggestion-group" key={group.label}>
                <p className="suggestion-group__label">
                  <span aria-hidden="true">{group.icon}</span> {group.label.toUpperCase()}
                </p>
                <div className="suggestion-grid">
                  {group.questions.map((s) => (
                    <button key={s} onClick={() => handleSend(s)} className="suggestion-card">
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="msg-list">
            {messages.map((m, i) => (
              <Message
                key={i}
                {...m}
                msgIndex={i}
                activeChunkId={active && active.msgIndex === i ? activeSource?.chunk_id : null}
                onViewSource={(index) => viewEvidence(i, index)}
              />
            ))}
            {pending && <Message role="assistant" isLoading />}
          </div>
        )}
      </main>

      <footer className="app-footer">
        <ChatInput onSend={handleSend} disabled={pending} />
        <p className="app-footer__note">
          Answers are grounded in retrieved contract clauses. <b>Verify against the source
          document</b> before relying on them.
        </p>
        <p className="app-footer__brand">
          SUNVANTA CONSULTANCY SOLUTIONS · CUSTOMER INTELLIGENT CHAT
        </p>
      </footer>
      </div>
      <PageViewer
        sources={activeSources}
        activeIndex={active?.index ?? null}
        onNavigate={navigateEvidence}
        onClose={() => setActive(null)}
      />
      </div>
    </div>
  );
}