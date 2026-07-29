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
      "What are the contractor's obligations?",
      "Who is responsible for maintenance during the defects liability period?",
    ],
  },
  {
   icon: "📄",
    label: "Contract Clauses",
    questions: [
      "What documents form the contract?",
      "Who is responsible for obtaining clearances?",
    ],
  },
];

export default function App() {
  const [messages, setMessages] = useState([]);
  const [pending, setPending] = useState(false);
  const [viewed, setViewed] = useState(null); // a SourceItem or null
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
      const firstWithImage = (sources || []).find((s) => s.image_url);
      if (firstWithImage) setViewed(firstWithImage);
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
            <p className="empty-state__eyebrow">Contract Intelligence · Government &amp; PSU</p>
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
              <Message key={i} {...m} onViewSource={setViewed} />
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
      <PageViewer source={viewed} onClose={() => setViewed(null)} />
      </div>
    </div>
  );
}