import { useRef, useState } from "react";

export default function ChatInput({ onSend, disabled }) {
  const [value, setValue] = useState("");
  const textareaRef = useRef(null);

  function handleInput(e) {
    setValue(e.target.value);
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 160) + "px";
    }
  }

  function submit() {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  return (
    <div className="chat-input">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={handleInput}
        onKeyDown={handleKeyDown}
        placeholder="Ask about a clause, obligation, or scope of work…"
        rows={1}
        disabled={disabled}
      />
      <button
        type="button"
        className="chat-input__send"
        onClick={submit}
        disabled={disabled || !value.trim()}
        aria-label="Send question"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
          <path
            d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"
            stroke="currentColor"
            strokeWidth="2.2"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        </svg>
      </button>
    </div>
  );
}
