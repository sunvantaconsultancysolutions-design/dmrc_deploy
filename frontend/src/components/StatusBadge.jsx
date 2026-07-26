import { useEffect, useState } from "react";
import { getStatus } from "../api.js";

export default function StatusBadge() {
  const [state, setState] = useState("checking"); // checking | ready | warming | offline

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const s = await getStatus();
        if (cancelled) return;
        // BUGFIX (pre-deployment review, confirmed Bug #2): previously
        // omitted s.gemma_model_loaded, so the badge could report
        // "Models ready" while Gemma -- by far the slowest model to warm
        // up (~18GB) and the one that actually generates the answer --
        // was still loading. A user's first real question could then
        // hang for minutes with the UI claiming everything was ready.
        const ready =
          s.dense_model_loaded &&
          s.reranker_model_loaded &&
          s.gemma_model_loaded &&
          s.chromadb_connected;
        setState(ready ? "ready" : "warming");
      } catch {
        if (!cancelled) setState("offline");
      }
    }

    poll();
    const id = setInterval(poll, 15000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const copy = {
    checking: "Connecting…",
    ready: "Models ready",
    warming: "Warming up GPU…",
    offline: "Backend unreachable",
  }[state];

  return (
    <div className={`status-badge status-badge--${state}`}>
      <span className="status-badge__dot" />
      {copy}
    </div>
  );
}