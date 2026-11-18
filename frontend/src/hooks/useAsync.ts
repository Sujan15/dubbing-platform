import { useCallback, useEffect, useRef, useState } from "react";

export interface AsyncState<T> {
  data: T | undefined;
  error: unknown;
  loading: boolean;
  reload: () => void;
  setData: (d: T) => void;
}

/**
 * Loads data with an AbortSignal; reloads when `deps` change. When `pollMs` is set and
 * `shouldPoll(data)` is true, the loader re-runs on that interval (used for job status).
 */
export function useAsync<T>(
  loader: (signal: AbortSignal) => Promise<T>,
  deps: unknown[],
  opts: { pollMs?: number; shouldPoll?: (d: T | undefined) => boolean } = {},
): AsyncState<T> {
  const [data, setData] = useState<T | undefined>(undefined);
  const [error, setError] = useState<unknown>(undefined);
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);
  const loaderRef = useRef(loader);
  loaderRef.current = loader;
  const pollRef = useRef(opts);
  pollRef.current = opts;

  useEffect(() => {
    const ctrl = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    let first = true;
    const run = async () => {
      if (first) setLoading(true);
      try {
        const d = await loaderRef.current(ctrl.signal);
        if (ctrl.signal.aborted) return;
        setData(d);
        setError(undefined);
        const { pollMs, shouldPoll } = pollRef.current;
        if (pollMs && (!shouldPoll || shouldPoll(d))) timer = setTimeout(run, pollMs);
      } catch (e) {
        if (ctrl.signal.aborted || (e instanceof DOMException && e.name === "AbortError")) return;
        setError(e);
        const { pollMs } = pollRef.current;
        if (pollMs) timer = setTimeout(run, pollMs * 3); // back off, keep trying
      } finally {
        if (!ctrl.signal.aborted && first) setLoading(false);
        first = false;
      }
    };
    run();
    return () => {
      ctrl.abort();
      if (timer) clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { data, error, loading, reload, setData };
}
