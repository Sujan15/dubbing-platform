// Thin fetch wrapper: JSON in/out, uniform ApiError, AbortSignal support, upload progress via XHR.

export const API_BASE = "/api/v1";

export class ApiError extends Error {
  readonly status: number;
  readonly details: string[];

  constructor(status: number, message: string, details: string[] = []) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}

function messageFrom(body: unknown, fallback: string): { message: string; details: string[] } {
  if (body && typeof body === "object") {
    const b = body as { detail?: unknown; errors?: unknown };
    const details = Array.isArray(b.errors) ? b.errors.map(String) : [];
    if (typeof b.detail === "string") return { message: b.detail, details };
    if (Array.isArray(b.detail)) {
      // FastAPI request validation errors
      const msgs = b.detail.map((d: { loc?: unknown[]; msg?: string }) =>
        `${(d.loc ?? []).slice(1).join(".")}: ${d.msg ?? "invalid"}`,
      );
      return { message: "Request validation failed", details: msgs };
    }
    if (b.detail && typeof b.detail === "object") {
      const problems = (b.detail as { configuration_problems?: string[] }).configuration_problems;
      if (problems) return { message: "Server configuration problems", details: problems };
    }
  }
  return { message: fallback, details: [] };
}

export async function request<T>(
  path: string,
  init: RequestInit & { json?: unknown } = {},
): Promise<T> {
  const { json, headers, ...rest } = init;
  const res = await fetch(`${API_BASE}${path}`, {
    ...rest,
    headers: { Accept: "application/json", ...(json !== undefined ? { "Content-Type": "application/json" } : {}), ...headers },
    body: json !== undefined ? JSON.stringify(json) : rest.body,
  });
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  let body: unknown = undefined;
  try {
    body = text ? JSON.parse(text) : undefined;
  } catch {
    body = text;
  }
  if (!res.ok) {
    const { message, details } = messageFrom(body, `${res.status} ${res.statusText}`);
    throw new ApiError(res.status, message, details);
  }
  return body as T;
}

export interface UploadHandle<T> {
  promise: Promise<T>;
  abort: () => void;
}

/** multipart/form-data upload with real byte progress (fetch cannot report upload progress). */
export function upload<T>(path: string, form: FormData, onProgress?: (fraction: number) => void): UploadHandle<T> {
  const xhr = new XMLHttpRequest();
  const promise = new Promise<T>((resolve, reject) => {
    xhr.open("POST", `${API_BASE}${path}`);
    xhr.responseType = "text";
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      let body: unknown = undefined;
      try {
        body = xhr.responseText ? JSON.parse(xhr.responseText) : undefined;
      } catch {
        body = xhr.responseText;
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as T);
      else {
        const { message, details } = messageFrom(body, `${xhr.status} ${xhr.statusText}`);
        reject(new ApiError(xhr.status, message, details));
      }
    };
    xhr.onerror = () => reject(new ApiError(0, "Network error during upload"));
    xhr.onabort = () => reject(new ApiError(0, "Upload cancelled"));
    xhr.send(form);
  });
  return { promise, abort: () => xhr.abort() };
}

export function errorText(e: unknown): string {
  if (e instanceof ApiError) return e.details.length ? `${e.message}: ${e.details.join("; ")}` : e.message;
  if (e instanceof Error) return e.message;
  return String(e);
}
