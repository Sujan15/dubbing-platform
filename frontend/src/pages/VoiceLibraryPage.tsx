import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { voiceApi, urls } from "../api";
import { errorText } from "../api/client";
import type { VoiceAudition, VoiceEngineInfo, VoiceProfile } from "../api/types";
import { useAsync } from "../hooks/useAsync";
import { Badge, Button, EmptyState, ErrorBanner, Spinner } from "../components/ui";

export default function VoiceLibraryPage() {
  const engines = useAsync((signal) => voiceApi.engines(signal), []);
  const voices = useAsync((signal) => voiceApi.list({ include_legacy: false, include_inactive: true }, signal), []);
  const enabled = (engines.data?.engines ?? []).filter((engine) => engine.enabled && !engine.legacy);
  const [selectedEngine, setSelectedEngine] = useState("");
  const [auditionId, setAuditionId] = useState("");
  const [auditionVoiceId, setAuditionVoiceId] = useState("");
  const audition = useAsync<VoiceAudition | null>(
    (signal) => auditionId ? voiceApi.auditionStatus(auditionId, signal) : Promise.resolve(null),
    [auditionId],
    { pollMs: 1200, shouldPoll: (result) => !!result && ["queued", "running"].includes(result.status) },
  );
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedEngine && engines.data?.default_engine) setSelectedEngine(engines.data.default_engine);
  }, [engines.data, selectedEngine]);

  if (engines.loading && !engines.data) return <Spinner label="Loading TTS engines" />;

  return (
    <section className="page">
      <div className="crumbs"><Link to="/">Projects</Link> / <span>Voice Library</span></div>
      <div className="page-head">
        <div>
          <h1>Voice Library</h1>
          <p className="muted">Create engine-backed profiles, audition them, then approve profiles for character assignment.</p>
        </div>
        <Link className="btn btn-secondary btn-md" to="/">Projects</Link>
      </div>
      <ErrorBanner error={engines.error ?? voices.error ?? error} onRetry={() => { engines.reload(); voices.reload(); }} />

      <div className="split">
        <VoiceCreateForm
          engines={enabled}
          selectedEngine={selectedEngine}
          onEngine={setSelectedEngine}
          onCreated={() => { setError(null); voices.reload(); }}
          onError={setError}
        />
        <div className="panel stack">
          <h2>Engine availability</h2>
          {enabled.map((engine) => (
            <div className="row gap-sm" key={engine.engine}>
              <Badge tone={engine.available ? "ok" : "warn"}>{engine.available ? "Available" : "Unavailable"}</Badge>
              <strong>{engine.engine}</strong>
              <span className="muted small">{engine.model_id ?? ""}</span>
              {!engine.available && <span className="warn-text small">{engine.reason}</span>}
            </div>
          ))}
          {enabled.length === 0 && <p className="muted">No supported TTS engines are enabled by the backend.</p>}
        </div>
      </div>

      <div className="section-head">
        <h2>Profiles ({voices.data?.voices.length ?? 0})</h2>
        <Button size="sm" onClick={voices.reload}>Refresh</Button>
      </div>
      {voices.loading && !voices.data ? <Spinner label="Loading profiles" /> : null}
      {!voices.loading && voices.data?.voices.length === 0 ? (
        <EmptyState title="No voice profiles yet">Create a profile above. New profiles require a successful audition before approval.</EmptyState>
      ) : (
        <div className="stack">
          {(voices.data?.voices ?? []).map((voice) => (
            <VoiceCard
              key={voice.voice_id}
              voice={voice}
              activeAudition={auditionVoiceId === voice.voice_id ? audition.data : null}
              auditionLoading={auditionVoiceId === voice.voice_id && audition.loading}
              onAudition={async (text) => {
                setError(null);
                try {
                  const started = await voiceApi.audition(voice.voice_id, text);
                  setAuditionVoiceId(voice.voice_id);
                  setAuditionId(started.audition_id);
                  audition.reload();
                } catch (e) {
                  setError(errorText(e));
                }
              }}
              onApproved={async (approved) => {
                setError(null);
                try {
                  await voiceApi.approve(voice.voice_id, approved);
                  voices.reload();
                } catch (e) {
                  setError(errorText(e));
                }
              }}
              onActive={async (active) => {
                setError(null);
                try {
                  await voiceApi.update(voice.voice_id, { active });
                  voices.reload();
                } catch (e) {
                  setError(errorText(e));
                }
              }}
            />
          ))}
        </div>
      )}
    </section>
  );
}

function VoiceCreateForm(props: {
  engines: VoiceEngineInfo[];
  selectedEngine: string;
  onEngine: (engine: string) => void;
  onCreated: () => void;
  onError: (message: string | null) => void;
}) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [prompt, setPrompt] = useState("");
  const [transcript, setTranscript] = useState("");
  const [reference, setReference] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const engine = props.engines.find((item) => item.engine === props.selectedEngine);
  const promptRequired = !!engine?.capabilities.prompt_conditioned;
  const referenceRequired = !!engine?.capabilities.needs_reference_transcript;

  return (
    <form
      className="panel stack"
      onSubmit={async (event) => {
        event.preventDefault();
        props.onError(null);
        setBusy(true);
        try {
          const created = await voiceApi.create({
            voice_name: name.trim(),
            engine: props.selectedEngine,
            description: description.trim(),
            ...(promptRequired ? { voice_description_prompt: prompt.trim() } : {}),
            ...(referenceRequired ? { reference_transcript: transcript.trim() } : {}),
          });
          props.onCreated();
          if (referenceRequired && reference) {
            await voiceApi.uploadReference(created.voice_id, reference, transcript.trim());
          }
          setName("");
          setDescription("");
          setPrompt("");
          setTranscript("");
          setReference(null);
        } catch (error) {
          props.onError(errorText(error));
        } finally {
          setBusy(false);
        }
      }}
    >
      <h2>Create a profile</h2>
      <label className="field">
        <span>Voice engine</span>
        <select className="input" value={props.selectedEngine} onChange={(e) => props.onEngine(e.target.value)} required>
          {props.engines.map((item) => <option key={item.engine} value={item.engine}>{item.engine}</option>)}
        </select>
      </label>
      <label className="field">
        <span>Profile name</span>
        <input className="input" value={name} onChange={(e) => setName(e.target.value)} maxLength={80} required />
      </label>
      <label className="field">
        <span>Description</span>
        <textarea className="input" rows={2} value={description} onChange={(e) => setDescription(e.target.value)} maxLength={2000} />
      </label>
      {promptRequired && (
        <label className="field">
          <span>Versioned voice description prompt</span>
          <textarea className="input" rows={4} value={prompt} onChange={(e) => setPrompt(e.target.value)} maxLength={2000} required />
        </label>
      )}
      {referenceRequired && (
        <>
          <label className="field">
            <span>Reference transcript</span>
            <textarea className="input" rows={2} value={transcript} onChange={(e) => setTranscript(e.target.value)} maxLength={4000} required />
          </label>
          <label className="field">
            <span>Reference audio</span>
            <input className="input" type="file" accept=".wav,.mp3,.m4a,.flac,.ogg" onChange={(e) => setReference(e.target.files?.[0] ?? null)} />
          </label>
          {!reference && <p className="muted small">A reference audio file is needed before this profile can be auditioned.</p>}
        </>
      )}
      <Button variant="primary" type="submit" busy={busy}
        disabled={!props.selectedEngine || !name.trim() || (promptRequired && !prompt.trim()) || (referenceRequired && (!reference || !transcript.trim()))}>
        Create profile
      </Button>
    </form>
  );
}

function VoiceCard(props: {
  voice: VoiceProfile;
  activeAudition: VoiceAudition | null | undefined;
  auditionLoading: boolean;
  onAudition: (text: string) => Promise<void>;
  onApproved: (approved: boolean) => Promise<void>;
  onActive: (active: boolean) => Promise<void>;
}) {
  const { voice } = props;
  const [text, setText] = useState(voice.sample_text ?? "");
  const previewRunId = props.activeAudition?.status === "done"
    ? props.activeAudition.audition_id
    : voice.preview?.run_id;
  const canApprove = !!previewRunId && !props.activeAudition?.error;
  return (
    <article className="panel stack">
      <div className="row gap wrap">
        <h3>{voice.voice_name}</h3>
        <Badge>{voice.engine}</Badge>
        <Badge tone={voice.approved && voice.selectable ? "ok" : voice.approved ? "warn" : "neutral"}>
          {voice.selectable ? "Approved & active" : voice.approved ? "Needs re-audition" : "Not approved"}
        </Badge>
        {!voice.active && <Badge tone="warn">Inactive</Badge>}
        <span className="muted small mono">{voice.voice_id}</span>
      </div>
      {voice.description && <p>{voice.description}</p>}
      {voice.voice_description_prompt && (
        <details>
          <summary>Prompt v{voice.prompt_version}</summary>
          <p>{voice.voice_description_prompt}</p>
          <p className="muted small">Prompt history: {voice.prompt_versions.map((version) => `v${version.version}`).join(", ")}</p>
        </details>
      )}
      <div className="row gap-sm wrap">
        <input className="input input-sm" aria-label={`Audition text for ${voice.voice_name}`} value={text}
          onChange={(e) => setText(e.target.value)} maxLength={500} />
        <Button size="sm" onClick={() => props.onAudition(text)} busy={props.auditionLoading} disabled={!text.trim()}>
          Audition
        </Button>
        {voice.approved ? (
          <Button size="sm" variant="ghost" onClick={() => props.onApproved(false)}>Revoke approval</Button>
        ) : (
          <Button size="sm" variant="primary" onClick={() => props.onApproved(true)} disabled={!canApprove}>
            Approve after audition
          </Button>
        )}
        <Button size="sm" variant="ghost" onClick={() => props.onActive(!voice.active)}>
          {voice.active ? "Deactivate" : "Reactivate"}
        </Button>
      </div>
      {props.activeAudition?.status === "running" || props.activeAudition?.status === "queued" ? (
        <p className="muted small">Audition {props.activeAudition.status}…</p>
      ) : null}
      {props.activeAudition?.status === "failed" && (
        <div className="banner banner-error">{props.activeAudition.error ?? "Audition failed."}</div>
      )}
      {props.activeAudition?.status === "done" && props.activeAudition.problems?.length ? (
        <div className="banner banner-warn">{props.activeAudition.problems.join(" · ")}</div>
      ) : null}
      {previewRunId && (
        <audio controls preload="none" src={urls.voicePreview(voice.voice_id, previewRunId)} aria-label={`${voice.voice_name} audition`} />
      )}
    </article>
  );
}
