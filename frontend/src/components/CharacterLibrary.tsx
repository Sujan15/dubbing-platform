import { useState } from "react";
import { Link } from "react-router-dom";
import { characterApi, voiceApi } from "../api";
import { errorText } from "../api/client";
import type { SeriesCharacter, VoiceProfile } from "../api/types";
import { useAsync } from "../hooks/useAsync";
import { Badge, Button, EmptyState, ErrorBanner, Spinner } from "./ui";

export default function CharacterLibrary({ projectId }: { projectId: string }) {
  const characters = useAsync((signal) => characterApi.list(projectId, signal), [projectId]);
  const voices = useAsync(
    (signal) => voiceApi.list({ approved: true, include_inactive: false, include_legacy: false }, signal),
    [],
  );
  const profiles = (voices.data?.voices ?? []).filter((voice) => voice.selectable);
  const [name, setName] = useState("");
  const [role, setRole] = useState("");
  const [gender, setGender] = useState("");
  const [initialVoice, setInitialVoice] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if ((characters.loading && !characters.data) || (voices.loading && !voices.data)) {
    return <Spinner label="Loading characters and approved voices" />;
  }

  const setCharacterVoice = async (character: SeriesCharacter, voiceId: string) => {
    if (!voiceId) return;
    setError(null);
    try {
      await characterApi.addVoice(projectId, character.character_id, {
        voice_source: "cast",
        voice_asset_id: voiceId,
        label: profiles.find((voice) => voice.voice_id === voiceId)?.voice_name ?? "",
        make_active: true,
        approved: true,
      });
      characters.reload();
    } catch (e) {
      setError(errorText(e));
    }
  };

  return (
    <div className="stack-lg">
      <ErrorBanner error={characters.error ?? voices.error ?? error} onRetry={() => { characters.reload(); voices.reload(); }} />
      {profiles.length === 0 && (
        <div className="banner banner-info">
          Approve and activate a profile in the <Link to="/voices">Voice Library</Link> before assigning it to a character.
        </div>
      )}
      <form className="panel stack" onSubmit={async (event) => {
        event.preventDefault();
        if (!name.trim()) return;
        setBusy(true);
        setError(null);
        try {
          const created = await characterApi.create(projectId, {
            canonical_name: name.trim(),
            ...(role ? { role } : {}),
            ...(gender ? { gender } : {}),
          });
          if (initialVoice) await setCharacterVoice(created, initialVoice);
          setName("");
          setRole("");
          setGender("");
          setInitialVoice("");
          characters.reload();
        } catch (e) {
          setError(errorText(e));
          characters.reload();
        } finally {
          setBusy(false);
        }
      }}>
        <h2>Add series character</h2>
        <div className="row gap wrap">
          <label className="field">
            <span>Character name</span>
            <input className="input" value={name} onChange={(e) => setName(e.target.value)} maxLength={80} required />
          </label>
          <label className="field">
            <span>Role</span>
            <select className="input" value={role} onChange={(e) => setRole(e.target.value)}>
              <option value="">Not specified</option>
              <option value="hero">Hero</option>
              <option value="heroine">Heroine</option>
              <option value="villain">Villain</option>
              <option value="support">Supporting</option>
              <option value="narrator">Narrator</option>
              <option value="other">Other</option>
            </select>
          </label>
          <label className="field">
            <span>Gender</span>
            <select className="input" value={gender} onChange={(e) => setGender(e.target.value)}>
              <option value="">Not specified</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="other">Other</option>
            </select>
          </label>
          <label className="field">
            <span>Approved voice profile (optional)</span>
            <select className="input" value={initialVoice} onChange={(e) => setInitialVoice(e.target.value)}>
              <option value="">Assign later</option>
              {profiles.map((voice) => <option key={voice.voice_id} value={voice.voice_id}>{voice.voice_name} · {voice.engine}</option>)}
            </select>
          </label>
        </div>
        <Button type="submit" variant="primary" busy={busy} disabled={!name.trim()}>Create character</Button>
      </form>

      <div className="section-head">
        <h2>Characters ({characters.data?.length ?? 0})</h2>
        <Button size="sm" onClick={characters.reload}>Refresh</Button>
      </div>
      {!characters.data?.length ? (
        <EmptyState title="No characters in this project">Create a character to manage series identity and a reusable approved voice.</EmptyState>
      ) : (
        <div className="stack">
          {characters.data.map((character) => (
            <CharacterCard key={character.character_id} character={character} voices={profiles}
              onVoiceChange={(voiceId) => setCharacterVoice(character, voiceId)} />
          ))}
        </div>
      )}
    </div>
  );
}

function CharacterCard({ character, voices, onVoiceChange }: {
  character: SeriesCharacter;
  voices: VoiceProfile[];
  onVoiceChange: (voiceId: string) => void;
}) {
  const activeProfile = character.voice_profiles.find((profile) => profile.voice_profile_id === character.active_voice_profile_id);
  const activeVoice = voices.find((voice) => voice.voice_id === activeProfile?.voice_asset_id);
  return (
    <article className="panel stack">
      <div className="row gap wrap">
        <h3>{character.canonical_name}</h3>
        {character.role && <Badge>{character.role}</Badge>}
        <Badge tone={character.has_approved_voice ? "ok" : "warn"}>
          {character.has_approved_voice ? "Approved voice assigned" : "No approved voice"}
        </Badge>
        <span className="muted small">{character.embedding_count} identity samples</span>
      </div>
      {character.description && <p>{character.description}</p>}
      <div className="row gap-sm wrap">
        <span>{activeVoice ? `${activeVoice.voice_name} · ${activeVoice.engine}` : "No active voice profile"}</span>
        <select className="input input-sm" aria-label={`Assign voice to ${character.canonical_name}`}
          value="" onChange={(e) => onVoiceChange(e.target.value)} disabled={voices.length === 0}>
          <option value="">Assign approved profile…</option>
          {voices.map((voice) => <option key={voice.voice_id} value={voice.voice_id}>{voice.voice_name} · {voice.engine}</option>)}
        </select>
      </div>
    </article>
  );
}
