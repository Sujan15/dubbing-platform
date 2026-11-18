import { Link, NavLink, Route, Routes } from "react-router-dom";
import { metaApi } from "./api";
import { useAsync } from "./hooks/useAsync";
import Dashboard from "./pages/Dashboard";
import JobPage from "./pages/JobPage";
import ProjectPage from "./pages/ProjectPage";
import ReviewPage from "./pages/ReviewPage";
import VoiceLibraryPage from "./pages/VoiceLibraryPage";
import { EmptyState } from "./components/ui";

export default function App() {
  const meta = useAsync((s) => metaApi.get(s), []);
  const problems = meta.data?.configuration_problems ?? [];
  return (
    <div className="app">
      <header className="topbar">
        <Link to="/" className="brand">
          <span className="brand-mark" aria-hidden>
            ▮▮▮
          </span>
          Dubbing Workstation
        </Link>
        <nav>
          <NavLink to="/" end>
            Projects
          </NavLink>
          <NavLink to="/voices">Voice Library</NavLink>
          <a href="/docs" target="_blank" rel="noreferrer">
            API docs
          </a>
        </nav>
      </header>
      {problems.length > 0 && (
        <div className="banner banner-warn config-banner" role="alert">
          <strong>Server configuration:</strong> {problems.join(" · ")}
        </div>
      )}
      <main className="main">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/voices" element={<VoiceLibraryPage />} />
          <Route path="/projects/:projectId" element={<ProjectPage meta={meta.data} />} />
          <Route path="/jobs/:jobId" element={<JobPage />} />
          <Route path="/jobs/:jobId/review" element={<ReviewPage meta={meta.data} />} />
          <Route path="*" element={<EmptyState title="Page not found"><Link to="/">Back to projects</Link></EmptyState>} />
        </Routes>
      </main>
    </div>
  );
}
