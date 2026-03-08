"""Session lifecycle and persistence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .models import Phase, Posture, Session
from .posture import select_posture

ALLOWED_TRANSITIONS: dict[Phase, list[Phase]] = {
    Phase.understand: [Phase.challenge],
    Phase.challenge: [Phase.synthesize],
    Phase.synthesize: [Phase.complete],
}


class SessionManager:
    """Manages session creation, persistence, and retrieval."""

    def __init__(self, base_path: str | Path) -> None:
        self.base_path = Path(base_path)
        self._sessions_dir = self.base_path / ".constrain" / "sessions"

    def create(self, posture_override: Posture | None = None) -> Session:
        posture = select_posture(posture_override)
        return Session(posture=posture)

    def save(self, session: Session) -> None:
        session.touch()
        first_time = not self._sessions_dir.exists()
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

        path = self._sessions_dir / f"{session.id}.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(session.model_dump_json(indent=2), encoding="utf-8")
            tmp.replace(path)
        except OSError as e:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to save session: {e}") from e

        if first_time:
            self._check_gitignore()

    def load(self, session_id: str) -> Session:
        path = self._sessions_dir / f"{session_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"No session found with ID '{session_id}'.")
        try:
            data = path.read_text(encoding="utf-8")
        except OSError as e:
            raise RuntimeError(f"Failed to read session file: {e}") from e
        try:
            return Session.model_validate_json(data)
        except Exception as e:
            raise ValueError(f"Session file is corrupted: {e}") from e

    def find_latest_incomplete(self) -> Session | None:
        if not self._sessions_dir.exists():
            return None
        latest: Session | None = None
        for p in self._sessions_dir.glob("*.json"):
            try:
                s = Session.model_validate_json(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if s.phase in (Phase.complete,):
                continue
            if latest is None or s.updated_at > latest.updated_at:
                latest = s
        return latest

    def list_all(self) -> list[dict]:
        if not self._sessions_dir.exists():
            return []
        results = []
        for p in self._sessions_dir.glob("*.json"):
            try:
                s = Session.model_validate_json(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            results.append({
                "id": s.id,
                "phase": s.phase.value,
                "posture": s.posture.value,
                "created_at": s.created_at,
                "updated_at": s.updated_at,
                "understand_rounds": s.understand_rounds,
                "challenge_rounds": s.challenge_rounds,
                "is_complete": s.phase == Phase.complete,
            })
        results.sort(key=lambda r: r["updated_at"], reverse=True)
        return results

    def transition_phase(self, session: Session, to_phase: Phase) -> None:
        allowed = ALLOWED_TRANSITIONS.get(session.phase, [])
        if to_phase not in allowed:
            raise ValueError(
                f"Cannot transition from '{session.phase.value}' to '{to_phase.value}'."
            )
        session.phase = to_phase
        session.touch()

    def _check_gitignore(self) -> None:
        gitignore = self.base_path / ".gitignore"
        if not gitignore.exists():
            print(
                "Tip: Add '.constrain/' to your .gitignore to keep session data out of version control.",
                file=sys.stderr,
            )
            return
        try:
            content = gitignore.read_text(encoding="utf-8")
        except OSError:
            return
        for line in content.splitlines():
            stripped = line.strip()
            if stripped in (".constrain", ".constrain/", ".constrain/*"):
                return
        print(
            "Tip: Add '.constrain/' to your .gitignore to keep session data out of version control.",
            file=sys.stderr,
        )
