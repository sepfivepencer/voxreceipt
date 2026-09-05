"""Conservative descriptor-bound filesystem helpers for reports and batch audio."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

from voxreceipt.errors import SafetyError

_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


def _child_name(name: str, *, label: str) -> str:
    """Validate one directory entry name without normalizing filesystem bytes."""

    separators = {"/", os.sep}
    if os.altsep is not None:
        separators.add(os.altsep)
    if name in {"", ".", ".."} or "\0" in name or any(item in name for item in separators):
        raise SafetyError(f"{label} filename is invalid")
    return name


def _open_directory_components(path: Path) -> tuple[int, Path]:
    """Open every POSIX component with no-follow semantics and return the final fd."""

    absolute = Path(os.path.abspath(path))
    if os.name != "posix" or os.open not in os.supports_dir_fd:
        try:
            return os.open(absolute, _DIRECTORY_FLAGS), absolute
        except OSError as exc:
            raise SafetyError(
                "directory is missing, inaccessible, or not a link-free real directory"
            ) from exc

    fd: int | None = None
    try:
        fd = os.open(os.sep, _DIRECTORY_FLAGS)
        for component in absolute.parts[1:]:
            next_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd, absolute
    except OSError as exc:
        if fd is not None:
            os.close(fd)
        raise SafetyError(
            "directory is missing, inaccessible, or not a link-free real directory"
        ) from exc


@dataclass
class DirectoryHandle:
    """A checked directory whose descriptor remains bound to the inspected inode."""

    path: Path
    fd: int
    device: int
    inode: int
    _closed: bool = False

    @classmethod
    def open(cls, path: Path, *, empty: bool = False) -> DirectoryHandle:
        fd, absolute = _open_directory_components(path)
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISDIR(metadata.st_mode):
                raise SafetyError("path must be a real directory, not a link")
            if empty and os.listdir(fd):
                raise SafetyError("output directory must be empty")
            return cls(absolute, fd, metadata.st_dev, metadata.st_ino)
        except SafetyError:
            os.close(fd)
            raise
        except OSError as exc:
            os.close(fd)
            raise SafetyError("directory could not be inspected safely") from exc

    def close(self) -> None:
        if not self._closed:
            with suppress(OSError):
                os.close(self.fd)
            self._closed = True

    def still_names_path(self) -> bool:
        """Return whether the original absolute path still names this directory inode."""

        try:
            with DirectoryHandle.open(self.path) as current:
                return (current.device, current.inode) == (self.device, self.inode)
        except SafetyError:
            return False

    def __enter__(self) -> DirectoryHandle:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def checked_directory(path: Path, *, empty: bool = False) -> Path:
    """Return an absolute real directory after a component-wise no-follow open."""

    with DirectoryHandle.open(path, empty=empty) as directory:
        return directory.path


def open_or_create_empty_directory(path: Path) -> DirectoryHandle:
    """Create a private directory with mkdir-at or hold an existing empty directory."""

    name = _child_name(path.name, label="output")
    with DirectoryHandle.open(path.parent) as parent:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent.fd)
        except FileExistsError:
            pass
        except OSError as exc:
            raise SafetyError("output directory could not be created") from exc

        fd: int | None = None
        try:
            fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent.fd)
            metadata = os.fstat(fd)
            if not stat.S_ISDIR(metadata.st_mode):
                raise SafetyError("path must be a real directory, not a link")
            if os.listdir(fd):
                raise SafetyError("output directory must be empty")
        except SafetyError:
            if fd is not None:
                os.close(fd)
            raise
        except OSError as exc:
            if fd is not None:
                os.close(fd)
            raise SafetyError("output directory is linked or inaccessible") from exc
    return DirectoryHandle(
        path=Path(os.path.abspath(path)),
        fd=fd,
        device=metadata.st_dev,
        inode=metadata.st_ino,
    )


def create_or_check_empty_directory(path: Path) -> Path:
    """Create a private directory or validate that an existing one is empty."""

    with open_or_create_empty_directory(path) as directory:
        return directory.path


def ensure_separate_directories(*directories: Path) -> None:
    """Reject duplicate or lexically nested checked directory roots."""

    absolute = [Path(os.path.abspath(item)) for item in directories]
    identities: list[tuple[int, int]] = []
    for item in absolute:
        with DirectoryHandle.open(item) as handle:
            identities.append((handle.device, handle.inode))
    for index, left in enumerate(absolute):
        for other_index, right in enumerate(absolute[index + 1 :], start=index + 1):
            if (
                identities[index] == identities[other_index]
                or left == right
                or left in right.parents
                or right in left.parents
            ):
                raise SafetyError("input, output, and reference roots must be separate")


def wav_files(path: Path) -> dict[str, Path]:
    """Return direct-child WAV files while rejecting links and unexpected entries."""

    with DirectoryHandle.open(path) as root:
        found: dict[str, Path] = {}
        try:
            entries = sorted(os.listdir(root.fd))
        except OSError as exc:
            raise SafetyError("audio directory cannot be inspected") from exc
        for name in entries:
            try:
                metadata = os.stat(name, dir_fd=root.fd, follow_symlinks=False)
            except OSError as exc:
                raise SafetyError("audio directory changed during inspection") from exc
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise SafetyError(
                    "audio directories may contain only single-link regular WAV files"
                )
            if Path(name).suffix.lower() != ".wav":
                raise SafetyError("audio directories may contain only WAV files")
            if name in found:
                raise SafetyError("duplicate audio filename")
            found[name] = root.path / name
        return found


@dataclass(frozen=True)
class CreatedFile:
    """Identity token for a direct child created during the current operation."""

    name: str
    device: int
    inode: int


@dataclass
class PreparedJsonReport:
    """Exclusively reserve a report and commit it through the same held descriptor."""

    directory: DirectoryHandle
    name: str
    fd: int
    device: int
    inode: int
    _committed: bool = False
    _closed: bool = False

    @classmethod
    def create(cls, path: Path) -> PreparedJsonReport:
        name = _child_name(path.name, label="report")
        directory = DirectoryHandle.open(path.parent)
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        fd: int | None = None
        try:
            fd = os.open(name, flags, 0o600, dir_fd=directory.fd)
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise SafetyError("report target is not a single-link regular file")
            return cls(directory, name, fd, metadata.st_dev, metadata.st_ino)
        except FileExistsError as exc:
            directory.close()
            raise SafetyError("report already exists; refusing to overwrite") from exc
        except SafetyError:
            if fd is not None:
                os.close(fd)
            directory.close()
            raise
        except OSError as exc:
            if fd is not None:
                os.close(fd)
            directory.close()
            raise SafetyError("report could not be reserved safely") from exc

    def commit(self, payload: Mapping[str, Any]) -> None:
        """Serialize then durably fill the reserved file; never reopen its pathname."""

        try:
            rendered = (
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            os.lseek(self.fd, 0, os.SEEK_SET)
            os.ftruncate(self.fd, 0)
            view = memoryview(rendered)
            while view:
                written = os.write(self.fd, view)
                if written <= 0:
                    raise OSError("short report write")
                view = view[written:]
            os.fsync(self.fd)
            current = os.stat(self.name, dir_fd=self.directory.fd, follow_symlinks=False)
            opened = os.fstat(self.fd)
            if (
                not stat.S_ISREG(current.st_mode)
                or current.st_nlink != 1
                or opened.st_nlink != 1
                or (current.st_dev, current.st_ino) != (self.device, self.inode)
                or (opened.st_dev, opened.st_ino) != (self.device, self.inode)
            ):
                raise OSError("report target changed during commit")
            self._committed = True
        except (OSError, TypeError, ValueError, UnicodeError) as exc:
            raise SafetyError("report could not be written safely") from exc

    def _unlink_if_original(self) -> None:
        try:
            metadata = os.stat(self.name, dir_fd=self.directory.fd, follow_symlinks=False)
            if (
                stat.S_ISREG(metadata.st_mode)
                and metadata.st_dev == self.device
                and metadata.st_ino == self.inode
            ):
                os.unlink(self.name, dir_fd=self.directory.fd)
        except OSError:
            pass

    def close(self) -> None:
        if self._closed:
            return
        try:
            with suppress(OSError):
                os.close(self.fd)
        finally:
            if not self._committed:
                self._unlink_if_original()
            self.directory.close()
            self._closed = True

    def __enter__(self) -> PreparedJsonReport:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def prepare_json_report(path: Path) -> PreparedJsonReport:
    """Reserve a new mode-0600 JSON target before expensive or trusted processing."""

    return PreparedJsonReport.create(path)


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON once through an exclusively reserved, no-follow descriptor."""

    with prepare_json_report(path) as report:
        report.commit(payload)


def remove_created_files(files: Sequence[CreatedFile], directory: DirectoryHandle) -> None:
    """Remove only matching direct children through the original held directory fd."""

    for created in files:
        try:
            metadata = os.stat(created.name, dir_fd=directory.fd, follow_symlinks=False)
            if (
                stat.S_ISREG(metadata.st_mode)
                and metadata.st_dev == created.device
                and metadata.st_ino == created.inode
            ):
                os.unlink(created.name, dir_fd=directory.fd)
        except OSError:
            continue
