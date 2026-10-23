from pathlib import Path
from uuid import UUID
import soundfile as sf
from fastapi import UploadFile

from app.config import Settings

_CHUNK_SIZE = 1024 * 1024  # copy 1 MB at a time instead of loading the whole file


class UploadRejected(Exception):
    """Base: the uploaded file can't be accepted. The message is safe to show the client.

    `code` is sent to clients, so keep it stable even if the class is renamed.
    """

    code = "upload_rejected"


class EmptyFile(UploadRejected):
    code = "empty_file"


class UnsupportedFileType(UploadRejected):
    code = "unsupported_file_type"


class FileTooLarge(UploadRejected):
    code = "file_too_large"


class UnreadableAudio(UploadRejected):
    code = "unreadable_audio"


def file_extension(file: UploadFile) -> str:
    return Path(file.filename or "").suffix.lstrip(".").lower()


def validate_file(file: UploadFile, settings: Settings) -> None:
    """Reject the upload unless it's a non-empty, readable audio file within the size limit.

    Leaves the file rewound to the start, ready to be saved.
    """
    extension = file_extension(file)
    if extension not in settings.allowed_extensions_set:
        allowed = ", ".join(sorted(settings.allowed_extensions_set))
        raise UnsupportedFileType(f"Unsupported file type '.{extension}'. Allowed: {allowed}")

    if not file.size:  # covers 0 and None
        raise EmptyFile("Uploaded file is empty")
    if file.size > settings.max_upload_bytes:
        raise FileTooLarge(f"File is larger than {settings.max_upload_mb} MB")

    # Reads only the header: catches renamed or corrupt files without decoding the audio
    try:
        sf.info(file.file)
    except (RuntimeError, OSError) as e:  # sf.LibsndfileError is a RuntimeError (same as inference)
        raise UnreadableAudio(f"'{file.filename}' is not a readable audio file") from e
    finally:
        file.file.seek(0)


async def save_upload(file: UploadFile, job_id: UUID, settings: Settings) -> Path:
    """Write the upload to the shared uploads folder as <job_id>.<ext> and return its path."""
    # Named after the job id, not the client's filename: unique, and can't escape upload_dir
    path = settings.upload_dir / f"{job_id}.{file_extension(file)}"
    with path.open("wb") as out:
        while chunk := await file.read(_CHUNK_SIZE):
            out.write(chunk)
    return path
