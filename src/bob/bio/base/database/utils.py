import bz2
import glob
import hashlib
import io
import logging
import tarfile
import warnings
import zipfile

from fnmatch import fnmatch
from os import PathLike
from pathlib import Path
from typing import IO, Any, Callable, TextIO, Union

import requests

from clapper.rc import UserDefaults

logger = logging.getLogger(__name__)


def _get_local_data_directory() -> Path:
    user_config = UserDefaults("bobrc.toml")
    return Path(
        user_config.get("bob_data_dir", default=Path.home() / "bob_data")
    )


def _path_and_subdir(
    archive_path: Union[str, PathLike[str]],
) -> tuple[Path, Union[Path, None]]:
    """Splits an archive's path from a sub directory (separated by ``:``)."""
    archive_path_str = Path(archive_path).as_posix()
    if ":" in archive_path_str:
        archive, sub_dir = archive_path_str.rsplit(":", 1)
        return Path(archive), Path(sub_dir)
    return Path(archive_path), None


def _is_bz2(path: Union[str, PathLike[str]]) -> bool:
    try:
        with bz2.BZ2File(path) as f:
            f.read(1024)
        return True
    except (OSError, EOFError):
        return False


def is_archive(path: Union[str, PathLike[str]]) -> bool:
    """Returns whether the path points in an archive.

    Any path pointing to a valid tar or zip archive or to a valid bz2
    file will return ``True``.
    """
    archive = _path_and_subdir(path)[0]
    try:
        return any(
            tester(_path_and_subdir(archive)[0])
            for tester in (tarfile.is_tarfile, zipfile.is_zipfile, _is_bz2)
        )
    except (FileNotFoundError, IsADirectoryError):
        return False


def search_in_archive_and_open(
    search_pattern: str,
    archive_path: Union[str, PathLike[str]],
    inner_dir: Union[str, PathLike[str], None] = None,
    open_as_binary: bool = False,
) -> Union[IO[bytes], TextIO, None]:
    """Returns a read-only stream of a file matching a pattern in an archive.

    Wildcards (``*``, ``?``, and ``**``) are supported (using
    :meth:`pathlib.Path.glob`).

    The first matching file will be open and returned.

    examples:

    .. code-block: text

        archive.tar.gz
            + subdir1
            |   + file1.txt
            |   + file2.txt
            |
            + subdir2
                + file1.txt

    ``search_and_open("archive.tar.gz", "file1.txt")``
    opens``archive.tar.gz/subdir1/file1.txt``

    ``search_and_open("archive.tar.gz:subdir2", "file1.txt")``
    opens ``archive.tar.gz/subdir2/file1.txt``

    ``search_and_open("archive.tar.gz", "*.txt")``
    opens ``archive.tar.gz/subdir1/file1.txt``


    Parameters
    ----------
    archive_path
        The ``.tar.gz`` archive file containing the wanted file. To match
        ``search_pattern`` in a sub path in that archive, append the sub path
        to ``archive_path`` with a ``:`` (e.g.
        ``/path/to/archive.tar.gz:sub/dir/``).
    search_pattern
        A string to match to the file. Wildcards are supported (Unix pattern
        matching).

    Returns
    -------
    io.TextIOBase or io.BytesIO
        A read-only file stream.
    """

    archive_path = Path(archive_path)

    if inner_dir is None:
        archive_path, inner_dir = _path_and_subdir(archive_path)

    if inner_dir is not None:
        pattern = (Path("/") / inner_dir / search_pattern).as_posix()
    else:
        pattern = (Path("/") / search_pattern).as_posix()

    if ".tar" in archive_path.suffixes:
        tar_arch = tarfile.open(archive_path)  # TODO File not closed
        for member in tar_arch:
            if member.isfile() and fnmatch("/" + member.name, pattern):
                break
        else:
            logger.debug(
                f"No file matching '{pattern}' were found in '{archive_path}'."
            )
            return None

        if open_as_binary:
            return tar_arch.extractfile(member)
        return io.TextIOWrapper(tar_arch.extractfile(member), encoding="utf-8")

    elif archive_path.suffix == ".zip":
        zip_arch = zipfile.ZipFile(archive_path)
        for name in zip_arch.namelist():
            if fnmatch("/" + name, pattern):
                break
        else:
            logger.debug(
                f"No file matching '{pattern}' were found in '{archive_path}'."
            )
        return zip_arch.open(name)

    raise ValueError(
        f"Unknown file extension '{''.join(archive_path.suffixes)}'"
    )


def list_dir_in_archive(
    archive_path: Union[str, PathLike[str]],
    inner_dir: Union[str, PathLike[str], None] = None,
    show_dirs: bool = True,
    show_files: bool = True,
) -> list[Path]:
    """Returns a list of all the elements in an archive or inner directory.

    Parameters
    ----------
    archive_path
        A path to an archive, or an inner directory of an archive (appended
        with a ``:``).
    inner_dir
        A path inside the archive with its root at the archive's root.
    show_dirs
        Returns directories.
    show_files
        Returns files.
    """

    archive_path, arch_inner_dir = _path_and_subdir(archive_path)
    inner_dir = Path(inner_dir or arch_inner_dir or Path("."))

    results = []
    # Read the archive info and iterate over the paths. Return the ones we want.
    if ".tar" in archive_path.suffixes:
        with tarfile.open(archive_path) as arch:
            for info in arch.getmembers():
                path = Path(info.name)
                if path.parent != inner_dir:
                    continue
                if info.isdir() and show_dirs:
                    results.append(Path("/") / path)
                if info.isfile() and show_files:
                    results.append(Path("/") / path)
    elif archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as arch:
            for zip_info in arch.infolist():
                zip_path = zipfile.Path(archive_path, zip_info.filename)
                if Path(zip_info.filename).parent != inner_dir:
                    continue
                if zip_path.is_dir() and show_dirs:
                    results.append(Path("/") / zip_info.filename)
                if not zip_path.is_dir() and show_files:
                    results.append(Path("/") / zip_info.filename)
    elif archive_path.suffix == ".bz2":
        if inner_dir != Path("."):
            raise ValueError(
                ".bz2 files don't have an inner structure (tried to access "
                f"'{archive_path}:{inner_dir}')."
            )
        results.extend([Path(archive_path.stem)] if show_files else [])
    else:
        raise ValueError(
            f"Unsupported archive extension '{''.join(archive_path.suffixes)}'."
        )
    return sorted(results)  # Fixes inconsistent file ordering across platforms


def extract_archive(
    archive_path: Union[str, PathLike[str]],
    inner_path: Union[str, PathLike[str], None] = None,
    destination: Union[str, PathLike[str], None] = None,
) -> Path:
    """Extract an archive and returns the location of the extracted data.

    Supports ``.zip``, ``.tar.gz``, ``.tar.bz2``, ``.tar.tgz``, and
    ``.tar.tbz2`` archives.
    Can also extract ``.bz2`` compressed files.

    Parameters
    ----------
    archive_path
        The compressed archive location. Pointing to a location inside a
        tarball can be achieved by appending ``:`` and the desired member to
        extract.
    inner_path
        A path with its root at the root of the archive file pointing to a
        specific file to extract.
    destination
        The desired location of the extracted file or directory. If not
        provided, the archive will be extracted where it stands (the parent of
        ``archive_path``).

    Returns
    -------
    pathlib.Path
        The extracted file or directory location.
        As an archive can contain any number of members, the parent directory
        is returned (where the archive content is extracted).

    Raises
    ------
    ValueError
        When ``archive_path`` does not point to a file with a known extension.
    """

    archive_path, arch_inner_dir = _path_and_subdir(archive_path)
    sub_dir = inner_path or arch_inner_dir

    if destination is None:
        destination = archive_path.parent

    if ".tar" in archive_path.suffixes:
        with tarfile.open(archive_path, mode="r") as arch:
            if sub_dir is None:
                arch.extractall(destination)
            else:
                arch.extract(Path(sub_dir).as_posix(), destination)
    elif ".zip" == archive_path.suffix:
        with zipfile.ZipFile(archive_path) as arch:
            if sub_dir is None:
                arch.extractall(destination)
            else:
                arch.extract(Path(sub_dir).as_posix(), destination)
    elif ".bz2" == archive_path.suffix:
        if sub_dir is not None:
            warnings.warn(
                f"Ignored sub directory ({sub_dir}). Not supported for `.bz2` files.",
                RuntimeWarning,
            )
        extracted_file = destination / Path(archive_path.stem)
        with bz2.BZ2File(archive_path) as arch, extracted_file.open(
            "wb"
        ) as dest:
            dest.write(arch.read())
    else:
        raise ValueError(
            f"Unknown file extension: {''.join(archive_path.suffixes)}"
        )
    return Path(destination)


def search_and_open(
    search_pattern: str,
    base_dir: Union[PathLike, None] = None,
    sub_dir: Union[PathLike, None] = None,
    open_as_binary: bool = False,
    **kwargs,
) -> Union[IO[bytes], TextIO, None]:
    """Searches for a matching file recursively in a directory.

    If ``base_dir`` points to an archive, the pattern will be searched inside that
    archive.

    Wildcards (``*``, ``?``, and ``**``) are supported (using
    :meth:`pathlib.Path.glob`).

    Parameters
    ----------
    search_pattern
        A string containing the wanted path pattern of the file to open.
        Supports ``fnmatch`` notation (``*``, ``**``, and ``?``).
    base_dir
        A path to a directory to search into. By default, will use the
        ``data_path`` user configuration.
    sub_dir
        A sub directory of ``base_dir`` to search into instead. Useful when
        using the default value of ``base_dir`` but still wanting to use a
        sub directory in there.
    open_as_binary
        Will open the file as a binary stream instead of a text file.

    Returns
    -------
    IO
        A read-only open file stream.
    """

    if base_dir is None:
        base_dir = _get_local_data_directory()

    if is_archive(base_dir):
        return search_in_archive_and_open(
            search_pattern=search_pattern,
            archive_path=base_dir,
            inner_dir=sub_dir,  # TODO not ok with config data_path / subdir
            **kwargs,
        )

    # If the input is local
    base_dir = Path(base_dir)
    final_dir = base_dir / sub_dir if sub_dir else base_dir
    if final_dir.is_dir():
        # we prepend './' to search_pattern because it might start with '/'
        pattern = final_dir / "**" / f"./{search_pattern}"
        for path in glob.iglob(pattern.as_posix(), recursive=True):
            if not Path(path).is_file():
                continue
            return open(path, mode="rb" if open_as_binary else "rt")
        raise FileNotFoundError(
            f"Unable to locate and open a file that matches '{pattern}' in "
            f"'{final_dir}'."
        )

    return open(final_dir, mode="rb" if open_as_binary else "rt")


def list_dir(
    base_directory: PathLike,
    sub_directory: Union[PathLike, None] = None,
    show_files: bool = True,
    show_dirs: bool = True,
) -> list[Path]:
    """Lists all directories and/or files in a directory (non-recursively)."""
    base_directory = Path(base_directory)

    if is_archive(base_directory):
        return list_dir_in_archive(
            archive_path=base_directory,
            inner_dir=sub_directory,
            show_dirs=show_dirs,
            show_files=show_files,
        )

    # Not an archive
    final_directory = (
        base_directory
        if sub_directory is None
        else base_directory / sub_directory
    )
    glob = list(final_directory.glob("*"))
    if not show_dirs:
        glob = [g for g in glob if not g.is_dir()]
    if not show_files:
        glob = [g for g in glob if not g.is_file()]
    return glob


def md5_hash(readable: Any, chunk_size: int = 65535) -> str:
    """Computes the md5 hash of any object with a read method."""
    hasher = hashlib.md5()
    for chunk in iter(lambda: readable.read(chunk_size), b""):
        hasher.update(chunk)
    return hasher.hexdigest()


def sha256_hash(readable: Any, chunk_size: int = 65535) -> str:
    """Computes the SHA256 hash of any object with a read method."""
    hasher = hashlib.sha256()
    for chunk in iter(lambda: readable.read(chunk_size), b""):
        hasher.update(chunk)
    return hasher.hexdigest()


def verify_file(
    file_path: Union[str, PathLike[str]],
    file_hash: str,
    hash_fct: Callable[[Any, int], str] = sha256_hash,
    full_match: bool = False,
) -> bool:
    """Returns True if the file computed hash corresponds to `file_hash`.

    For comfort, we allow ``file_hash`` to match with the first
    characters of the digest, allowing storing only e.g. the first 8
    char.

    Parameters
    ----------
    file_path
        The path to the file needing verification.
    file_hash
        The expected file hash digest.
    hash_fct
        A function taking a path and returning a digest. Defaults to SHA256.
    full_match
        If set to False, allows ``file_hash`` to match the first characters of
        the files digest (this allows storing e.g. 8 chars of a digest instead
        of the whole 64 characters of SHA256, and still matching.)
    """
    file_path = Path(file_path)
    with file_path.open("rb") as f:
        digest = hash_fct(f, 65535)
    return digest == file_hash if full_match else digest.startswith(file_hash)


def compute_crc(
    file_path: Union[str, PathLike[str]],
    hash_fct: Callable[[Any, int], str] = sha256_hash,
) -> str:
    """Returns the CRC of a file."""
    file_path = Path(file_path)
    with file_path.open("rb") as f:
        return hash_fct(f, 65535)


def _infer_filename_from_urls(urls=Union[list[str], str]) -> str:
    """Retrieves the remote filename from the URLs.

    Parameters
    ----------
    urls
        One or multiple URLs pointing to files with the same name.

    Returns
    -------
    The remote file name.

    Raises
    ------
    ValueError
        When urls point to files with different names.
    """
    if isinstance(urls, str):
        return urls.split("/")[-1]

    # Check that all urls point to the same file name
    names = [u.split("/")[-1] for u in urls]
    if not all(n == names[0] for n in names):
        raise ValueError(
            f"Cannot infer file name when urls point to different files ({names=})."
        )
    return urls[0].split("/")[-1]


def download_file(
    urls: Union[list[str], str],
    destination_directory: Union[str, PathLike[str], None] = None,
    destination_sub_directory: Union[str, None] = None,
    destination_filename: Union[str, None] = None,
    checksum: Union[str, None] = None,
    checksum_fct: Callable[[Any, int], str] = sha256_hash,
    force: bool = True,
    extract: bool = False,
    makedirs: bool = True,
) -> Path:
    """Downloads a remote file locally.

    This will overwrite any existing file with the same name.

    Parameters
    ----------
    urls
        The remote location of the server. If multiple addresses are given, we will try
        to download from them in order until one succeeds.
    destination_directory
        A path to a local directory where the file will be saved. If omitted, the file
        will be saved in the folder pointed by the ``wdr.local_directory`` key in the
        user configuration.
    destination_sub_directory
        An additional layer added to the destination directory (useful when using
        ``destination_directory=None``).
    destination_filename
        The final name of the local file. If omitted, the file will keep the name of
        the remote file.
    checksum
        When provided, will compute the file's checksum and compare to this.
    force
        Re-download and overwrite any existing file with the same name.
    extract
        Extract an archive or zip file next to the downloaded file.
        If this is set, the parent directory path will be returned.
    makedirs
        Automatically make the parent directories of the new local file.

    Returns
    -------
    The path to the new local file (or the parent directory if ``extract`` is True).

    Raises
    ------
    RuntimeError
        When the URLs provided are all invalid.
    ValueError
        When ``destination_filename`` is omitted and URLs point to files with different
        names.
        When the checksum of the file does not correspond to the provided ``checksum``.
    """

    if destination_filename is None:
        destination_filename = _infer_filename_from_urls(urls=urls)

    if destination_directory is None:
        destination_directory = _get_local_data_directory()

    destination_directory = Path(destination_directory)

    if destination_sub_directory is not None:
        destination_directory = (
            destination_directory / destination_sub_directory
        )

    local_file = destination_directory / destination_filename
    needs_download = True

    if not force and local_file.is_file():
        logger.info(
            f"File {local_file} already exists, skipping download ({force=})."
        )
        needs_download = False

    if needs_download:
        if isinstance(urls, str):
            urls = [urls]

        for tries, url in enumerate(urls):
            logger.debug(f"Retrieving file from '{url}'.")
            try:
                response = requests.get(url=url, timeout=10)
            except requests.exceptions.ConnectionError as e:
                if tries < len(urls) - 1:
                    logger.info(
                        f"Could not connect to {url}. Trying other URLs."
                    )
                logger.debug(e)
                continue

            logger.debug(
                f"http response: '{response.status_code}: {response.reason}'."
            )

            if response.ok:
                logger.debug(f"Got file from {url}.")
                break
            elif tries < len(urls) - 1:
                logger.info(
                    f"Failed to get file from {url}, trying other URLs."
                )
                logger.debug(f"requests.response was\n{response}")
        else:
            raise RuntimeError(
                f"Could not retrieve file from any of the provided URLs! ({urls=})"
            )

        if makedirs:
            local_file.parent.mkdir(parents=True, exist_ok=True)

        with local_file.open("wb") as f:
            f.write(response.content)

    if checksum is not None:
        if not verify_file(local_file, checksum, hash_fct=checksum_fct):
            if not needs_download:
                raise ValueError(
                    f"The local file hash does not correspond to '{checksum}' "
                    f"and {force=} prevents overwriting."
                )
            raise ValueError(
                "The downloaded file hash ('"
                f"{compute_crc(local_file, hash_fct=checksum_fct)}') does not "
                f"correspond to '{checksum}'."
            )

    if extract:
        local_file = extract_archive(local_file)

    return local_file
