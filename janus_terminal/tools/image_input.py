"""Security-critical image-input validation for the multimodal ``/analyze`` flow.

Users upload report-page images (URL, base64/data-URL, or local path). Every
source is treated as **untrusted**. This module is the single choke point that
turns an :class:`ImageRef` into a vetted :class:`ValidatedImage` ready for an
OpenAI-vision payload, or raises :class:`ImageValidationError` with a *safe*
(non-leaky) message.

Threat model & mitigations
---------------------------
* **SSRF** — :func:`_assert_public_url` runs BEFORE any fetch: enforces an
  https-only scheme allowlist, resolves the host with ``socket.getaddrinfo``
  and rejects EVERY resolved IP that is loopback / private / link-local /
  reserved / multicast / unspecified (incl. the ``169.254.169.254`` cloud
  metadata endpoint and IPv4-mapped IPv6 like ``::ffff:127.0.0.1``), and
  honours an optional exact host allowlist.
* **DNS rebinding / TOCTOU SSRF** — :func:`_default_fetch` performs DNS
  resolution and TCP connection as ONE controlled operation (true IP pinning).
  The host is resolved ONCE; every returned IP is validated; the socket then
  connects DIRECTLY to a validated IP. This eliminates the second resolution
  that a plain ``opener.open(url)`` would trigger — an attacker can no longer
  rebind the domain to a private/metadata IP between the check and the connect.
  TLS still verifies the certificate against the ORIGINAL hostname (SNI +
  ``check_hostname``) by passing ``server_hostname=<host>`` to
  ``ssl_context.wrap_socket``. Redirects remain disabled so a 3xx to a private
  host can never be chased. See :class:`_PinnedHTTPSConnection`.
* **Decompression bombs** — ``max_bytes`` caps the ENCODED size and is enforced
  BEFORE base64 decode / before reading file bytes; ``max_pixels`` caps
  ``width * height`` and is checked immediately after ``Image.open`` reads the
  header, BEFORE ``verify()`` decodes the raster. Pillow's
  ``DecompressionBombError`` is caught and converted to a generic error.
* **Parser attack surface** — the format allowlist (PNG/JPEG/WEBP) is enforced
  immediately after ``Image.open`` and BEFORE ``verify()``, so TIFF/ICO/SVG/
  polyglot inputs are never fully parsed. The true MIME is derived from Pillow,
  never from the caller's claim.
* **Path traversal / symlink TOCTOU** — local paths are opt-in
  (``allow_local_paths``) and must resolve *inside* ``upload_dir``; the file is
  then opened fd-based with ``O_NOFOLLOW`` and size-checked via ``fstat`` so a
  symlink swapped in after the containment check cannot be followed.
* **fetch_fn is TEST-ONLY** — ``load_image(..., fetch_fn=...)`` exists so unit
  tests inject bytes without network I/O. Production MUST leave it ``None`` so
  the hardened :func:`_default_fetch` (IP pinning + no redirects + byte cap)
  is used; a custom ``fetch_fn`` bypasses the SSRF protections.
* **Information leak** — error messages are generic; resolved IPs and absolute
  filesystem paths never reach the caller.
"""
from __future__ import annotations

import base64
import binascii
import ipaddress
import os
import socket
import stat
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable, Literal, Sequence
from urllib.parse import urlsplit

# Map Pillow format names to canonical MIME types (our safe allowlist).
_FORMAT_TO_MIME: dict[str, str] = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
}


# ───────────────────────────── Contract types ─────────────────────────────


@dataclass(frozen=True)
class ImageRef:
    """An untrusted reference to an image supplied by a caller."""

    kind: Literal["url", "base64", "path"]
    value: str  # url / base64 (possibly a data: URL) / filesystem path
    mime: str | None = None  # caller's CLAIMED mime — never trusted


@dataclass(frozen=True)
class ValidatedImage:
    """A vetted image, normalized for an OpenAI-vision payload."""

    data_url: str  # "data:<mime>;base64,<...>"
    mime: str  # true mime derived from Pillow, e.g. "image/png"
    width: int
    height: int
    num_bytes: int  # decoded/raw byte length


@dataclass(frozen=True)
class ImagePolicy:
    """Validation policy with safe-by-default settings."""

    max_images: int = 4
    max_bytes: int = 8_000_000  # per image
    max_pixels: int = 24_000_000  # width*height guard (decompression bomb)
    allowed_schemes: frozenset[str] = frozenset({"https"})  # http OFF by default
    allow_local_paths: bool = False  # SAFE DEFAULT — paths opt-in only
    upload_dir: Path | None = None  # local paths must resolve inside this dir
    url_allowlist: tuple[str, ...] = ()  # exact host allowlist; empty = any PUBLIC host
    fetch_timeout: float = 5.0

    @classmethod
    def from_env(cls) -> "ImagePolicy":
        """Build a policy from ``VISION_*`` environment variables.

        Recognised vars (all optional; fall back to class defaults):
        ``VISION_MAX_IMAGES``, ``VISION_MAX_BYTES``, ``VISION_MAX_PIXELS``,
        ``VISION_ALLOW_LOCAL_PATHS``, ``VISION_UPLOAD_DIR``,
        ``VISION_URL_ALLOWLIST`` (comma-separated hosts), ``VISION_FETCH_TIMEOUT``.
        """
        defaults = cls()

        def _int(name: str, fallback: int) -> int:
            raw = os.environ.get(name)
            if raw is None or raw.strip() == "":
                return fallback
            try:
                return int(raw)
            except ValueError:
                return fallback

        def _float(name: str, fallback: float) -> float:
            raw = os.environ.get(name)
            if raw is None or raw.strip() == "":
                return fallback
            try:
                return float(raw)
            except ValueError:
                return fallback

        def _bool(name: str, fallback: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None or raw.strip() == "":
                return fallback
            return raw.strip().lower() in {"1", "true", "yes", "on"}

        upload_raw = os.environ.get("VISION_UPLOAD_DIR")
        upload_dir = Path(upload_raw) if upload_raw and upload_raw.strip() else defaults.upload_dir

        allowlist_raw = os.environ.get("VISION_URL_ALLOWLIST", "")
        allowlist = tuple(
            host.strip().lower() for host in allowlist_raw.split(",") if host.strip()
        )

        return cls(
            max_images=_int("VISION_MAX_IMAGES", defaults.max_images),
            max_bytes=_int("VISION_MAX_BYTES", defaults.max_bytes),
            max_pixels=_int("VISION_MAX_PIXELS", defaults.max_pixels),
            allowed_schemes=defaults.allowed_schemes,
            allow_local_paths=_bool("VISION_ALLOW_LOCAL_PATHS", defaults.allow_local_paths),
            upload_dir=upload_dir,
            url_allowlist=allowlist or defaults.url_allowlist,
            fetch_timeout=_float("VISION_FETCH_TIMEOUT", defaults.fetch_timeout),
        )


class ImageValidationError(Exception):
    """Raised on any validation failure.

    Messages are intentionally generic so internal IPs / filesystem paths /
    fetch internals never reach the (untrusted) caller.
    """


# ───────────────────────────── SSRF guard ─────────────────────────────


def _ip_is_disallowed(ip: ipaddress._BaseAddress) -> bool:
    """Return True if ``ip`` is anything other than a routable PUBLIC address.

    Handles IPv4-mapped IPv6 (``::ffff:a.b.c.d``) by unwrapping to the embedded
    v4 address before the private/loopback checks. Covers loopback, RFC1918
    private (10/8, 172.16/12, 192.168/16), link-local (169.254/16 incl. the
    169.254.169.254 metadata IP, fe80::/10), unique-local (fc00::/7),
    multicast, reserved, and unspecified (0.0.0.0, ::).
    """
    # Unwrap IPv4-mapped IPv6 so e.g. ::ffff:127.0.0.1 is judged as 127.0.0.1.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped

    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _parse_host_and_port(url: str, policy: ImagePolicy) -> tuple[str, int]:
    """Validate scheme + allowlist and return ``(host, port)`` (lowercased host)."""
    try:
        parts = urlsplit(url)
    except ValueError as exc:  # malformed URL
        raise ImageValidationError("invalid image source") from exc

    scheme = (parts.scheme or "").lower()
    if scheme not in policy.allowed_schemes:
        raise ImageValidationError("disallowed URL scheme")

    host = parts.hostname  # urlsplit strips [] from IPv6 literals; never includes userinfo
    if not host:
        raise ImageValidationError("invalid image source")
    # Drop a trailing dot (FQDN root) so "host." == "host" for the allowlist
    # and so it cannot be used to dodge an exact-match allowlist.
    host = host.rstrip(".").lower()
    if not host:
        raise ImageValidationError("invalid image source")

    # urlsplit raises on a bad port; treat that as a generic invalid source.
    try:
        port = parts.port
    except ValueError as exc:
        raise ImageValidationError("invalid image source") from exc
    port = port if port is not None else 443

    # Optional exact host allowlist (case-insensitive).
    if policy.url_allowlist:
        allowed = {h.rstrip(".").lower() for h in policy.url_allowlist}
        if host not in allowed:
            raise ImageValidationError("disallowed URL host")

    return host, port


def _resolve_and_validate(host: str, port: int) -> list[tuple[int, str]]:
    """Resolve ``host`` ONCE and validate every IP; return ``[(family, ip)]``.

    Rejects (without leaking the address) if any resolved IP is loopback /
    private / link-local / reserved / multicast / unspecified, or if any
    numeric form (decimal, hex, octal, IPv4-mapped IPv6) decodes to such.
    Used by both the pre-fetch guard and the pinned fetcher so the validated
    set and the connect target are derived from the SAME resolution.
    """
    results: list[tuple[int, str]] = []

    # If the host is already an IP literal, judge it directly (no DNS needed).
    # ``ip_address`` accepts decimal/hex/octal-packed integers too when fed via
    # socket; we normalise through getaddrinfo below for non-literal forms.
    try:
        literal = ipaddress.ip_address(host)
        family = socket.AF_INET6 if literal.version == 6 else socket.AF_INET
        if _ip_is_disallowed(literal):
            raise ImageValidationError("disallowed URL host")
        return [(family, str(literal))]
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ImageValidationError("disallowed URL host") from exc

    seen: set[str] = set()
    for info in infos:
        family = info[0]
        addr = info[4][0]
        if addr in seen:
            continue
        seen.add(addr)
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError as exc:
            raise ImageValidationError("disallowed URL host") from exc
        # Reject if ANY resolved IP is non-public (mixed-record poisoning guard).
        if _ip_is_disallowed(ip):
            raise ImageValidationError("disallowed URL host")
        results.append((family, addr))

    if not results:
        raise ImageValidationError("disallowed URL host")
    return results


def _assert_public_url(url: str, policy: ImagePolicy) -> None:
    """Validate ``url`` is safe to fetch, raising ImageValidationError otherwise.

    Runs BEFORE any network I/O. Performs (in order): scheme allowlist check,
    host extraction, optional exact host allowlist check, DNS resolution via
    ``socket.getaddrinfo``, and per-IP private/loopback/link-local rejection.
    """
    host, port = _parse_host_and_port(url, policy)
    _resolve_and_validate(host, port)


# ───────────────────────────── Default fetcher ─────────────────────────────


def _build_pinned_connection_cls(pinned_ip: str) -> type:
    """Return an ``HTTPSConnection`` subclass that connects to ``pinned_ip``.

    The connection is opened to the validated IP directly (no second DNS
    lookup), but TLS still uses the original hostname for SNI and certificate
    verification because :meth:`connect` passes ``server_hostname=self.host``
    (the original host) to ``wrap_socket``. This is what makes IP pinning safe:
    we connect to a vetted address yet verify the cert against the name.
    """
    import http.client

    class _PinnedHTTPSConnection(http.client.HTTPSConnection):
        def connect(self) -> None:  # type: ignore[override]
            # Open the raw socket to the PINNED validated IP, not self.host.
            sock = socket.create_connection(
                (pinned_ip, self.port), timeout=self.timeout
            )
            if getattr(self, "_tunnel_host", None):  # pragma: no cover - no proxy use
                self.sock = sock
                self._tunnel()
            # Wrap with TLS using the ORIGINAL hostname for SNI + cert check.
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)

    return _PinnedHTTPSConnection


def _default_fetch(url: str, policy: ImagePolicy) -> bytes:
    """Fetch ``url`` with IP pinning, a hard size cap, and NO redirects.

    Resolution and connection are a SINGLE controlled operation: the host is
    resolved once, every IP validated, and the socket connects directly to a
    validated IP (see :func:`_build_pinned_connection_cls`). This closes the
    DNS-rebinding / TOCTOU window that a plain ``urllib.open(url)`` leaves open
    by re-resolving the hostname at connect time. Redirects are disabled (a 3xx
    is surfaced as an error, never chased to a private host) and the body read
    is capped at ``policy.max_bytes + 1`` to bound memory.
    """
    import ssl

    host, port = _parse_host_and_port(url, policy)
    validated = _resolve_and_validate(host, port)
    pinned_family, pinned_ip = validated[0]

    parts = urlsplit(url)
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"

    context = ssl.create_default_context()
    conn_cls = _build_pinned_connection_cls(pinned_ip)
    cap = policy.max_bytes + 1
    conn = conn_cls(host, port=port, timeout=policy.fetch_timeout, context=context)
    try:
        # Host header carries the original hostname (+ non-default port).
        host_header = host if port == 443 else f"{host}:{port}"
        conn.request("GET", path, headers={"Host": host_header})
        resp = conn.getresponse()
        status = resp.status
        if status in (301, 302, 303, 307, 308) or 300 <= status < 400:
            # Refuse redirects outright; re-validating each hop is fragile
            # against rebinding, so we never chase a 3xx.
            raise ImageValidationError("disallowed URL host")
        if status != 200:
            raise ImageValidationError("invalid image source")
        data = resp.read(cap)
    except ImageValidationError:
        raise
    except (OSError, ssl.SSLError, ValueError) as exc:
        # Generic message; never leak the underlying network error / IP.
        raise ImageValidationError("invalid image source") from exc
    finally:
        conn.close()

    if len(data) > policy.max_bytes:
        raise ImageValidationError("image too large")
    return data


# ───────────────────────────── Byte loaders ─────────────────────────────


def _decode_base64(value: str, policy: ImagePolicy) -> bytes:
    """Decode raw base64 or a ``data:<mime>;base64,<...>`` URL into bytes.

    The ENCODED length is bounded BEFORE decoding so a malicious payload cannot
    force a large allocation: base64 expands 3 input bytes into 4 chars, so the
    longest legal encoding of ``max_bytes`` decoded bytes is
    ``4 * ceil(max_bytes / 3)`` chars (plus a little slack for padding/newlines).
    """
    payload = value.strip()
    if payload.lower().startswith("data:"):
        # data:<mediatype>;base64,<data>
        header, _, b64 = payload.partition(",")
        if "base64" not in header.lower() or not b64:
            raise ImageValidationError("invalid image source")
        payload = b64.strip()

    # Encoded-length bound (pre-decode memory-DoS guard).
    max_encoded = 4 * ((policy.max_bytes + 2) // 3) + 4
    if len(payload) > max_encoded:
        raise ImageValidationError("image too large")

    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageValidationError("invalid image source") from exc
    if not data:
        raise ImageValidationError("invalid image source")
    if len(data) > policy.max_bytes:
        raise ImageValidationError("image too large")
    return data


def _candidate_components(value: str, anchors: tuple[Path, ...]) -> list[str]:
    """Return the path components of ``value`` RELATIVE to one of ``anchors``.

    Rejects (with a generic error) any input that is not a safe, strictly
    descending relative path: ``..`` traversal, empty/``.`` components, or an
    absolute path not anchored under any acceptable upload-dir spelling.

    ``anchors`` holds the acceptable upload-dir prefixes — both the configured
    path and its symlink-resolved form — because on some platforms (e.g. macOS
    ``/var`` → ``/private/var``) the caller's absolute path and the resolved
    base differ only by a leading OS symlink. We strip whichever prefix matches,
    then walk the remaining components ourselves via ``dir_fd`` (never trusting
    ``resolve()`` to have collapsed an attacker-controlled symlink).
    """
    raw = Path(value)
    if raw.is_absolute():
        rel: Path | None = None
        for anchor in anchors:
            try:
                rel = raw.relative_to(anchor)
                break
            except ValueError:
                continue
        if rel is None:
            raise ImageValidationError("invalid local image path")
    else:
        rel = raw

    parts = rel.parts
    if not parts:
        raise ImageValidationError("invalid local image path")
    for comp in parts:
        # Reject traversal / empty / current-dir / any separator-bearing comp.
        if comp in ("", ".", "..") or os.sep in comp or (os.altsep and os.altsep in comp):
            raise ImageValidationError("invalid local image path")
    return list(parts)


def _read_local_path(value: str, policy: ImagePolicy) -> bytes:
    """Read a local file via an openat-style walk anchored at ``upload_dir``.

    We NEVER trust ``Path.resolve()`` for the final open: instead we open the
    trusted ``upload_dir`` once, then descend one component at a time with
    ``dir_fd=`` and ``O_NOFOLLOW | O_DIRECTORY`` on every intermediate hop. This
    refuses an intermediate directory that is — or is swapped to be — a symlink
    (closing the TOCTOU window that ``O_NOFOLLOW`` on the final component alone
    leaves open). The final component is opened with ``O_NOFOLLOW``, fstat'd to
    confirm a regular file within ``max_bytes``, and read with a hard cap.
    """
    if not policy.allow_local_paths:
        raise ImageValidationError("local paths are disallowed")
    if policy.upload_dir is None:
        raise ImageValidationError("local paths are disallowed")

    try:
        base = policy.upload_dir.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ImageValidationError("invalid local image path") from exc

    # Accept the caller's absolute path whether it is spelled with the
    # configured upload_dir or its symlink-resolved form (e.g. macOS
    # /var → /private/var). The openat walk below still anchors on ``base``.
    anchors = (base, policy.upload_dir)
    components = _candidate_components(value, anchors)

    dir_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_DIRECTORY", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)

    open_fds: list[int] = []
    try:
        # 1) Open the trusted upload dir itself (refuse if IT is a symlink).
        try:
            base_fd = os.open(str(base), dir_flags)
        except OSError as exc:
            raise ImageValidationError("invalid local image path") from exc
        open_fds.append(base_fd)
        parent_fd = base_fd

        # 2-4) Walk each component with dir_fd; intermediates as directories,
        #       the last as the file. O_NOFOLLOW on every hop refuses symlinks.
        *intermediate, leaf = components
        for comp in intermediate:
            try:
                next_fd = os.open(comp, dir_flags, dir_fd=parent_fd)
            except OSError as exc:
                raise ImageValidationError("invalid local image path") from exc
            open_fds.append(next_fd)
            parent_fd = next_fd

        try:
            file_fd = os.open(leaf, file_flags, dir_fd=parent_fd)
        except OSError as exc:
            raise ImageValidationError("invalid local image path") from exc
        open_fds.append(file_fd)

        # 5) Validate via fstat on the open fd; read with a hard cap.
        try:
            st = os.fstat(file_fd)
            if not stat.S_ISREG(st.st_mode):
                raise ImageValidationError("invalid local image path")
            if st.st_size > policy.max_bytes:
                raise ImageValidationError("image too large")
            data = os.read(file_fd, policy.max_bytes + 1)
        except ImageValidationError:
            raise
        except OSError as exc:
            raise ImageValidationError("invalid local image path") from exc
    finally:
        # 6) Close all fds (deepest first is fine; order irrelevant for close).
        for fd in reversed(open_fds):
            try:
                os.close(fd)
            except OSError:  # pragma: no cover - best-effort cleanup
                pass

    if len(data) > policy.max_bytes:
        raise ImageValidationError("image too large")
    return data


# ───────────────────────────── Content validation ─────────────────────────────


def _validate_image_bytes(data: bytes, policy: ImagePolicy) -> ValidatedImage:
    """Confirm ``data`` is a real, allowed, non-bomb image; return ValidatedImage.

    Order matters for attack-surface reduction:
      1. open ONCE and read only the header (format + size) — cheap, no raster;
      2. reject formats outside {PNG,JPEG,WEBP} BEFORE any decode, so TIFF/ICO/
         SVG/polyglot parsers are never fully exercised;
      3. reject ``width*height > max_pixels`` immediately (decompression bomb)
         BEFORE ``verify()`` decodes pixels;
      4. only THEN ``verify()`` the accepted, bounded image.
    Pillow's ``DecompressionBombError`` (a plain ``Exception`` subclass) is
    caught explicitly so it cannot escape as a non-``ImageValidationError``.
    """
    from PIL import Image, UnidentifiedImageError

    if len(data) > policy.max_bytes:
        raise ImageValidationError("image too large")

    # 1) Open once; read header-level metadata only.
    try:
        with Image.open(BytesIO(data)) as img:
            fmt = (img.format or "").upper()
            width, height = img.size
    except Image.DecompressionBombError as exc:
        raise ImageValidationError("image resolution too large") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageValidationError("invalid image") from exc

    # 2) Format allowlist BEFORE any pixel decode.
    mime = _FORMAT_TO_MIME.get(fmt)
    if mime is None:
        raise ImageValidationError("unsupported image format")

    if width <= 0 or height <= 0:
        raise ImageValidationError("invalid image")

    # 3) Pixel-bomb guard BEFORE verify() touches the raster.
    if width * height > policy.max_pixels:
        raise ImageValidationError("image resolution too large")

    # 4) verify() the accepted, size-bounded image (consumes the stream, so we
    #    re-open from the original bytes).
    try:
        with Image.open(BytesIO(data)) as probe:
            probe.verify()
    except Image.DecompressionBombError as exc:
        raise ImageValidationError("image resolution too large") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageValidationError("invalid image") from exc

    b64 = base64.b64encode(data).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    return ValidatedImage(
        data_url=data_url,
        mime=mime,
        width=width,
        height=height,
        num_bytes=len(data),
    )


# ───────────────────────────── Public API ─────────────────────────────


def load_image(
    ref: ImageRef,
    policy: ImagePolicy,
    *,
    fetch_fn: Callable[[str, ImagePolicy], bytes] | None = None,
) -> ValidatedImage:
    """Validate a single :class:`ImageRef` into a :class:`ValidatedImage`.

    ``fetch_fn`` is **TEST-ONLY**: it lets unit tests inject image bytes without
    network I/O. Production code MUST leave it ``None`` so the hardened
    :func:`_default_fetch` is used (IP pinning to a validated address, no
    redirect following, and a hard byte cap). Supplying a custom ``fetch_fn``
    bypasses these SSRF protections — only do so in tests. The pre-fetch
    :func:`_assert_public_url` guard still runs for ``kind="url"`` regardless.
    """
    if ref.kind == "url":
        _assert_public_url(ref.value, policy)
        fetcher = fetch_fn or _default_fetch
        data = fetcher(ref.value, policy)
    elif ref.kind == "base64":
        data = _decode_base64(ref.value, policy)
    elif ref.kind == "path":
        data = _read_local_path(ref.value, policy)
    else:  # pragma: no cover - guarded by Literal typing
        raise ImageValidationError("invalid image source")

    return _validate_image_bytes(data, policy)


def load_images(
    refs: Sequence[ImageRef],
    policy: ImagePolicy,
    *,
    fetch_fn: Callable[[str, ImagePolicy], bytes] | None = None,
) -> list[ValidatedImage]:
    """Validate a batch of refs, enforcing ``policy.max_images`` and order."""
    if len(refs) > policy.max_images:
        raise ImageValidationError("too many images")
    return [load_image(ref, policy, fetch_fn=fetch_fn) for ref in refs]
