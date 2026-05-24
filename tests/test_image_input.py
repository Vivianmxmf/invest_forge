"""Tests for invest_forge/tools/image_input.py — all offline / network-free.

TDD: these tests were written FIRST; the implementation follows.

All tests are marked ``@pytest.mark.unit``.
"""
from __future__ import annotations

import base64
import socket
from io import BytesIO
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Helpers — generate real tiny images in-memory so tests never touch disk or
# the network.
# ---------------------------------------------------------------------------


def _make_tiny_png(width: int = 2, height: int = 2) -> bytes:
    """Return raw bytes of a valid minimal PNG image."""
    from PIL import Image

    buf = BytesIO()
    img = Image.new("RGB", (width, height), color=(255, 0, 0))
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_tiny_jpeg(width: int = 2, height: int = 2) -> bytes:
    """Return raw bytes of a valid minimal JPEG image."""
    from PIL import Image

    buf = BytesIO()
    img = Image.new("RGB", (width, height), color=(0, 255, 0))
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_tiny_webp(width: int = 2, height: int = 2) -> bytes:
    """Return raw bytes of a valid minimal WEBP image."""
    from PIL import Image

    buf = BytesIO()
    img = Image.new("RGB", (width, height), color=(0, 0, 255))
    img.save(buf, format="WEBP")
    return buf.getvalue()


def _png_data_url(width: int = 2, height: int = 2) -> str:
    """Return a ``data:image/png;base64,...`` URL for a tiny PNG."""
    data = _make_tiny_png(width, height)
    b64 = base64.b64encode(data).decode()
    return f"data:image/png;base64,{b64}"


def _raw_b64_png(width: int = 2, height: int = 2) -> str:
    """Return raw (no data-URL header) base64 for a tiny PNG."""
    return base64.b64encode(_make_tiny_png(width, height)).decode()


# ---------------------------------------------------------------------------
# Import the module under test (will fail until image_input.py exists)
# ---------------------------------------------------------------------------

from invest_forge.tools.image_input import (  # noqa: E402
    ImagePolicy,
    ImageRef,
    ImageValidationError,
    ValidatedImage,
    _assert_public_url,
    _default_fetch,
    _resolve_and_validate,
    load_image,
    load_images,
)

# ===========================================================================
# 1. SSRF guard — scheme rejection
# ===========================================================================


@pytest.mark.unit
def test_ssrf_rejects_http_scheme() -> None:
    """Plain http:// is not in default allowed_schemes (https only)."""
    policy = ImagePolicy()
    with pytest.raises(ImageValidationError, match=r"(?i)scheme|disallowed"):
        _assert_public_url("http://example.com/img.png", policy)


@pytest.mark.unit
def test_ssrf_rejects_file_scheme() -> None:
    policy = ImagePolicy()
    with pytest.raises(ImageValidationError, match=r"(?i)scheme|disallowed"):
        _assert_public_url("file:///etc/passwd", policy)


@pytest.mark.unit
def test_ssrf_rejects_ftp_scheme() -> None:
    policy = ImagePolicy()
    with pytest.raises(ImageValidationError, match=r"(?i)scheme|disallowed"):
        _assert_public_url("ftp://host/img.png", policy)


@pytest.mark.unit
def test_ssrf_rejects_gopher_scheme() -> None:
    policy = ImagePolicy()
    with pytest.raises(ImageValidationError, match=r"(?i)scheme|disallowed"):
        _assert_public_url("gopher://host/img.png", policy)


# ===========================================================================
# 2. SSRF guard — private/loopback IP rejection
# ===========================================================================


@pytest.mark.unit
def test_ssrf_rejects_loopback_ipv4_literal() -> None:
    """https://127.0.0.1/x — loopback, rejected without DNS lookup."""
    policy = ImagePolicy()
    with pytest.raises(ImageValidationError, match=r"(?i)host|disallowed|private"):
        _assert_public_url("https://127.0.0.1/x", policy)


@pytest.mark.unit
def test_ssrf_rejects_cloud_metadata_ip() -> None:
    """169.254.169.254 — AWS/GCP metadata endpoint, link-local."""
    policy = ImagePolicy()
    with pytest.raises(ImageValidationError, match=r"(?i)host|disallowed|private"):
        _assert_public_url("https://169.254.169.254/latest/meta-data", policy)


@pytest.mark.unit
def test_ssrf_rejects_rfc1918_10_block() -> None:
    """10.0.0.5 — RFC1918 private address."""
    policy = ImagePolicy()
    with pytest.raises(ImageValidationError, match=r"(?i)host|disallowed|private"):
        _assert_public_url("https://10.0.0.5/", policy)


@pytest.mark.unit
def test_ssrf_rejects_rfc1918_192168_block() -> None:
    """192.168.1.1 — RFC1918 private address."""
    policy = ImagePolicy()
    with pytest.raises(ImageValidationError, match=r"(?i)host|disallowed|private"):
        _assert_public_url("https://192.168.1.1/", policy)


@pytest.mark.unit
def test_ssrf_rejects_ipv6_loopback_literal() -> None:
    """https://[::1]/ — IPv6 loopback literal."""
    policy = ImagePolicy()
    with pytest.raises(ImageValidationError, match=r"(?i)host|disallowed|private"):
        _assert_public_url("https://[::1]/", policy)


@pytest.mark.unit
def test_ssrf_rejects_ipv4_mapped_ipv6() -> None:
    """https://[::ffff:127.0.0.1]/ — IPv4-mapped loopback."""
    policy = ImagePolicy()
    with pytest.raises(ImageValidationError, match=r"(?i)host|disallowed|private"):
        _assert_public_url("https://[::ffff:127.0.0.1]/", policy)


@pytest.mark.unit
def test_ssrf_rejects_localhost_via_monkeypatched_dns() -> None:
    """'localhost' resolves to 127.0.0.1 — DNS-based check must catch it."""
    policy = ImagePolicy()

    def _fake_getaddrinfo(host, port, *args, **kwargs):  # type: ignore[override]
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", port or 0))]

    with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo):
        with pytest.raises(ImageValidationError, match=r"(?i)host|disallowed|private"):
            _assert_public_url("https://localhost/img.png", policy)


# ===========================================================================
# 3. SSRF guard — error message safety (no IP/path leakage)
# ===========================================================================


@pytest.mark.unit
def test_ssrf_error_message_hides_ip() -> None:
    """The ImageValidationError must NOT echo back the IP address."""
    policy = ImagePolicy()
    try:
        _assert_public_url("https://127.0.0.1/x", policy)
        pytest.fail("Expected ImageValidationError")
    except ImageValidationError as exc:
        assert "127.0.0.1" not in str(exc), "IP address leaked in error message"


@pytest.mark.unit
def test_ssrf_error_message_hides_resolved_ip_for_hostname() -> None:
    """Resolved IPs for hostnames must not leak in error messages."""
    policy = ImagePolicy()

    def _fake_getaddrinfo(host, port, *args, **kwargs):  # type: ignore[override]
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.168.99.1", port or 0))]

    with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo):
        try:
            _assert_public_url("https://internal.corp/img.png", policy)
            pytest.fail("Expected ImageValidationError")
        except ImageValidationError as exc:
            assert "192.168.99.1" not in str(exc), "Resolved IP leaked in error message"


# ===========================================================================
# 4. URL allowlist
# ===========================================================================


@pytest.mark.unit
def test_url_allowlist_rejects_unlisted_host() -> None:
    """url_allowlist set, host not in it → rejected."""
    policy = ImagePolicy(url_allowlist=("images.example.com",))
    with pytest.raises(ImageValidationError, match=r"(?i)host|allowlist|disallowed"):
        _assert_public_url("https://other.example.com/img.png", policy)


@pytest.mark.unit
def test_url_allowlist_case_insensitive_match() -> None:
    """Host matching in allowlist is case-insensitive; public IP resolves fine."""
    # We use an IP literal so no DNS lookup is needed for the non-private path.
    # The allowlist check happens after the scheme check.
    policy = ImagePolicy(url_allowlist=("IMAGES.EXAMPLE.COM",))
    # A public IP literal (e.g. a known CDN) — the guard should PASS the
    # allowlist check. We use a public IP literal so no DNS patch is needed.
    # Just verify the guard does NOT raise due to allowlist for a matching host.
    policy_with_public = ImagePolicy(
        url_allowlist=("images.example.com",),
        allowed_schemes=frozenset({"https"}),
    )
    # Matching host (case-insensitive) — allowlist check should pass.
    # We still need a public IP; use monkeypatching to return a public IP.
    def _fake_getaddrinfo(host, port, *args, **kwargs):  # type: ignore[override]
        # 1.2.3.4 is a public IP
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.2.3.4", port or 0))]

    with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo):
        # Should NOT raise (allowed host, public IP)
        _assert_public_url("https://Images.Example.Com/img.png", policy_with_public)


@pytest.mark.unit
def test_url_allowlist_rejects_non_listed_even_if_public() -> None:
    """Even a public IP must be rejected when not in allowlist."""
    policy = ImagePolicy(url_allowlist=("images.example.com",))

    def _fake_getaddrinfo(host, port, *args, **kwargs):  # type: ignore[override]
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.2.3.4", port or 0))]

    with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo):
        with pytest.raises(ImageValidationError, match=r"(?i)host|allowlist|disallowed"):
            _assert_public_url("https://other.com/img.png", policy)


# ===========================================================================
# 5. Valid URL fetch (injected fetch_fn — no network)
# ===========================================================================


@pytest.mark.unit
def test_valid_url_fetch_returns_validated_image() -> None:
    """A URL that passes the guard + returns a tiny PNG → ValidatedImage."""
    png_bytes = _make_tiny_png()

    def _fetch(url: str, policy: ImagePolicy) -> bytes:
        return png_bytes

    # Use an IP literal so getaddrinfo is called predictably.
    def _fake_getaddrinfo(host, port, *args, **kwargs):  # type: ignore[override]
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.2.3.4", port or 0))]

    ref = ImageRef(kind="url", value="https://cdn.example.com/img.png")
    policy = ImagePolicy()

    with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo):
        result = load_image(ref, policy, fetch_fn=_fetch)

    assert isinstance(result, ValidatedImage)
    assert result.mime == "image/png"
    assert result.width == 2
    assert result.height == 2
    assert result.num_bytes == len(png_bytes)
    assert result.data_url.startswith("data:image/png;base64,")


@pytest.mark.unit
def test_valid_url_fetch_jpeg() -> None:
    jpeg_bytes = _make_tiny_jpeg()

    def _fetch(url: str, policy: ImagePolicy) -> bytes:
        return jpeg_bytes

    def _fake_getaddrinfo(host, port, *args, **kwargs):  # type: ignore[override]
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("5.5.5.5", port or 0))]

    ref = ImageRef(kind="url", value="https://cdn.example.com/img.jpg")
    policy = ImagePolicy()

    with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo):
        result = load_image(ref, policy, fetch_fn=_fetch)

    assert result.mime in ("image/jpeg", "image/jpg")
    assert result.data_url.startswith("data:image/jpeg;base64,")


@pytest.mark.unit
def test_valid_url_fetch_webp() -> None:
    webp_bytes = _make_tiny_webp()

    def _fetch(url: str, policy: ImagePolicy) -> bytes:
        return webp_bytes

    def _fake_getaddrinfo(host, port, *args, **kwargs):  # type: ignore[override]
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("5.5.5.5", port or 0))]

    ref = ImageRef(kind="url", value="https://cdn.example.com/img.webp")
    policy = ImagePolicy()

    with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo):
        result = load_image(ref, policy, fetch_fn=_fetch)

    assert result.mime == "image/webp"
    assert result.data_url.startswith("data:image/webp;base64,")


# ===========================================================================
# 6. base64 / data URL
# ===========================================================================


@pytest.mark.unit
def test_base64_data_url_accepted() -> None:
    """A valid PNG data URL is accepted and returns ValidatedImage."""
    ref = ImageRef(kind="base64", value=_png_data_url())
    result = load_image(ref, ImagePolicy())
    assert result.mime == "image/png"
    assert result.data_url.startswith("data:image/png;base64,")


@pytest.mark.unit
def test_raw_base64_accepted() -> None:
    """Raw base64 string (no data: prefix) is also accepted."""
    ref = ImageRef(kind="base64", value=_raw_b64_png())
    result = load_image(ref, ImagePolicy())
    assert result.mime == "image/png"


@pytest.mark.unit
def test_base64_non_image_bytes_rejected() -> None:
    """Random bytes that aren't a valid image must be rejected."""
    garbage = base64.b64encode(b"this is not an image at all!!").decode()
    ref = ImageRef(kind="base64", value=garbage)
    with pytest.raises(ImageValidationError, match=r"(?i)invalid|not.*image|image"):
        load_image(ref, ImagePolicy())


@pytest.mark.unit
def test_base64_oversized_rejected() -> None:
    """Decoded size > max_bytes must be rejected BEFORE Pillow opens it."""
    # Make a tiny policy with max_bytes=10 (much less than any real PNG).
    policy = ImagePolicy(max_bytes=10)
    ref = ImageRef(kind="base64", value=_png_data_url())
    with pytest.raises(ImageValidationError, match=r"(?i)too large|size|bytes"):
        load_image(ref, policy)


@pytest.mark.unit
def test_base64_mime_derived_from_content_not_caller() -> None:
    """Even if caller claims mime=image/jpeg, Pillow's detection wins."""
    # Pass a PNG but claim JPEG in mime field.
    png_b64 = _raw_b64_png()
    ref = ImageRef(kind="base64", value=png_b64, mime="image/jpeg")
    result = load_image(ref, ImagePolicy())
    # True mime must come from Pillow, not from caller's claim.
    assert result.mime == "image/png"


# ===========================================================================
# 7. Pixel bomb protection
# ===========================================================================


@pytest.mark.unit
def test_pixel_bomb_rejected() -> None:
    """An image whose width*height > max_pixels must be rejected."""
    # Create a tiny policy: max_pixels = 3 (i.e. a 2×2 image has 4 pixels → rejected).
    policy = ImagePolicy(max_pixels=3)
    ref = ImageRef(kind="base64", value=_png_data_url(2, 2))  # 4 pixels
    with pytest.raises(ImageValidationError, match=r"(?i)pixel|too large|resolution"):
        load_image(ref, policy)


@pytest.mark.unit
def test_pixel_bomb_just_under_limit_accepted() -> None:
    """An image whose width*height == max_pixels is accepted."""
    policy = ImagePolicy(max_pixels=4)  # exactly 2×2 = 4 pixels
    ref = ImageRef(kind="base64", value=_png_data_url(2, 2))
    result = load_image(ref, policy)
    assert result.width == 2 and result.height == 2


# ===========================================================================
# 8. Unsupported image format rejection
# ===========================================================================


@pytest.mark.unit
def test_bmp_format_rejected() -> None:
    """BMP is not in the allowed formats set."""
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (2, 2)).save(buf, format="BMP")
    bmp_b64 = base64.b64encode(buf.getvalue()).decode()
    ref = ImageRef(kind="base64", value=bmp_b64)
    with pytest.raises(ImageValidationError, match=r"(?i)format|unsupported|not allowed"):
        load_image(ref, ImagePolicy())


# ===========================================================================
# 9. Local path validation
# ===========================================================================


@pytest.mark.unit
def test_local_path_default_policy_raises() -> None:
    """Default policy has allow_local_paths=False → any path kind raises."""
    ref = ImageRef(kind="path", value="/tmp/img.png")
    with pytest.raises(ImageValidationError, match=r"(?i)path|local|disallowed"):
        load_image(ref, ImagePolicy())


@pytest.mark.unit
def test_local_path_allowed_inside_upload_dir(tmp_path: Path) -> None:
    """A file inside upload_dir is accepted when allow_local_paths=True."""
    img_path = tmp_path / "report.png"
    img_path.write_bytes(_make_tiny_png())

    policy = ImagePolicy(allow_local_paths=True, upload_dir=tmp_path)
    ref = ImageRef(kind="path", value=str(img_path))
    result = load_image(ref, policy)
    assert result.mime == "image/png"


@pytest.mark.unit
def test_local_path_escape_via_dotdot_rejected(tmp_path: Path) -> None:
    """A path that resolves outside upload_dir must be rejected."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    # Create a real file one level up
    outside = tmp_path / "secret.png"
    outside.write_bytes(_make_tiny_png())

    policy = ImagePolicy(allow_local_paths=True, upload_dir=upload_dir)
    # Try a path that traverses out via ../
    escape_path = upload_dir / ".." / "secret.png"
    ref = ImageRef(kind="path", value=str(escape_path))
    with pytest.raises(ImageValidationError, match=r"(?i)path|outside|disallowed"):
        load_image(ref, policy)


@pytest.mark.unit
def test_local_path_symlink_escape_rejected(tmp_path: Path) -> None:
    """A symlink inside upload_dir that points outside must be rejected."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(_make_tiny_png())
    # Create a symlink inside upload_dir pointing to the outside file.
    symlink = upload_dir / "link.png"
    symlink.symlink_to(outside)

    policy = ImagePolicy(allow_local_paths=True, upload_dir=upload_dir)
    ref = ImageRef(kind="path", value=str(symlink))
    with pytest.raises(ImageValidationError, match=r"(?i)path|outside|symlink|disallowed"):
        load_image(ref, policy)


@pytest.mark.unit
def test_local_path_no_upload_dir_raises() -> None:
    """allow_local_paths=True but upload_dir=None → must raise (no sandbox)."""
    policy = ImagePolicy(allow_local_paths=True, upload_dir=None)
    ref = ImageRef(kind="path", value="/tmp/any.png")
    with pytest.raises(ImageValidationError, match=r"(?i)upload_dir|directory|disallowed|path"):
        load_image(ref, policy)


@pytest.mark.unit
def test_local_path_error_hides_filesystem_path(tmp_path: Path) -> None:
    """Error messages for path violations must not echo the full path."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    outside = tmp_path / "secret.png"
    outside.write_bytes(_make_tiny_png())

    policy = ImagePolicy(allow_local_paths=True, upload_dir=upload_dir)
    ref = ImageRef(kind="path", value=str(outside))
    try:
        load_image(ref, policy)
        pytest.fail("Expected ImageValidationError")
    except ImageValidationError as exc:
        assert str(tmp_path) not in str(exc), "Absolute path leaked in error message"
        assert str(outside) not in str(exc), "Full path leaked in error message"


# ===========================================================================
# 10. load_images — batch
# ===========================================================================


@pytest.mark.unit
def test_load_images_respects_max_images() -> None:
    """Passing more refs than policy.max_images must raise."""
    policy = ImagePolicy(max_images=2)
    refs = [
        ImageRef(kind="base64", value=_png_data_url()),
        ImageRef(kind="base64", value=_png_data_url()),
        ImageRef(kind="base64", value=_png_data_url()),  # 3rd exceeds limit
    ]
    with pytest.raises(ImageValidationError, match=r"(?i)too many|max.*image|image.*limit"):
        load_images(refs, policy)


@pytest.mark.unit
def test_load_images_returns_ordered_results() -> None:
    """Results are in the same order as the input refs."""
    png = _make_tiny_png(2, 2)
    jpeg = _make_tiny_jpeg(2, 2)

    refs = [
        ImageRef(kind="base64", value=f"data:image/png;base64,{base64.b64encode(png).decode()}"),
        ImageRef(kind="base64", value=f"data:image/jpeg;base64,{base64.b64encode(jpeg).decode()}"),
    ]
    policy = ImagePolicy(max_images=4)
    results = load_images(refs, policy)
    assert len(results) == 2
    assert results[0].mime == "image/png"
    assert results[1].mime in ("image/jpeg", "image/jpg")


@pytest.mark.unit
def test_load_images_empty_list() -> None:
    """Empty input returns empty list."""
    results = load_images([], ImagePolicy())
    assert results == []


# ===========================================================================
# 11. ImagePolicy.from_env
# ===========================================================================


@pytest.mark.unit
def test_policy_from_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """from_env with no env vars returns default policy."""
    for var in (
        "VISION_MAX_IMAGES", "VISION_MAX_BYTES", "VISION_MAX_PIXELS",
        "VISION_ALLOW_LOCAL_PATHS", "VISION_UPLOAD_DIR",
        "VISION_URL_ALLOWLIST", "VISION_FETCH_TIMEOUT",
    ):
        monkeypatch.delenv(var, raising=False)

    p = ImagePolicy.from_env()
    assert p.max_images == ImagePolicy().max_images
    assert p.max_bytes == ImagePolicy().max_bytes
    assert p.allow_local_paths is False


@pytest.mark.unit
def test_policy_from_env_custom_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """from_env correctly parses all environment variables."""
    monkeypatch.setenv("VISION_MAX_IMAGES", "8")
    monkeypatch.setenv("VISION_MAX_BYTES", "1000000")
    monkeypatch.setenv("VISION_MAX_PIXELS", "5000000")
    monkeypatch.setenv("VISION_ALLOW_LOCAL_PATHS", "true")
    monkeypatch.setenv("VISION_UPLOAD_DIR", str(tmp_path))
    monkeypatch.setenv("VISION_URL_ALLOWLIST", "a.com,b.com")
    monkeypatch.setenv("VISION_FETCH_TIMEOUT", "10.0")

    p = ImagePolicy.from_env()
    assert p.max_images == 8
    assert p.max_bytes == 1_000_000
    assert p.max_pixels == 5_000_000
    assert p.allow_local_paths is True
    assert p.upload_dir == tmp_path
    assert set(p.url_allowlist) == {"a.com", "b.com"}
    assert p.fetch_timeout == 10.0


# ===========================================================================
# 12. URL fetch size cap (via fetch_fn that returns too many bytes)
# ===========================================================================


@pytest.mark.unit
def test_url_fetch_oversized_rejected() -> None:
    """If fetch_fn returns bytes > max_bytes, must raise."""
    large_data = b"\x89PNG" + b"\x00" * 10_000  # not a valid PNG, but size check comes first

    def _fetch(url: str, policy: ImagePolicy) -> bytes:
        return large_data

    def _fake_getaddrinfo(host, port, *args, **kwargs):  # type: ignore[override]
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.2.3.4", port or 0))]

    ref = ImageRef(kind="url", value="https://cdn.example.com/big.png")
    policy = ImagePolicy(max_bytes=100)  # tiny limit

    with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo):
        with pytest.raises(ImageValidationError, match=r"(?i)too large|size|bytes"):
            load_image(ref, policy, fetch_fn=_fetch)


# ===========================================================================
# 13. Dataclass immutability (frozen=True) / contract structure
# ===========================================================================


@pytest.mark.unit
def test_imageref_is_frozen() -> None:
    ref = ImageRef(kind="url", value="https://example.com/img.png")
    with pytest.raises((AttributeError, TypeError)):
        ref.value = "mutated"  # type: ignore[misc]


@pytest.mark.unit
def test_validatedimage_is_frozen() -> None:
    vi = ValidatedImage(
        data_url="data:image/png;base64,abc",
        mime="image/png",
        width=2,
        height=2,
        num_bytes=100,
    )
    with pytest.raises((AttributeError, TypeError)):
        vi.width = 999  # type: ignore[misc]


@pytest.mark.unit
def test_imagepolicy_is_frozen() -> None:
    p = ImagePolicy()
    with pytest.raises((AttributeError, TypeError)):
        p.max_images = 99  # type: ignore[misc]


# ===========================================================================
# 14. P3 — SSRF regression: numeric encodings, trailing dot, userinfo,
#     mixed records, IPv6 ULA/link-local.
# ===========================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "url",
    [
        "https://2130706433/x",  # decimal-packed 127.0.0.1
        "https://0x7f000001/x",  # hex-packed 127.0.0.1
        "https://127.1/x",  # short-form 127.0.0.1
        "https://0177.0.0.1/x",  # octal-ish 127.0.0.1
    ],
)
def test_ssrf_rejects_numeric_ip_encodings(url: str) -> None:
    """Decimal/hex/octal/short loopback encodings must be rejected.

    These resolve to 127.x via getaddrinfo; we monkeypatch it to mimic the OS
    expanding the numeric literal to loopback so the test stays offline.
    """
    policy = ImagePolicy()

    def _fake_getaddrinfo(host, port, *args, **kwargs):  # type: ignore[override]
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", port or 0))]

    with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo):
        with pytest.raises(ImageValidationError, match=r"(?i)host|disallowed|private"):
            _assert_public_url(url, policy)


@pytest.mark.unit
def test_ssrf_trailing_dot_hostname_normalized_for_allowlist() -> None:
    """A trailing-dot FQDN must not dodge an exact-match allowlist."""
    policy = ImagePolicy(url_allowlist=("images.example.com",))
    # "images.example.com." should normalize and pass the allowlist; then a
    # public IP keeps it valid.
    def _fake_getaddrinfo(host, port, *args, **kwargs):  # type: ignore[override]
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.2.3.4", port or 0))]

    with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo):
        _assert_public_url("https://images.example.com./img.png", policy)

    # And a trailing-dot non-listed host is still rejected.
    with pytest.raises(ImageValidationError, match=r"(?i)host|allowlist|disallowed"):
        _assert_public_url("https://evil.example.com./img.png", policy)


@pytest.mark.unit
def test_ssrf_userinfo_does_not_smuggle_host() -> None:
    """userinfo like user@127.0.0.1 must be parsed as host=127.0.0.1."""
    policy = ImagePolicy()
    # urlsplit().hostname strips the userinfo, leaving 127.0.0.1 as the host.
    with pytest.raises(ImageValidationError, match=r"(?i)host|disallowed|private"):
        _assert_public_url("https://allowed.com@127.0.0.1/x", policy)


@pytest.mark.unit
def test_ssrf_userinfo_with_allowlist_uses_real_host() -> None:
    """An allowlisted name in userinfo must not bypass the real-host check."""
    policy = ImagePolicy(url_allowlist=("allowed.com",))
    with pytest.raises(ImageValidationError, match=r"(?i)host|allowlist|disallowed|private"):
        _assert_public_url("https://allowed.com@127.0.0.1/x", policy)


@pytest.mark.unit
def test_ssrf_rejects_mixed_public_and_private_a_records() -> None:
    """If ANY resolved A record is private, the whole host is rejected."""
    policy = ImagePolicy()

    def _fake_getaddrinfo(host, port, *args, **kwargs):  # type: ignore[override]
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.2.3.4", port or 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.5", port or 0)),
        ]

    with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo):
        with pytest.raises(ImageValidationError, match=r"(?i)host|disallowed|private"):
            _assert_public_url("https://mixed.example.com/img.png", policy)


@pytest.mark.unit
def test_ssrf_rejects_unspecified_zero_address() -> None:
    policy = ImagePolicy()
    with pytest.raises(ImageValidationError, match=r"(?i)host|disallowed|private"):
        _assert_public_url("https://0.0.0.0/x", policy)


@pytest.mark.unit
@pytest.mark.parametrize(
    "url",
    [
        "https://[fc00::1]/x",  # IPv6 unique-local (ULA)
        "https://[fe80::1]/x",  # IPv6 link-local
        "https://[::]/x",  # IPv6 unspecified
    ],
)
def test_ssrf_rejects_ipv6_ula_and_linklocal(url: str) -> None:
    policy = ImagePolicy()
    with pytest.raises(ImageValidationError, match=r"(?i)host|disallowed|private"):
        _assert_public_url(url, policy)


@pytest.mark.unit
def test_resolve_and_validate_returns_public_ip() -> None:
    """The shared resolver returns the validated (family, ip) for a public host."""
    def _fake_getaddrinfo(host, port, *args, **kwargs):  # type: ignore[override]
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.2.3.4", port or 0))]

    with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo):
        out = _resolve_and_validate("cdn.example.com", 443)
    assert out == [(socket.AF_INET, "1.2.3.4")]


# ===========================================================================
# 15. P1 — DNS-rebinding: hardened fetch must refuse even if the SECOND
#     resolution (at connect time) would return a private IP.
# ===========================================================================


def _make_fake_https_base(connected_to: list, captured: dict, response_factory):
    """Build a standalone HTTPSConnection stand-in (NOT subclassing the real one).

    Our production ``_build_pinned_connection_cls`` subclasses whatever
    ``http.client.HTTPSConnection`` currently is, overriding ``connect()``. By
    patching the name with a plain fake (no real ``__init__`` chain), the pinned
    subclass inherits this fake's ``__init__`` and we still exercise the pinned
    ``connect()`` (socket-to-IP + ``wrap_socket(server_hostname=host)``) logic.
    """

    class _FakeHTTPS:
        def __init__(self, host, port=None, *, timeout=None, context=None, **kw):
            self.host = host
            self.port = port or 443
            self.timeout = timeout
            self._context = context
            self.sock = None
            self._tunnel_host = None

        # connect() is overridden by the pinned subclass under test.
        def request(self, method, url, body=None, headers=None):  # type: ignore[override]
            captured["method"] = method
            captured["path"] = url
            captured["host_header"] = (headers or {}).get("Host")
            self.connect()  # trigger the real pinned connect()

        def getresponse(self):  # type: ignore[override]
            return response_factory()

        def close(self):  # type: ignore[override]
            pass

    return _FakeHTTPS


@pytest.mark.unit
def test_hardened_fetch_pins_first_resolution_against_rebinding() -> None:
    """getaddrinfo returns public THEN private across two calls.

    The pinned fetcher resolves once and connects to the validated public IP;
    it must NOT re-resolve to the private IP. We capture what
    ``socket.create_connection`` is asked to connect to and the SNI hostname.
    """
    png = _make_tiny_png()
    calls = {"n": 0}

    def _fake_getaddrinfo(host, port, *args, **kwargs):  # type: ignore[override]
        calls["n"] += 1
        if calls["n"] == 1:
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.2.3.4", port or 0))]
        # A second resolution (the rebind) would hand back loopback.
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", port or 0))]

    connected_to: list = []

    class _FakeSock:
        def close(self) -> None:  # pragma: no cover - trivial
            pass

    def _fake_create_connection(address, timeout=None, *a, **k):  # type: ignore[override]
        connected_to.append(("CONNECT", address))
        return _FakeSock()

    class _FakeResponse:
        status = 200

        def read(self, n: int = -1) -> bytes:
            return png

    class _FakeContext:
        check_hostname = True

        def wrap_socket(self, sock, server_hostname=None):  # type: ignore[override]
            connected_to.append(("SNI", server_hostname))
            return sock

    captured: dict = {}
    fake_https = _make_fake_https_base(connected_to, captured, lambda: _FakeResponse())

    with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo), mock.patch(
        "socket.create_connection", side_effect=_fake_create_connection
    ), mock.patch("ssl.create_default_context", return_value=_FakeContext()), mock.patch(
        "http.client.HTTPSConnection", fake_https
    ):
        data = _default_fetch("https://cdn.example.com/img.png", ImagePolicy())

    assert data == png
    # Connected to the FIRST validated public IP, never the rebind 127.0.0.1.
    connect_targets = [addr for tag, addr in connected_to if tag == "CONNECT"]
    assert ("1.2.3.4", 443) in connect_targets
    assert all(addr[0] != "127.0.0.1" for addr in connect_targets)
    # TLS SNI / cert verification used the ORIGINAL hostname, not the IP.
    sni = [val for tag, val in connected_to if tag == "SNI"]
    assert sni == ["cdn.example.com"]
    # Host header preserved the original hostname.
    assert captured["host_header"] == "cdn.example.com"


@pytest.mark.unit
def test_hardened_fetch_refuses_redirect() -> None:
    """A 3xx response must be refused, never chased."""

    def _fake_getaddrinfo(host, port, *args, **kwargs):  # type: ignore[override]
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.2.3.4", port or 0))]

    class _RedirectResponse:
        status = 302

        def read(self, n: int = -1) -> bytes:  # pragma: no cover - not reached
            return b""

    connected_to: list = []
    captured: dict = {}
    fake_https = _make_fake_https_base(connected_to, captured, lambda: _RedirectResponse())

    class _FakeSock:
        def close(self) -> None:  # pragma: no cover
            pass

    class _FakeContext:
        def wrap_socket(self, sock, server_hostname=None):  # type: ignore[override]
            return sock

    with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo), mock.patch(
        "socket.create_connection", side_effect=lambda *a, **k: _FakeSock()
    ), mock.patch("ssl.create_default_context", return_value=_FakeContext()), mock.patch(
        "http.client.HTTPSConnection", fake_https
    ):
        with pytest.raises(ImageValidationError, match=r"(?i)host|disallowed|invalid"):
            _default_fetch("https://cdn.example.com/img.png", ImagePolicy())


# ===========================================================================
# 16. P1 — base64 encoded-length bound (pre-decode memory-DoS guard).
# ===========================================================================


@pytest.mark.unit
def test_base64_encoded_length_bound_rejected_before_decode() -> None:
    """An over-long base64 string is rejected by length BEFORE decoding."""
    policy = ImagePolicy(max_bytes=100)
    # Build a base64 string far longer than 4*ceil(100/3)+4 chars.
    huge_b64 = "A" * 10_000  # decodes to ~7500 bytes, but length check fires first
    ref = ImageRef(kind="base64", value=huge_b64)
    with pytest.raises(ImageValidationError, match=r"(?i)too large|size|bytes"):
        load_image(ref, policy)


@pytest.mark.unit
def test_base64_data_url_encoded_length_bound() -> None:
    """The encoded-length bound also applies to data: URLs."""
    policy = ImagePolicy(max_bytes=100)
    huge = "data:image/png;base64," + "A" * 10_000
    ref = ImageRef(kind="base64", value=huge)
    with pytest.raises(ImageValidationError, match=r"(?i)too large|size|bytes"):
        load_image(ref, policy)


# ===========================================================================
# 17. P2 — DecompressionBombError path is converted to ImageValidationError.
# ===========================================================================


@pytest.mark.unit
def test_decompression_bomb_error_converted(monkeypatch: pytest.MonkeyPatch) -> None:
    """If Pillow raises DecompressionBombError, we surface a generic error."""
    from PIL import Image

    png = _make_tiny_png()

    real_open = Image.open

    def _boom(*args, **kwargs):
        raise Image.DecompressionBombError("too many pixels")

    monkeypatch.setattr(Image, "open", _boom)
    ref = ImageRef(kind="base64", value=base64.b64encode(png).decode())
    with pytest.raises(ImageValidationError, match=r"(?i)resolution|too large|pixel"):
        load_image(ref, ImagePolicy())

    # restore (monkeypatch auto-restores, this is just defensive)
    monkeypatch.setattr(Image, "open", real_open)


# ===========================================================================
# 18. P2 — local-path symlink TOCTOU: O_NOFOLLOW refuses a symlink target.
# ===========================================================================


@pytest.mark.unit
def test_local_path_direct_symlink_target_refused(tmp_path: Path) -> None:
    """A symlink whose own path is passed is refused by O_NOFOLLOW even when it
    points to a file INSIDE upload_dir (defends against post-check swap)."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    real = upload_dir / "real.png"
    real.write_bytes(_make_tiny_png())
    link = upload_dir / "link.png"
    link.symlink_to(real)  # symlink inside upload_dir, target also inside

    policy = ImagePolicy(allow_local_paths=True, upload_dir=upload_dir)
    ref = ImageRef(kind="path", value=str(link))
    # O_NOFOLLOW on the final component => refuse the symlink.
    with pytest.raises(ImageValidationError, match=r"(?i)invalid|path|disallowed"):
        load_image(ref, policy)


# ===========================================================================
# 19. CRITICAL — local-path INTERMEDIATE-component symlink TOCTOU.
#     openat-style walk anchored at upload_dir refuses a symlinked dir hop.
# ===========================================================================


def _symlink_supported(tmp_path: Path) -> bool:
    probe = tmp_path / "__sym_probe"
    target = tmp_path / "__sym_target"
    target.write_text("x")
    try:
        probe.symlink_to(target)
    except (OSError, NotImplementedError):
        return False
    finally:
        if probe.is_symlink() or probe.exists():
            probe.unlink()
        target.unlink()
    return True


@pytest.mark.unit
def test_local_path_intermediate_symlink_dir_rejected(tmp_path: Path) -> None:
    """An INTERMEDIATE dir component that is a symlink pointing OUTSIDE the
    upload tree must be rejected (O_NOFOLLOW on every descend hop)."""
    if not _symlink_supported(tmp_path):
        pytest.skip("platform lacks symlink support")

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    # A real directory OUTSIDE the upload tree, containing a real image.
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.png").write_bytes(_make_tiny_png())
    # An intermediate symlink INSIDE upload_dir that points to the outside dir.
    evil_link = upload_dir / "sub"
    evil_link.symlink_to(outside, target_is_directory=True)

    policy = ImagePolicy(allow_local_paths=True, upload_dir=upload_dir)
    # Path goes through the symlinked intermediate dir: uploads/sub/secret.png
    ref = ImageRef(kind="path", value=str(upload_dir / "sub" / "secret.png"))
    with pytest.raises(ImageValidationError, match=r"(?i)invalid|path|disallowed"):
        load_image(ref, policy)


@pytest.mark.unit
def test_local_path_nested_real_file_accepted(tmp_path: Path) -> None:
    """A normal nested file (real dirs all the way down) is still accepted."""
    upload_dir = tmp_path / "uploads"
    nested = upload_dir / "2026" / "q1"
    nested.mkdir(parents=True)
    img = nested / "report.png"
    img.write_bytes(_make_tiny_png())

    policy = ImagePolicy(allow_local_paths=True, upload_dir=upload_dir)
    ref = ImageRef(kind="path", value=str(img))
    result = load_image(ref, policy)
    assert result.mime == "image/png"


@pytest.mark.unit
def test_local_path_absolute_outside_rejected(tmp_path: Path) -> None:
    """An absolute path that is not anchored under upload_dir is rejected."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    outside = tmp_path / "elsewhere.png"
    outside.write_bytes(_make_tiny_png())

    policy = ImagePolicy(allow_local_paths=True, upload_dir=upload_dir)
    ref = ImageRef(kind="path", value=str(outside))
    with pytest.raises(ImageValidationError, match=r"(?i)invalid|path|disallowed"):
        load_image(ref, policy)


@pytest.mark.unit
def test_local_path_dotdot_component_rejected(tmp_path: Path) -> None:
    """A path containing a `..` component is rejected before any open."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (tmp_path / "secret.png").write_bytes(_make_tiny_png())

    policy = ImagePolicy(allow_local_paths=True, upload_dir=upload_dir)
    # Absolute form with a .. component embedded under the base prefix.
    ref = ImageRef(kind="path", value=str(upload_dir / "sub" / ".." / ".." / "secret.png"))
    with pytest.raises(ImageValidationError, match=r"(?i)invalid|path|disallowed"):
        load_image(ref, policy)


@pytest.mark.unit
def test_local_path_dotdot_relative_rejected(tmp_path: Path) -> None:
    """A relative path with a leading `..` is rejected."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    policy = ImagePolicy(allow_local_paths=True, upload_dir=upload_dir)
    ref = ImageRef(kind="path", value="../secret.png")
    with pytest.raises(ImageValidationError, match=r"(?i)invalid|path|disallowed"):
        load_image(ref, policy)
