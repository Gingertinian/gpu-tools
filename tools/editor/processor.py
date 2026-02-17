"""
Editor Processor

RunPod-side media editor with timeline-ready config contract.
Current implementation focuses on reliable FFmpeg rendering primitives:
- Trim start/end
- Playback rate
- Resize / fit mode
- FPS normalization
- Audio mute / volume

It accepts both camelCase and snake_case keys so it works with
backend key normalization.
"""

import json
import os
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFilter = None
    ImageFont = None

if Image is not None:
    if hasattr(Image, "Resampling"):
        LANCZOS_RESAMPLE = Image.Resampling.LANCZOS
    else:
        LANCZOS_RESAMPLE = getattr(Image, "LANCZOS", getattr(Image, "BICUBIC", 3))
else:
    LANCZOS_RESAMPLE = None

try:
    import numpy as np
except ImportError:
    np = None

# Import captioner functions for integrated caption rendering
try:
    from tools.captioner.processor import (
        create_caption_overlay,
        resolve_font_path,
        get_font_line_height,
        measure_text_width,
        wrap_text_mixed,
        draw_mixed_text_line,
    )

    CAPTIONER_AVAILABLE = True
except ImportError:
    CAPTIONER_AVAILABLE = False
    create_caption_overlay = None
    resolve_font_path = None
    get_font_line_height = None
    measure_text_width = None
    wrap_text_mixed = None
    draw_mixed_text_line = None

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac"}

VALID_PRESETS = {
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
}


def _cfg(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in config and config[key] not in (None, ""):
            return config[key]
    return default


def _clamp_number(
    value: Any,
    minimum: float,
    maximum: float,
    default: float,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number < minimum:
        return minimum
    if number > maximum:
        return maximum
    return number


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return default


def _normalize_hex_color(value: Any) -> str:
    if value is None:
        return "#000000"

    color = str(value).strip()
    if not color:
        return "#000000"

    if not color.startswith("#"):
        color = f"#{color}"

    if len(color) == 4:
        color = f"#{color[1] * 2}{color[2] * 2}{color[3] * 2}"

    if len(color) != 7:
        return "#000000"

    hex_chars = "0123456789abcdefABCDEF"
    if any(ch not in hex_chars for ch in color[1:]):
        return "#000000"

    return color.lower()


def _to_ffmpeg_color(hex_color: str) -> str:
    return f"0x{hex_color.lstrip('#')}"


def _is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def _is_audio(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in AUDIO_EXTENSIONS


def _build_atempo_filters(rate: float) -> List[str]:
    # FFmpeg atempo supports [0.5, 2.0] per filter instance.
    filters: List[str] = []
    remaining = max(0.25, min(4.0, rate))

    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0

    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining *= 2.0

    filters.append(f"atempo={remaining:.5f}")
    return filters


def _run_ffmpeg(command: List[str]) -> None:
    command_to_run = list(command)
    filter_script_path: Optional[str] = None

    try:
        if os.name == "nt" and "-filter_complex" in command_to_run:
            fc_idx = command_to_run.index("-filter_complex")
            if fc_idx + 1 < len(command_to_run):
                filter_complex_value = command_to_run[fc_idx + 1]
                command_length = sum(len(part) + 1 for part in command_to_run)
                if (
                    isinstance(filter_complex_value, str)
                    and filter_complex_value
                    and (len(filter_complex_value) > 7000 or command_length > 26000)
                ):
                    fd, filter_script_path = tempfile.mkstemp(
                        prefix="editor_fc_",
                        suffix=".ffscript",
                        text=True,
                    )
                    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                        handle.write(filter_complex_value)

                    command_to_run = (
                        command_to_run[:fc_idx]
                        + ["-filter_complex_script", filter_script_path]
                        + command_to_run[fc_idx + 2 :]
                    )

        result = subprocess.run(command_to_run, capture_output=True, text=True)
    finally:
        if filter_script_path:
            try:
                os.remove(filter_script_path)
            except OSError:
                pass

    if result.returncode != 0:
        error_lines = (result.stderr or "").strip().splitlines()
        tail = "\n".join(error_lines[-15:]) if error_lines else "No stderr output"
        raise RuntimeError(f"FFmpeg failed (exit {result.returncode}):\n{tail}")


def _has_audio_stream(input_path: str) -> bool:
    probe_command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        input_path,
    ]
    result = subprocess.run(probe_command, capture_output=True, text=True)
    if result.returncode != 0:
        return False
    return bool((result.stdout or "").strip())


def _extract_audio_mono_f32(input_path: str, sample_rate: int = 22050) -> Any:
    if np is None:
        return None

    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        input_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        return None

    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size < sample_rate:
        return None

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return audio


def _detect_audio_rhythm_profile(input_path: str) -> Dict[str, float]:
    profile = {
        "bpm": 0.0,
        "amount": 0.0,
        "phaseSec": 0.0,
        "confidence": 0.0,
        "lowBandEnergy": 0.0,
        "highBandEnergy": 0.0,
        "transientDensity": 0.0,
    }

    if np is None or not _has_audio_stream(input_path):
        return profile
    assert np is not None

    sample_rate = 22050
    audio = _extract_audio_mono_f32(input_path=input_path, sample_rate=sample_rate)
    if audio is None:
        return profile

    audio = audio.astype(np.float32, copy=False)
    audio -= float(np.mean(audio))

    if float(np.max(np.abs(audio))) < 1e-5:
        return profile

    frame_size = int(sample_rate * 0.02)
    hop = int(sample_rate * 0.01)
    if frame_size <= 0 or hop <= 0 or audio.size <= frame_size:
        return profile

    frame_count = 1 + (audio.size - frame_size) // hop
    if frame_count < 64:
        return profile

    envelope = np.empty(frame_count, dtype=np.float32)
    for idx in range(frame_count):
        start = idx * hop
        frame = audio[start : start + frame_size]
        envelope[idx] = float(np.sqrt(np.mean(frame * frame) + 1e-9))

    diff = np.diff(envelope, prepend=envelope[0])
    onset = np.maximum(diff, 0.0)

    smooth_kernel = np.array(
        [0.08, 0.12, 0.16, 0.28, 0.16, 0.12, 0.08], dtype=np.float32
    )
    smooth_kernel /= np.sum(smooth_kernel)
    onset = np.convolve(onset, smooth_kernel, mode="same")

    onset -= float(np.mean(onset))
    onset_std = float(np.std(onset))
    if onset_std < 1e-7:
        return profile
    onset /= onset_std

    env_rate = float(sample_rate) / float(hop)
    min_bpm = 68.0
    max_bpm = 176.0
    lag_min = int(env_rate * 60.0 / max_bpm)
    lag_max = int(env_rate * 60.0 / min_bpm)

    autocorr = np.correlate(onset, onset, mode="full")
    autocorr = autocorr[autocorr.size // 2 :]

    if lag_max >= autocorr.size:
        lag_max = autocorr.size - 1
    if lag_min < 1:
        lag_min = 1
    if lag_max <= lag_min:
        return profile

    search = autocorr[lag_min : lag_max + 1]
    if search.size == 0:
        return profile

    best_rel = int(np.argmax(search))
    best_lag = lag_min + best_rel
    best_value = float(search[best_rel])

    median_value = float(np.median(search))
    p90_value = float(np.percentile(search, 90))
    spread = max(1e-6, p90_value - median_value)
    confidence = max(0.0, min(1.0, (best_value - median_value) / spread))

    bpm = 60.0 * env_rate / float(best_lag)
    phase_sec = 0.0

    if bpm < min_bpm or bpm > max_bpm or confidence < 0.07:
        threshold = float(np.percentile(onset, 85))
        min_peak_distance = max(2, int(env_rate * 60.0 / 196.0))
        peaks: List[int] = []
        last_peak = -(10**9)
        for i in range(1, onset.size - 1):
            if i - last_peak < min_peak_distance:
                continue
            if onset[i] < threshold:
                continue
            if onset[i] >= onset[i - 1] and onset[i] >= onset[i + 1]:
                peaks.append(i)
                last_peak = i

        if len(peaks) >= 4:
            intervals = np.diff(np.array(peaks, dtype=np.int32)).astype(np.float32)
            median_interval = float(np.median(intervals))
            if median_interval > 1e-6:
                bpm_from_peaks = 60.0 * env_rate / median_interval
                if min_bpm <= bpm_from_peaks <= max_bpm:
                    interval_std = float(np.std(intervals))
                    regularity = 1.0 - min(
                        1.0, interval_std / max(1e-6, median_interval)
                    )
                    peak_conf = max(0.0, min(1.0, regularity))
                    bpm = bpm_from_peaks
                    phase_sec = float(peaks[0] / env_rate)
                    confidence = max(confidence, peak_conf)

        if bpm < min_bpm or bpm > max_bpm:
            return profile

    if phase_sec <= 1e-9:
        beat_count = max(8, int(onset.size / best_lag))
        best_phase_idx = 0
        best_phase_score = -1e18
        for phase_idx in range(best_lag):
            indices = phase_idx + np.arange(beat_count, dtype=np.int32) * best_lag
            indices = indices[indices < onset.size]
            if indices.size == 0:
                continue
            score = float(np.sum(onset[indices]))
            if score > best_phase_score:
                best_phase_score = score
                best_phase_idx = phase_idx
        phase_sec = float(best_phase_idx / env_rate)

    onset_p95 = float(np.percentile(onset, 95))
    onset_p50 = float(np.percentile(onset, 50))
    punchiness = max(0.0, onset_p95 - onset_p50)
    amount = 0.16 + 0.54 * confidence + 0.18 * min(1.0, punchiness / 2.5)
    amount = max(0.12, min(0.95, amount))

    fft_size = 1024
    fft_hop = 512
    if audio.size > fft_size:
        fft_frames = 1 + (audio.size - fft_size) // fft_hop
        if fft_frames > 8:
            window = np.hanning(fft_size).astype(np.float32)
            spectrum_acc = None
            for i in range(fft_frames):
                start = i * fft_hop
                frame = audio[start : start + fft_size]
                if frame.size < fft_size:
                    break
                mag = np.abs(np.fft.rfft(frame * window)).astype(np.float32)
                if spectrum_acc is None:
                    spectrum_acc = mag
                else:
                    spectrum_acc += mag

            if spectrum_acc is not None:
                freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
                low_mask = (freqs >= 40.0) & (freqs <= 220.0)
                high_mask = (freqs >= 1800.0) & (freqs <= 6500.0)
                total_energy = float(np.sum(spectrum_acc) + 1e-9)
                low_energy = float(np.sum(spectrum_acc[low_mask]) / total_energy)
                high_energy = float(np.sum(spectrum_acc[high_mask]) / total_energy)
                profile["lowBandEnergy"] = float(
                    round(max(0.0, min(1.0, low_energy * 6.0)), 3)
                )
                profile["highBandEnergy"] = float(
                    round(max(0.0, min(1.0, high_energy * 10.0)), 3)
                )

    transient_threshold = float(np.percentile(onset, 93))
    transient_hits = float(np.sum(onset > transient_threshold))
    transient_density = transient_hits / max(1.0, float(onset.size))
    profile["transientDensity"] = float(
        round(max(0.0, min(1.0, transient_density * 18.0)), 3)
    )

    profile["bpm"] = float(round(bpm, 2))
    profile["amount"] = float(round(amount, 3))
    profile["phaseSec"] = float(round(phase_sec, 4))
    profile["confidence"] = float(round(confidence, 3))
    return profile


def _detect_audio_rhythm_from_timeline_sources(
    input_path: str,
    clips: List[Dict[str, Any]],
) -> Dict[str, float]:
    best = {
        "bpm": 0.0,
        "amount": 0.0,
        "phaseSec": 0.0,
        "confidence": 0.0,
        "lowBandEnergy": 0.0,
        "highBandEnergy": 0.0,
        "transientDensity": 0.0,
    }

    candidates: List[str] = []
    if isinstance(input_path, str) and input_path:
        candidates.append(input_path)

    for clip in clips:
        if str(clip.get("mediaType", "")).lower() != "video":
            continue
        source = clip.get("source")
        if not isinstance(source, str) or not source:
            continue
        if _looks_like_url(source):
            continue
        if not os.path.exists(source):
            continue
        candidates.append(source)

    seen: set = set()
    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)

        profile = _detect_audio_rhythm_profile(normalized)
        if profile.get("confidence", 0.0) > best.get("confidence", 0.0):
            best = profile

    return best


def _looks_like_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.startswith("http://") or value.startswith("https://")


def _escape_drawtext_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace(":", "\\:")
        .replace("%", "\\%")
        .replace("\n", "\\n")
    )


def _escape_drawtext_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace(":", "\\:")


def _resolve_fontfile(preferred: Any = None, bold: bool = False) -> Optional[str]:
    def _candidate_paths(candidate: str) -> List[str]:
        paths: List[str] = [candidate]
        if os.path.isabs(candidate):
            return paths

        processor_dir = os.path.dirname(os.path.abspath(__file__))
        paths.extend(
            [
                os.path.join(os.getcwd(), candidate),
                os.path.join(processor_dir, candidate),
                os.path.join(processor_dir, "..", candidate),
                os.path.join(processor_dir, "..", "..", candidate),
            ]
        )
        return paths

    candidates: List[str] = []

    if isinstance(preferred, str) and preferred.strip():
        requested = preferred.strip()
        normalized = requested.lower()
        if normalized in {"tiktok", "tiktokbold", "tiktok-bold"}:
            candidates.append("fonts/TikTokBold.otf")
        elif normalized in {"lightitalic", "tiktok-italic", "italic"}:
            candidates.append("fonts/LightItalic.otf")
        else:
            candidates.append(requested)

    if bold:
        candidates.extend(
            [
                "fonts/TikTokBold.otf",
                "/app/fonts/TikTokBold.otf",
                "C:/Windows/Fonts/arialbd.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "fonts/LightItalic.otf",
                "fonts/TikTokBold.otf",
                "/app/fonts/LightItalic.otf",
                "/app/fonts/TikTokBold.otf",
                "C:/Windows/Fonts/arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
        )

    for candidate in candidates:
        for path in _candidate_paths(candidate):
            normalized = os.path.normpath(path)
            if os.path.exists(normalized):
                return normalized

    return None


def _resolve_tiktok_fontfile() -> Optional[str]:
    """Resolve TikTokBold.otf with absolute path priority."""
    # Get the directory of this file (processor.py)
    processor_dir = os.path.dirname(os.path.abspath(__file__))

    # Build absolute paths
    candidates = [
        os.path.join(processor_dir, "..", "..", "fonts", "TikTokBold.otf"),
        os.path.join(processor_dir, "..", "..", "fonts", "tiktokbold.otf"),
        "/app/fonts/TikTokBold.otf",
        "/app/fonts/tiktokbold.otf",
        "C:/Windows/Fonts/TikTokBold.otf",
        "fonts/TikTokBold.otf",
    ]

    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if os.path.exists(normalized):
            return normalized
    return None


def _normalize_text_style(value: Any) -> str:
    if not isinstance(value, str):
        return "default"
    return value.strip().lower().replace("_", "-")


def _is_captioner_tiktok_style(style: str) -> bool:
    return style in {
        "captioner-tiktok",
        "tiktok-captioner",
        "captioner",
        "tiktok-caption",
        "tiktok-style",
    }


def _render_captioner_tiktok_overlay_png(
    output_png_path: str,
    width: int,
    height: int,
    overlay: Dict[str, Any],
    captioner_defaults: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Render caption overlay using the captioner processor directly.
    This ensures 100% identical rendering to the standalone captioner tool.
    """
    if not CAPTIONER_AVAILABLE or create_caption_overlay is None:
        raise RuntimeError(
            "Captioner integration not available. Cannot render TikTok-style captions."
        )

    text_value = str(overlay.get("text", "")).strip()
    if not text_value:
        return False

    if _as_bool(overlay.get("uppercase", False), default=False):
        text_value = text_value.upper()

    defaults = captioner_defaults if isinstance(captioner_defaults, dict) else {}

    # Build captioner config from overlay and defaults
    captioner_config: Dict[str, Any] = {
        "text": text_value,
        "showBackground": True,
        "enableTitleSyntax": False,
        "font": "TikTokBold.otf",
        "fontFamily": "TikTokBold.otf",
        "textColor": "#000000",
        "backgroundColor": "#FFFFFF",
        "backgroundOpacity": 1.0,
        "strokeWidth": 0,
        "shadow": False,
        "emojiStyle": "ios",
    }

    # Apply defaults first
    if "fontSize" in defaults:
        captioner_config["fontSize"] = int(defaults["fontSize"])
    if "textWidthRatio" in defaults:
        captioner_config["textWidthRatio"] = float(defaults["textWidthRatio"])
    if "alignment" in defaults:
        captioner_config["alignment"] = str(defaults["alignment"])
    if "position" in defaults:
        captioner_config["position"] = str(defaults["position"])
    if "positionX" in defaults:
        captioner_config["positionX"] = float(defaults["positionX"])
    if "positionY" in defaults:
        captioner_config["positionY"] = float(defaults["positionY"])

    # Apply overlay settings (they override defaults)
    if "fontSize" in overlay:
        captioner_config["fontSize"] = int(overlay["fontSize"])
    if "textWidthRatio" in overlay:
        captioner_config["textWidthRatio"] = float(overlay["textWidthRatio"])
    if "align" in overlay:
        captioner_config["alignment"] = str(overlay["align"])
    if "alignment" in overlay:
        captioner_config["alignment"] = str(overlay["alignment"])
    if "position" in overlay:
        captioner_config["position"] = str(overlay["position"])
    if "yPercent" in overlay:
        captioner_config["positionY"] = float(overlay["yPercent"])

    try:
        # Use the captioner processor directly - this guarantees identical rendering
        overlay_img = create_caption_overlay(
            width=width,
            height=height,
            config=captioner_config,
        )

        if overlay_img is None:
            raise RuntimeError("create_caption_overlay returned None")

        overlay_img.save(output_png_path, format="PNG")
        return True

    except Exception as e:
        raise RuntimeError(
            f"Failed to render caption using captioner processor: {e}"
        ) from e


def _text_position_y(position: str, height: int, custom_y: float) -> int:
    pos = position.strip().lower()
    if pos in {"top", "header"}:
        return int(height * 0.14)
    if pos in {"center", "middle"}:
        return int(height * 0.5)
    if pos in {"bottom", "footer"}:
        return int(height * 0.80)
    return int(height * max(0.05, min(0.95, custom_y)))


def _normalize_text_overlay(
    raw_overlay: Dict[str, Any],
    index: int,
    default_duration: float,
) -> Optional[Dict[str, Any]]:
    text_raw = raw_overlay.get("text", raw_overlay.get("value"))
    if not isinstance(text_raw, str) or not text_raw.strip():
        return None

    start_sec = _clamp_number(
        raw_overlay.get("startSec", raw_overlay.get("start_sec", 0)),
        0.0,
        max(0.1, default_duration),
        0.0,
    )
    duration_sec = _clamp_number(
        raw_overlay.get("durationSec", raw_overlay.get("duration_sec", 1.5)),
        0.15,
        max(0.15, default_duration),
        1.5,
    )

    style = _normalize_text_style(
        raw_overlay.get(
            "style",
            raw_overlay.get(
                "textStyle",
                raw_overlay.get(
                    "text_style", raw_overlay.get("captionStyle", "default")
                ),
            ),
        )
    )
    captioner_tiktok_style = _is_captioner_tiktok_style(style)

    default_font = "TikTokBold.otf" if captioner_tiktok_style else None
    default_color = "#000000" if captioner_tiktok_style else "#FFFFFF"
    default_stroke_width = 0.0 if captioner_tiktok_style else 4.0
    default_box = True if captioner_tiktok_style else True
    default_box_color = "#FFFFFF" if captioner_tiktok_style else "#000000"
    default_box_opacity = 1.0 if captioner_tiktok_style else 0.38
    default_box_border_w = 18 if captioner_tiktok_style else 12

    overlay = {
        "id": _safe_clip_name(raw_overlay.get("id"), f"txt-{index + 1}"),
        "text": text_raw.strip(),
        "startSec": start_sec,
        "durationSec": duration_sec,
        "style": style,
        "captionerTikTokStyle": captioner_tiktok_style,
        "position": str(raw_overlay.get("position", "bottom")),
        "align": str(raw_overlay.get("align", raw_overlay.get("alignment", "center"))),
        "yPercent": _clamp_number(
            raw_overlay.get("yPercent", raw_overlay.get("y_percent", 0.8)),
            0.05,
            0.95,
            0.8,
        ),
        "fontSize": int(
            _clamp_number(
                raw_overlay.get("fontSize", raw_overlay.get("font_size", 64)),
                20,
                190,
                64,
            )
        ),
        "font": raw_overlay.get("font", raw_overlay.get("fontFamily", default_font)),
        "bold": _as_bool(raw_overlay.get("bold", True), default=True),
        "uppercase": _as_bool(raw_overlay.get("uppercase", False), default=False),
        "color": _normalize_hex_color(raw_overlay.get("color", default_color)),
        "strokeColor": _normalize_hex_color(
            raw_overlay.get("strokeColor", raw_overlay.get("stroke_color", "#000000"))
        ),
        "strokeWidth": _clamp_number(
            raw_overlay.get(
                "strokeWidth", raw_overlay.get("stroke_width", default_stroke_width)
            ),
            0.0,
            14.0,
            default_stroke_width,
        ),
        "box": _as_bool(raw_overlay.get("box", default_box), default=default_box),
        "boxColor": _normalize_hex_color(
            raw_overlay.get("boxColor", raw_overlay.get("box_color", default_box_color))
        ),
        "boxOpacity": _clamp_number(
            raw_overlay.get(
                "boxOpacity", raw_overlay.get("box_opacity", default_box_opacity)
            ),
            0.0,
            1.0,
            default_box_opacity,
        ),
        "boxBorderW": int(
            _clamp_number(
                raw_overlay.get(
                    "boxBorderW",
                    raw_overlay.get("box_border_w", default_box_border_w),
                ),
                0,
                52,
                default_box_border_w,
            )
        ),
        "animation": str(raw_overlay.get("animation", "pop")).strip().lower(),
        "motionStrength": _clamp_number(
            raw_overlay.get("motionStrength", raw_overlay.get("motion_strength", 0.9)),
            0.0,
            1.0,
            0.9,
        ),
        "wordByWord": _as_bool(
            raw_overlay.get("wordByWord", raw_overlay.get("word_by_word", False)),
            default=False,
        ),
        "maxWordsPerChunk": int(
            _clamp_number(
                raw_overlay.get(
                    "maxWordsPerChunk", raw_overlay.get("max_words_per_chunk", 1)
                ),
                1,
                5,
                1,
            )
        ),
        "wordDurationSec": _clamp_number(
            raw_overlay.get(
                "wordDurationSec", raw_overlay.get("word_duration_sec", 0.28)
            ),
            0.08,
            1.2,
            0.28,
        ),
        "wordGapSec": _clamp_number(
            raw_overlay.get("wordGapSec", raw_overlay.get("word_gap_sec", 0.03)),
            0.0,
            0.5,
            0.03,
        ),
    }
    return overlay


def _extract_text_overlays_from_config(
    config: Dict[str, Any],
    default_duration: float,
) -> List[Dict[str, Any]]:
    raw_payload = _cfg(
        config,
        "textOverlaysJson",
        "text_overlays_json",
        "textOverlays",
        "text_overlays",
        default=[],
    )

    overlays_payload: Any = []
    if isinstance(raw_payload, str) and raw_payload.strip():
        try:
            overlays_payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            overlays_payload = []
    else:
        overlays_payload = raw_payload

    raw_overlays: List[Dict[str, Any]] = []
    if isinstance(overlays_payload, list):
        raw_overlays = [entry for entry in overlays_payload if isinstance(entry, dict)]
    elif isinstance(overlays_payload, dict):
        nested_candidates = [
            overlays_payload.get("overlays"),
            overlays_payload.get("textOverlays"),
            overlays_payload.get("captions"),
            overlays_payload.get("items"),
        ]
        for candidate in nested_candidates:
            if isinstance(candidate, list):
                raw_overlays = [entry for entry in candidate if isinstance(entry, dict)]
                if raw_overlays:
                    break

    normalized: List[Dict[str, Any]] = []
    for index, raw_overlay in enumerate(raw_overlays):
        item = _normalize_text_overlay(raw_overlay, index, default_duration)
        if item is not None:
            normalized.append(item)

    normalized.sort(key=lambda overlay: float(overlay["startSec"]))
    return normalized


def _expand_word_by_word_overlays(
    text_overlays: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []

    for overlay in text_overlays:
        if not _as_bool(overlay.get("wordByWord", False), default=False):
            expanded.append(overlay)
            continue

        raw_text = str(overlay.get("text", "")).strip()
        if not raw_text:
            continue

        tokens = [token for token in raw_text.split() if token.strip()]
        if not tokens:
            continue

        max_words_per_chunk = int(
            _clamp_number(overlay.get("maxWordsPerChunk", 1), 1, 5, 1)
        )
        chunks: List[str] = []
        for i in range(0, len(tokens), max_words_per_chunk):
            chunks.append(" ".join(tokens[i : i + max_words_per_chunk]))

        start_sec = max(0.0, float(overlay.get("startSec", 0.0)))
        total_duration = max(0.2, float(overlay.get("durationSec", 1.5)))
        word_duration = max(
            0.08,
            min(
                1.2,
                float(overlay.get("wordDurationSec", 0.28)),
            ),
        )
        word_gap = max(0.0, min(0.5, float(overlay.get("wordGapSec", 0.03))))

        consumed = len(chunks) * word_duration + max(0, len(chunks) - 1) * word_gap
        if consumed > total_duration and len(chunks) > 0:
            ratio = total_duration / consumed
            word_duration = max(0.06, word_duration * ratio)
            word_gap = max(0.0, word_gap * ratio)

        cursor = start_sec
        for chunk_index, chunk_text in enumerate(chunks):
            chunk = dict(overlay)
            chunk["id"] = f"{overlay.get('id', 'txt')}-w{chunk_index + 1}"
            chunk["text"] = chunk_text
            chunk["startSec"] = round(cursor, 3)
            chunk["durationSec"] = round(word_duration, 3)
            chunk["boxOpacity"] = _clamp_number(
                overlay.get("boxOpacity", 0.38),
                0.0,
                1.0,
                0.38,
            )
            expanded.append(chunk)
            cursor += word_duration + word_gap

    expanded.sort(key=lambda overlay: float(overlay.get("startSec", 0.0)))
    return expanded


def _extract_editor_captioner_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    enabled = _as_bool(
        _cfg(
            config,
            "captionerInEditorEnabled",
            "captioner_in_editor_enabled",
            "editorCaptionerEnabled",
            "editor_captioner_enabled",
            default=False,
        ),
        default=False,
    )
    if not enabled:
        return {}

    raw = _cfg(
        config,
        "captionerConfigJson",
        "captioner_config_json",
        "editorCaptionerConfigJson",
        "editor_captioner_config_json",
        default={},
    )

    defaults: Dict[str, Any] = {}
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                defaults = dict(parsed)
        except json.JSONDecodeError:
            defaults = {}
    elif isinstance(raw, dict):
        defaults = dict(raw)

    # Force TikTok visual identity when captioner-in-editor is enabled.
    defaults["showBackground"] = True
    defaults["font"] = "TikTokBold.otf"
    defaults["fontFamily"] = "TikTokBold.otf"
    defaults["textColor"] = "#000000"
    defaults["backgroundColor"] = "#FFFFFF"
    defaults["backgroundOpacity"] = 1.0
    defaults["strokeWidth"] = 0
    defaults["shadow"] = False
    defaults["enableTitleSyntax"] = False

    return defaults


def _build_tiktok_ads_overlays(
    config: Dict[str, Any],
    default_duration: float,
) -> List[Dict[str, Any]]:
    style_preset = (
        str(
            _cfg(
                config,
                "stylePreset",
                "style_preset",
                "tiktokStylePreset",
                default="none",
            )
        )
        .strip()
        .lower()
    )
    enabled = _as_bool(
        _cfg(
            config,
            "tiktokAdsEnabled",
            "tiktok_ads_enabled",
            "adsMode",
            default=False,
        ),
        default=False,
    )

    if not enabled and style_preset not in {
        "tiktok-ads",
        "tiktok_ads",
        "tiktok",
        "remotion-ads",
        "tiktok-ads-ugc",
        "tiktok_ads_ugc",
        "ugc",
        "tiktok-ads-direct-response",
        "tiktok_ads_direct_response",
        "direct-response",
        "tiktok-ads-problem-solution",
        "tiktok_ads_problem_solution",
        "problem-solution",
        "tiktok-ads-v6",
        "tiktok_ads_v6",
        "tiktok-v6",
        "remotion-v6",
        "tiktok-ads-v7",
        "tiktok_ads_v7",
        "tiktok-v7",
        "remotion-v7",
    }:
        return []

    duration = max(3.0, default_duration)
    hook_text = (
        str(
            _cfg(
                config,
                "tiktokAdsHook",
                "adHookText",
                "hookText",
                default="STOP THE SCROLL",
            )
        ).strip()
        or "STOP THE SCROLL"
    )
    offer_text = (
        str(
            _cfg(
                config,
                "tiktokAdsOffer",
                "adOfferText",
                "offerText",
                default="THIS WORKS FOR TIKTOK ADS",
            )
        ).strip()
        or "THIS WORKS FOR TIKTOK ADS"
    )
    proof_text = (
        str(
            _cfg(
                config,
                "tiktokAdsProof",
                "adProofText",
                "proofText",
                default="FAST CUTS + KINETIC TEXT + AUTO ZOOMS",
            )
        ).strip()
        or "FAST CUTS + KINETIC TEXT + AUTO ZOOMS"
    )
    cta_text = (
        str(
            _cfg(config, "tiktokAdsCta", "adCtaText", "ctaText", default="TRY IT NOW")
        ).strip()
        or "TRY IT NOW"
    )

    variant = "default"
    if style_preset in {
        "tiktok-ads-ugc",
        "tiktok_ads_ugc",
        "ugc",
    }:
        variant = "ugc"
    elif style_preset in {
        "tiktok-ads-direct-response",
        "tiktok_ads_direct_response",
        "direct-response",
    }:
        variant = "direct"
    elif style_preset in {
        "tiktok-ads-problem-solution",
        "tiktok_ads_problem_solution",
        "problem-solution",
    }:
        variant = "problem"
    elif style_preset in {
        "tiktok-ads-v6",
        "tiktok_ads_v6",
        "tiktok-v6",
        "remotion-v6",
    }:
        variant = "v6"
    elif style_preset in {
        "tiktok-ads-v7",
        "tiktok_ads_v7",
        "tiktok-v7",
        "remotion-v7",
    }:
        variant = "v7"

    hook_duration = min(2.2, duration * 0.16)
    cta_duration = min(2.6, max(1.8, duration * 0.2))

    base_hook = {
        "id": "ads-hook",
        "text": hook_text,
        "startSec": 0.12,
        "durationSec": hook_duration,
        "position": "top",
        "fontSize": 90,
        "bold": True,
        "uppercase": True,
        "style": "captioner-tiktok",
        "color": "#000000",
        "strokeColor": "#000000",
        "strokeWidth": 0,
        "box": True,
        "boxColor": "#FFFFFF",
        "boxOpacity": 1.0,
        "boxBorderW": 20,
        "animation": "punch",
        "motionStrength": 0.98,
        "wordByWord": True,
        "maxWordsPerChunk": 1,
        "wordDurationSec": 0.22,
    }

    base_offer = {
        "id": "ads-offer",
        "text": offer_text,
        "startSec": max(1.5, duration * 0.25),
        "durationSec": min(3.0, duration * 0.2),
        "position": "center",
        "fontSize": 70,
        "bold": True,
        "uppercase": True,
        "style": "captioner-tiktok",
        "color": "#000000",
        "strokeColor": "#000000",
        "strokeWidth": 0,
        "box": True,
        "boxColor": "#FFFFFF",
        "boxOpacity": 1.0,
        "boxBorderW": 18,
        "animation": "slide-up",
        "motionStrength": 0.9,
        "wordByWord": True,
        "maxWordsPerChunk": 2,
        "wordDurationSec": 0.24,
    }

    base_proof = {
        "id": "ads-proof",
        "text": proof_text,
        "startSec": max(3.5, duration * 0.52),
        "durationSec": min(3.1, duration * 0.2),
        "position": "center",
        "fontSize": 56,
        "bold": True,
        "uppercase": True,
        "style": "captioner-tiktok",
        "color": "#000000",
        "strokeColor": "#000000",
        "strokeWidth": 0,
        "box": True,
        "boxColor": "#FFFFFF",
        "boxOpacity": 1.0,
        "boxBorderW": 16,
        "animation": "slide-up",
        "motionStrength": 0.9,
        "wordByWord": True,
        "maxWordsPerChunk": 2,
        "wordDurationSec": 0.24,
    }

    base_cta = {
        "id": "ads-cta",
        "text": cta_text,
        "startSec": max(0.0, duration - cta_duration),
        "durationSec": cta_duration,
        "position": "bottom",
        "fontSize": 82,
        "bold": True,
        "uppercase": True,
        "style": "captioner-tiktok",
        "color": "#000000",
        "strokeColor": "#000000",
        "strokeWidth": 0,
        "box": True,
        "boxColor": "#FFFFFF",
        "boxOpacity": 1.0,
        "boxBorderW": 20,
        "animation": "punch",
        "motionStrength": 1.0,
        "wordByWord": True,
        "maxWordsPerChunk": 1,
        "wordDurationSec": 0.20,
    }

    overlays: List[Dict[str, Any]] = [base_hook, base_offer, base_proof, base_cta]

    if variant == "v6":
        hook_duration = min(1.9, duration * 0.14)
        overlays = [
            {
                **base_hook,
                "text": hook_text,
                "durationSec": hook_duration,
                "fontSize": 98,
                "wordDurationSec": 0.18,
                "motionStrength": 1.0,
            },
            {
                **base_offer,
                "id": "ads-problem",
                "text": "LOW HOLD RATE = LOST SALES",
                "startSec": max(1.05, duration * 0.18),
                "durationSec": min(1.8, duration * 0.13),
                "fontSize": 72,
                "animation": "punch",
                "maxWordsPerChunk": 2,
                "wordDurationSec": 0.2,
                "motionStrength": 0.98,
            },
            {
                **base_offer,
                "id": "ads-offer",
                "text": offer_text,
                "startSec": max(2.3, duration * 0.32),
                "durationSec": min(2.1, duration * 0.15),
                "fontSize": 74,
                "animation": "slide-up",
                "maxWordsPerChunk": 2,
                "wordDurationSec": 0.21,
                "motionStrength": 0.95,
            },
            {
                **base_proof,
                "id": "ads-proof",
                "text": proof_text,
                "startSec": max(4.3, duration * 0.5),
                "durationSec": min(2.2, duration * 0.16),
                "fontSize": 64,
                "animation": "punch",
                "maxWordsPerChunk": 2,
                "wordDurationSec": 0.2,
                "motionStrength": 0.95,
            },
            {
                **base_proof,
                "id": "ads-urgency",
                "text": "LIMITED TIME - BUYERS MOVE FAST",
                "startSec": max(6.8, duration * 0.69),
                "durationSec": min(1.9, duration * 0.14),
                "fontSize": 70,
                "animation": "punch",
                "maxWordsPerChunk": 2,
                "wordDurationSec": 0.19,
                "motionStrength": 1.0,
            },
            {
                **base_cta,
                "text": cta_text,
                "startSec": max(0.0, duration - min(2.8, duration * 0.2)),
                "durationSec": min(2.8, duration * 0.2),
                "fontSize": 92,
                "animation": "punch",
                "wordDurationSec": 0.18,
                "motionStrength": 1.0,
            },
        ]
    elif variant == "v7":
        hook_duration = min(1.6, duration * 0.12)
        overlays = [
            {
                **base_hook,
                "id": "ads-hook-v7",
                "text": hook_text,
                "startSec": 0.08,
                "durationSec": hook_duration,
                "fontSize": 102,
                "wordDurationSec": 0.15,
                "motionStrength": 1.0,
            },
            {
                **base_offer,
                "id": "ads-problem-v7",
                "text": "NO MOTION = NO ATTENTION",
                "startSec": max(0.95, duration * 0.16),
                "durationSec": min(1.45, duration * 0.11),
                "fontSize": 74,
                "animation": "punch",
                "maxWordsPerChunk": 2,
                "wordDurationSec": 0.17,
                "motionStrength": 1.0,
            },
            {
                **base_offer,
                "id": "ads-offer-v7",
                "text": offer_text,
                "startSec": max(2.0, duration * 0.28),
                "durationSec": min(1.8, duration * 0.13),
                "fontSize": 78,
                "animation": "slide-up",
                "maxWordsPerChunk": 2,
                "wordDurationSec": 0.18,
                "motionStrength": 0.95,
            },
            {
                **base_proof,
                "id": "ads-proof-v7",
                "text": proof_text,
                "startSec": max(3.9, duration * 0.47),
                "durationSec": min(1.9, duration * 0.14),
                "fontSize": 66,
                "animation": "punch",
                "maxWordsPerChunk": 2,
                "wordDurationSec": 0.18,
                "motionStrength": 0.95,
            },
            {
                **base_proof,
                "id": "ads-urgency-v7",
                "text": "LIMITED WINDOW - MOVE NOW",
                "startSec": max(5.9, duration * 0.64),
                "durationSec": min(1.6, duration * 0.12),
                "fontSize": 72,
                "animation": "punch",
                "maxWordsPerChunk": 2,
                "wordDurationSec": 0.16,
                "motionStrength": 1.0,
            },
            {
                **base_cta,
                "id": "ads-cta-v7",
                "text": cta_text,
                "startSec": max(0.0, duration - min(2.6, duration * 0.19)),
                "durationSec": min(2.6, duration * 0.19),
                "fontSize": 96,
                "animation": "punch",
                "wordDurationSec": 0.16,
                "motionStrength": 1.0,
            },
        ]

    if variant == "ugc":
        overlays[0]["fontSize"] = 84
        overlays[1]["text"] = f"REAL CREATOR FEEL - {offer_text}"
        overlays[2]["text"] = "SOCIAL PROOF + BEFORE/AFTER + TRUST"
        overlays[3]["text"] = "TRY THIS CREATIVE NOW"
    elif variant == "direct":
        overlays[0]["text"] = f"{hook_text} - THIS IS THE OFFER"
        overlays[1]["text"] = f"LIMITED TIME - {offer_text}"
        overlays[2]["text"] = "CLEAR BENEFIT + NUMBERS + DEADLINE"
        overlays[3]["text"] = "CLICK NOW - START TODAY"
    elif variant == "problem":
        overlays[0]["text"] = "PROBLEM"
        overlays[0]["wordByWord"] = False
        overlays[1]["text"] = "AGITATE - WHAT IT COSTS YOU"
        overlays[2]["text"] = "SOLUTION - FARMIUM EDITOR FLOW"
        overlays[3]["text"] = "CTA - TEST IT ON YOUR NEXT AD"

    normalized: List[Dict[str, Any]] = []
    for idx, raw_overlay in enumerate(overlays):
        item = _normalize_text_overlay(raw_overlay, idx, duration)
        if item is not None:
            normalized.append(item)
    return normalized


def _safe_clip_name(value: Any, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _safe_effect_type(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip().lower().replace("_", "").replace("-", "")
    return normalized


def _extract_effect_type(effect: Dict[str, Any]) -> str:
    return _safe_effect_type(
        effect.get("type")
        or effect.get("effectType")
        or effect.get("effect_type")
        or effect.get("name")
        or effect.get("effect")
    )


def _extract_effect_value(effect: Dict[str, Any], key: str, default: Any = None) -> Any:
    if key in effect:
        return effect[key]
    params = effect.get("params")
    if isinstance(params, dict) and key in params:
        return params[key]
    return default


def _parse_transition(raw_transition: Any) -> Dict[str, Any]:
    """
    Parse transition configuration into structured format.
    Supports: fade, dissolve, crossfade, wipe, slide, zoom, pixelize, etc.
    """
    result = {
        "type": "none",
        "durationSec": 0.0,
        "direction": "left",
        "color": "#000000",
        "easing": "linear",
    }

    if not isinstance(raw_transition, dict):
        return result

    transition_type = _safe_effect_type(
        raw_transition.get("type")
        or raw_transition.get("transitionType")
        or raw_transition.get("transition_type")
        or raw_transition.get("name")
        or "fade"
    )

    # Map all transition types
    valid_transitions = {
        # Fade transitions
        "fade",
        "dissolve",
        "crossfade",
        "dip",
        "diptocolor",
        "fadeblack",
        "fadewhite",
        "fadetocolor",
        # Directional wipes
        "wipe",
        "wipeleft",
        "wiperight",
        "wipeup",
        "wipedown",
        "wipelefttoprightbottom",
        "wiperighttopleftbottom",
        "wipein",
        "wipeout",
        # Slides
        "slide",
        "slideleft",
        "slideright",
        "slideup",
        "slidedown",
        "slidepush",
        "slideoverlap",
        # Zooms
        "zoom",
        "zoomin",
        "zoomout",
        "zoompan",
        # Special effects
        "pixelize",
        "pixelate",
        "mosaic",
        "blur",
        "radialblur",
        "cube",
        "swap",
        "flip",
        "rotate",
        # Glitch/VHS
        "glitch",
        "vhs",
        "rgbshift",
        # Light effects
        "flash",
        "lightflash",
        "glow",
    }

    if transition_type not in valid_transitions:
        return result

    result["type"] = transition_type

    # Parse duration
    duration_raw = (
        raw_transition.get("durationSec")
        or raw_transition.get("duration_sec")
        or raw_transition.get("duration")
        or _extract_effect_value(raw_transition, "durationSec")
        or 0.3
    )
    result["durationSec"] = _clamp_number(duration_raw, 0.0, 5.0, 0.3)

    # Parse direction
    direction = str(
        raw_transition.get("direction") or raw_transition.get("dir") or "left"
    ).lower()
    result["direction"] = direction

    # Parse color for dip/fade
    color = raw_transition.get("color") or raw_transition.get("dipColor")
    if color:
        result["color"] = _normalize_hex_color(color)

    # Parse easing
    easing = str(
        raw_transition.get("easing") or raw_transition.get("ease") or "linear"
    ).lower()
    result["easing"] = easing

    return result


def _parse_transition_fade_sec(raw_transition: Any) -> float:
    """Legacy function for backward compatibility."""
    parsed = _parse_transition(raw_transition)
    return parsed["durationSec"]


def _get_xfade_transition_name(trans_type: str, direction: str = "left") -> str:
    """
    Map transition type to FFmpeg xfade transition name.
    See: https://ffmpeg.org/ffmpeg-filters.html#xfade
    """
    transition_map = {
        # Fade/crossfade
        "fade": "fade",
        "dissolve": "fade",
        "crossfade": "fade",
        "dip": "fadeblack",
        "fadeblack": "fadeblack",
        "fadewhite": "fadewhite",
        "fadetocolor": "fade",
        # Wipes
        "wipe": "wipeleft",
        "wipeleft": "wipeleft",
        "wiperight": "wiperight",
        "wipeup": "wipeup",
        "wipedown": "wipedown",
        "wipelefttoprightbottom": "wipeleft",
        "wiperighttopleftbottom": "wiperight",
        "wipein": "smoothleft",
        "wipeout": "smoothright",
        # Slides
        "slide": "slideleft",
        "slideleft": "slideleft",
        "slideright": "slideright",
        "slideup": "slideup",
        "slidedown": "slidedown",
        "slidepush": "hslide",
        "slideoverlap": "coverleft",
        # Zooms
        "zoom": "zoomin",
        "zoomin": "zoomin",
        "zoomout": "zoomout",
        "zoompan": "zoompan",
        # Special effects
        "pixelize": "pixelize",
        "pixelate": "pixelize",
        "mosaic": "pixelize",
        "blur": "fade",
        "radialblur": "radialblur",
        "cube": "cube",
        "swap": "swap",
        "flip": "fliph",
        "rotate": "rotatate",
        # Glitch/VHS
        "glitch": "pixelize",
        "vhs": "pixelize",
        "rgbshift": "pixelize",
        # Light effects
        "flash": "fadewhite",
        "lightflash": "fadewhite",
        "glow": "fadewhite",
        # Other creative transitions
        "smoothleft": "smoothleft",
        "smoothright": "smoothright",
        "smoothup": "smoothup",
        "smoothdown": "smoothdown",
        "circlecrop": "circlecrop",
        "rectcrop": "rectcrop",
        "circleopen": "circleopen",
        "circleclose": "circleclose",
        "horzopen": "horzopen",
        "horzclose": "horzclose",
        "vertopen": "vertopen",
        "vertclose": "vertclose",
        "diagtl": "diagtl",
        "diagtr": "diagtr",
        "diagbl": "diagbl",
        "diagbr": "diagbr",
        "hlslice": "hlslice",
        "hrslice": "hrslice",
        "vuslice": "vuslice",
        "vdslice": "vdslice",
        "squeezeh": "squeezeh",
        "squeezev": "squeezev",
        "radial": "radial",
        "hblur": "hblur",
        "wipetl": "wipetl",
        "wipetr": "wipetr",
        "wipebl": "wipebl",
        "wipebr": "wipebr",
        "slicel": "slicel",
        "slicer": "slicer",
        "sliceu": "sliceu",
        "sliced": "sliced",
        "coverleft": "coverleft",
        "coverright": "coverright",
        "coverup": "coverup",
        "coverdown": "coverdown",
    }

    return transition_map.get(trans_type.lower(), "fade")


def _resolve_clip_effects(
    raw_effect_stack: Any, clip_duration: float
) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {
        "brightness": 0.0,
        "contrast": 1.0,
        "saturation": 1.0,
        "gamma": 1.0,
        "hueDeg": 0.0,
        "grayscale": False,
        "blurSigma": 0.0,
        "denoiseStrength": 0.0,
        "sharpenAmount": 0.0,
        "rotateDeg": 0,
        "flipHorizontal": False,
        "flipVertical": False,
        "speed": 1.0,
        "mute": False,
        "volume": 1.0,
        "fadeInSec": 0.0,
        "fadeOutSec": 0.0,
        "motionPreset": "none",
        "motionIntensity": 0.0,
        "motionSpeed": 1.0,
        "motionShake": 0.0,
        "zoomBoost": 0.0,
        "beatBpm": 0.0,
        "beatAmount": 0.0,
        "beatPhaseSec": 0.0,
        "beatZoomFactor": 1.0,
        "beatShakeFactor": 1.0,
    }

    if not isinstance(raw_effect_stack, list):
        return resolved

    for raw_effect in raw_effect_stack:
        if not isinstance(raw_effect, dict):
            continue
        if not _as_bool(raw_effect.get("enabled", True), default=True):
            continue

        effect_type = _extract_effect_type(raw_effect)
        if not effect_type:
            continue

        if effect_type in {"brightness"}:
            value = _extract_effect_value(raw_effect, "value", raw_effect.get("amount"))
            resolved["brightness"] = _clamp_number(
                value, -0.3, 0.3, resolved["brightness"]
            )
        elif effect_type in {"contrast"}:
            value = _extract_effect_value(raw_effect, "value", raw_effect.get("amount"))
            resolved["contrast"] = _clamp_number(value, 0.5, 2.0, resolved["contrast"])
        elif effect_type in {"saturation", "sat"}:
            value = _extract_effect_value(raw_effect, "value", raw_effect.get("amount"))
            resolved["saturation"] = _clamp_number(
                value, 0.0, 3.0, resolved["saturation"]
            )
        elif effect_type in {"gamma"}:
            value = _extract_effect_value(raw_effect, "value", raw_effect.get("amount"))
            resolved["gamma"] = _clamp_number(value, 0.5, 2.0, resolved["gamma"])
        elif effect_type in {"hue", "huerotate", "colorrotate"}:
            value = _extract_effect_value(raw_effect, "degrees", None)
            if value is None:
                value = _extract_effect_value(raw_effect, "hueDeg", None)
            if value is None:
                value = _extract_effect_value(
                    raw_effect, "value", raw_effect.get("amount")
                )
            resolved["hueDeg"] = _clamp_number(value, -180.0, 180.0, resolved["hueDeg"])
        elif effect_type in {"grayscale", "greyscale", "bw", "blackwhite"}:
            resolved["grayscale"] = _as_bool(
                _extract_effect_value(raw_effect, "value", True),
                default=resolved["grayscale"],
            )
        elif effect_type in {"blur", "gaussianblur", "gblur"}:
            value = _extract_effect_value(raw_effect, "sigma", None)
            if value is None:
                value = _extract_effect_value(
                    raw_effect, "value", raw_effect.get("amount")
                )
            resolved["blurSigma"] = _clamp_number(
                value, 0.0, 8.0, resolved["blurSigma"]
            )
        elif effect_type in {"denoise", "noise", "hqdn3d"}:
            value = _extract_effect_value(raw_effect, "strength", None)
            if value is None:
                value = _extract_effect_value(
                    raw_effect, "value", raw_effect.get("amount")
                )
            resolved["denoiseStrength"] = _clamp_number(
                value,
                0.0,
                10.0,
                resolved["denoiseStrength"],
            )
        elif effect_type in {"sharpen", "unsharp"}:
            value = _extract_effect_value(raw_effect, "amount", None)
            if value is None:
                value = _extract_effect_value(
                    raw_effect, "value", raw_effect.get("intensity")
                )
            resolved["sharpenAmount"] = _clamp_number(
                value,
                0.0,
                2.0,
                resolved["sharpenAmount"],
            )
        elif effect_type in {"rotate", "rotation"}:
            value = _extract_effect_value(raw_effect, "degrees", None)
            if value is None:
                value = _extract_effect_value(raw_effect, "value", 0)
            try:
                rotate_deg = int(round(float(value)))
            except (TypeError, ValueError):
                rotate_deg = 0
            if rotate_deg in {0, 90, 180, 270}:
                resolved["rotateDeg"] = rotate_deg
        elif effect_type in {"fliphorizontal", "hflip"}:
            resolved["flipHorizontal"] = _as_bool(
                _extract_effect_value(raw_effect, "value", True),
                default=resolved["flipHorizontal"],
            )
        elif effect_type in {"flipvertical", "vflip"}:
            resolved["flipVertical"] = _as_bool(
                _extract_effect_value(raw_effect, "value", True),
                default=resolved["flipVertical"],
            )
        elif effect_type in {"speed", "playbackrate", "timescale"}:
            value = _extract_effect_value(raw_effect, "value", raw_effect.get("amount"))
            resolved["speed"] = _clamp_number(value, 0.25, 4.0, resolved["speed"])
        elif effect_type in {"volume", "gain"}:
            value = _extract_effect_value(raw_effect, "value", raw_effect.get("amount"))
            resolved["volume"] = _clamp_number(value, 0.0, 2.0, resolved["volume"])
        elif effect_type in {"mute", "silence"}:
            resolved["mute"] = _as_bool(
                _extract_effect_value(raw_effect, "value", True),
                default=resolved["mute"],
            )
        elif effect_type in {"fadein", "fadeinvideo"}:
            value = _extract_effect_value(raw_effect, "durationSec", None)
            if value is None:
                value = _extract_effect_value(raw_effect, "duration", None)
            if value is None:
                value = _extract_effect_value(raw_effect, "value", 0)
            resolved["fadeInSec"] = _clamp_number(
                value,
                0.0,
                min(5.0, max(0.0, clip_duration)),
                resolved["fadeInSec"],
            )
        elif effect_type in {"fadeout", "fadeoutvideo"}:
            value = _extract_effect_value(raw_effect, "durationSec", None)
            if value is None:
                value = _extract_effect_value(raw_effect, "duration", None)
            if value is None:
                value = _extract_effect_value(raw_effect, "value", 0)
            resolved["fadeOutSec"] = _clamp_number(
                value,
                0.0,
                min(5.0, max(0.0, clip_duration)),
                resolved["fadeOutSec"],
            )
        elif effect_type in {
            "motion",
            "kineticmotion",
            "autopan",
            "remotion",
            "kenburns",
            "cameramotion",
        }:
            preset_raw = _extract_effect_value(raw_effect, "preset", None)
            if preset_raw is None:
                preset_raw = _extract_effect_value(raw_effect, "style", None)
            if preset_raw is None:
                preset_raw = _extract_effect_value(raw_effect, "value", "dynamic")
            if isinstance(preset_raw, str) and preset_raw.strip():
                resolved["motionPreset"] = preset_raw.strip()

            intensity_raw = _extract_effect_value(raw_effect, "intensity", None)
            if intensity_raw is None:
                intensity_raw = _extract_effect_value(raw_effect, "amount", None)
            if intensity_raw is None:
                intensity_raw = _extract_effect_value(raw_effect, "strength", 0.82)
            resolved["motionIntensity"] = _clamp_number(
                intensity_raw,
                0.0,
                1.0,
                resolved["motionIntensity"],
            )

            speed_raw = _extract_effect_value(raw_effect, "speed", None)
            if speed_raw is None:
                speed_raw = _extract_effect_value(raw_effect, "rate", 1.0)
            resolved["motionSpeed"] = _clamp_number(
                speed_raw,
                0.35,
                3.0,
                resolved["motionSpeed"],
            )
        elif effect_type in {"shake", "camerashake", "jitter", "vibrato"}:
            value = _extract_effect_value(raw_effect, "value", raw_effect.get("amount"))
            resolved["motionShake"] = _clamp_number(
                value,
                0.0,
                1.0,
                resolved["motionShake"],
            )
        elif effect_type in {"zoom", "autozoom", "zoompulse", "impactzoom"}:
            value = _extract_effect_value(raw_effect, "value", raw_effect.get("amount"))
            resolved["zoomBoost"] = _clamp_number(
                value,
                0.0,
                0.45,
                resolved["zoomBoost"],
            )
        elif effect_type in {"beatsync", "beat", "audiobeat", "musicpulse"}:
            bpm_raw = _extract_effect_value(raw_effect, "bpm", None)
            if bpm_raw is None:
                bpm_raw = _extract_effect_value(raw_effect, "tempo", 0)
            amount_raw = _extract_effect_value(raw_effect, "amount", None)
            if amount_raw is None:
                amount_raw = _extract_effect_value(raw_effect, "value", 0)
            phase_raw = _extract_effect_value(raw_effect, "phaseSec", None)
            if phase_raw is None:
                phase_raw = _extract_effect_value(raw_effect, "phase", 0)
            resolved["beatBpm"] = _clamp_number(
                bpm_raw,
                0.0,
                220.0,
                resolved["beatBpm"],
            )
            resolved["beatAmount"] = _clamp_number(
                amount_raw,
                0.0,
                1.0,
                resolved["beatAmount"],
            )
            resolved["beatPhaseSec"] = _clamp_number(
                phase_raw,
                -8.0,
                8.0,
                resolved["beatPhaseSec"],
            )
            zoom_factor_raw = _extract_effect_value(raw_effect, "zoomFactor", None)
            if zoom_factor_raw is None:
                zoom_factor_raw = _extract_effect_value(raw_effect, "zoom_factor", 1.0)
            shake_factor_raw = _extract_effect_value(raw_effect, "shakeFactor", None)
            if shake_factor_raw is None:
                shake_factor_raw = _extract_effect_value(
                    raw_effect, "shake_factor", 1.0
                )
            resolved["beatZoomFactor"] = _clamp_number(
                zoom_factor_raw,
                0.2,
                2.4,
                resolved["beatZoomFactor"],
            )
            resolved["beatShakeFactor"] = _clamp_number(
                shake_factor_raw,
                0.2,
                2.6,
                resolved["beatShakeFactor"],
            )

    return resolved


def _normalize_motion_preset(value: Any, default: str = "none") -> str:
    if not isinstance(value, str):
        return default
    preset = value.strip().lower().replace("_", "-")
    aliases = {
        "default": "dynamic",
        "auto": "dynamic",
        "pan": "dynamic",
        "zoomin": "push-in",
        "zoomout": "push-out",
        "driftleft": "drift-left",
        "driftright": "drift-right",
        "whipleft": "whip-left",
        "whipright": "whip-right",
    }
    preset = aliases.get(preset, preset)
    allowed = {
        "none",
        "dynamic",
        "push-in",
        "push-out",
        "drift-left",
        "drift-right",
        "whip-left",
        "whip-right",
        "impact",
    }
    if preset in allowed:
        return preset
    return default


def _build_clip_motion_video_filters(
    clip_effects: Dict[str, Any],
    clip_duration: float,
    width: int,
    height: int,
    clip_index: int,
) -> List[str]:
    preset = _normalize_motion_preset(clip_effects.get("motionPreset", "none"))
    intensity = _clamp_number(clip_effects.get("motionIntensity", 0.0), 0.0, 1.0, 0.0)
    speed = _clamp_number(clip_effects.get("motionSpeed", 1.0), 0.35, 3.0, 1.0)
    shake = _clamp_number(clip_effects.get("motionShake", 0.0), 0.0, 1.0, 0.0)
    zoom_boost = _clamp_number(clip_effects.get("zoomBoost", 0.0), 0.0, 0.45, 0.0)
    beat_bpm = _clamp_number(clip_effects.get("beatBpm", 0.0), 0.0, 220.0, 0.0)
    beat_amount = _clamp_number(clip_effects.get("beatAmount", 0.0), 0.0, 1.0, 0.0)
    beat_phase_sec = _clamp_number(
        clip_effects.get("beatPhaseSec", 0.0), -8.0, 8.0, 0.0
    )
    beat_zoom_factor = _clamp_number(
        clip_effects.get("beatZoomFactor", 1.0),
        0.2,
        2.4,
        1.0,
    )
    beat_shake_factor = _clamp_number(
        clip_effects.get("beatShakeFactor", 1.0),
        0.2,
        2.6,
        1.0,
    )

    if (
        preset == "none"
        and intensity <= 1e-6
        and shake <= 1e-6
        and zoom_boost <= 1e-6
        and beat_amount <= 1e-6
    ):
        return []

    if preset == "none":
        preset = "dynamic"

    seed = ((clip_index % 11) + 1) * 0.173
    duration_guard = max(0.28, clip_duration)
    motion_freq = 0.9 + speed * 1.35

    base_zoom = min(1.55, 1.06 + intensity * 0.18 + zoom_boost * 0.70)
    pulse_amp = min(0.11, 0.012 + intensity * 0.05 + zoom_boost * 0.22)

    center_x = "(in_w-out_w)/2"
    center_y = "(in_h-out_h)/2"
    drift_x = min(0.9, 0.28 + intensity * 0.34 + shake * 0.1)
    drift_y = min(0.85, 0.2 + intensity * 0.28 + shake * 0.08)

    zoom_expr = f"{base_zoom:.4f}+{pulse_amp:.4f}*sin((t+{seed:.3f})*{motion_freq:.3f})"
    x_expr = f"{center_x}+(in_w-out_w)*{drift_x:.3f}*0.40*sin((t+{seed:.3f})*{motion_freq * 1.40:.3f})"
    y_expr = f"{center_y}+(in_h-out_h)*{drift_y:.3f}*0.36*cos((t+{seed * 1.4:.3f})*{motion_freq * 1.15:.3f})"

    if preset == "push-in":
        growth = min(0.34, 0.10 + intensity * 0.22 + zoom_boost * 0.4)
        zoom_expr = f"1+{growth:.4f}*min(1,t/{duration_guard:.3f})"
        x_expr = f"{center_x}+(in_w-out_w)*{drift_x:.3f}*0.18*sin((t+{seed:.3f})*{motion_freq:.3f})"
        y_expr = f"{center_y}+(in_h-out_h)*{drift_y:.3f}*0.14*cos((t+{seed * 1.2:.3f})*{motion_freq * 0.95:.3f})"
    elif preset == "push-out":
        reduction = min(0.3, 0.08 + intensity * 0.2 + zoom_boost * 0.35)
        start_zoom = min(1.6, 1.14 + intensity * 0.2 + zoom_boost * 0.5)
        zoom_expr = f"{start_zoom:.4f}-{reduction:.4f}*min(1,t/{duration_guard:.3f})"
        x_expr = f"{center_x}+(in_w-out_w)*{drift_x:.3f}*0.14*sin((t+{seed:.3f})*{motion_freq:.3f})"
        y_expr = f"{center_y}+(in_h-out_h)*{drift_y:.3f}*0.12*cos((t+{seed * 1.2:.3f})*{motion_freq * 0.9:.3f})"
    elif preset == "drift-left":
        zoom_expr = f"{base_zoom:.4f}+{pulse_amp * 0.7:.4f}*sin((t+{seed:.3f})*{motion_freq:.3f})"
        x_expr = f"{center_x}+(in_w-out_w)*{drift_x:.3f}*0.58*(1-2*min(1,t/{duration_guard:.3f}))"
        y_expr = f"{center_y}+(in_h-out_h)*{drift_y:.3f}*0.18*sin((t+{seed * 1.2:.3f})*{motion_freq:.3f})"
    elif preset == "drift-right":
        zoom_expr = f"{base_zoom:.4f}+{pulse_amp * 0.7:.4f}*sin((t+{seed:.3f})*{motion_freq:.3f})"
        x_expr = f"{center_x}+(in_w-out_w)*{drift_x:.3f}*0.58*(2*min(1,t/{duration_guard:.3f})-1)"
        y_expr = f"{center_y}+(in_h-out_h)*{drift_y:.3f}*0.18*cos((t+{seed * 1.2:.3f})*{motion_freq:.3f})"
    elif preset in {"whip-left", "whip-right", "impact"}:
        whip_dir = -1 if preset == "whip-left" else 1
        if preset == "impact":
            whip_dir = -1 if (clip_index % 2 == 0) else 1
        zoom_expr = (
            f"{min(1.62, base_zoom + 0.14):.4f}+{min(0.13, pulse_amp + 0.04):.4f}"
            f"*sin((t+{seed:.3f})*{motion_freq * 1.65:.3f})"
        )
        x_expr = f"{center_x}+{whip_dir}*(in_w-out_w)*{drift_x:.3f}*0.62*(1-min(1,t/{max(0.24, duration_guard * 0.44):.3f}))"
        y_expr = f"{center_y}+(in_h-out_h)*{drift_y:.3f}*0.24*sin((t+{seed * 1.4:.3f})*{motion_freq * 1.5:.3f})"

    if shake > 1e-6:
        shake_x = min(0.22, 0.04 + shake * 0.14)
        shake_y = min(0.18, 0.03 + shake * 0.11)
        x_expr = f"({x_expr})+(in_w-out_w)*{shake_x:.3f}*sin((t+{seed:.3f})*38.0)"
        y_expr = f"({y_expr})+(in_h-out_h)*{shake_y:.3f}*cos((t+{seed * 1.1:.3f})*29.0)"

    if beat_bpm > 1e-6 and beat_amount > 1e-6:
        beat_omega = beat_bpm / 60.0 * 6.283185
        beat_zoom_amp = min(0.24, (0.03 + beat_amount * 0.1) * beat_zoom_factor)
        beat_shake_x = min(0.16, (0.015 + beat_amount * 0.05) * beat_shake_factor)
        beat_shake_y = min(0.12, (0.012 + beat_amount * 0.04) * beat_shake_factor)
        beat_pulse = (
            f"pow(max(0,sin((t+{seed + beat_phase_sec:.3f})*{beat_omega:.4f})),2.8)"
        )
        zoom_expr = f"({zoom_expr})+{beat_zoom_amp:.4f}*{beat_pulse}"
        x_expr = f"({x_expr})+(in_w-out_w)*{beat_shake_x:.3f}*{beat_pulse}"
        y_expr = f"({y_expr})+(in_h-out_h)*{beat_shake_y:.3f}*{beat_pulse}"

    filters: List[str] = [
        f"scale=w='trunc(iw*({zoom_expr})/2)*2':h='trunc(ih*({zoom_expr})/2)*2':eval=frame",
        f"crop={width}:{height}:x='max(0,min(in_w-out_w,{x_expr}))':y='max(0,min(in_h-out_h,{y_expr}))'",
    ]

    if preset in {"whip-left", "whip-right", "impact"}:
        filters.append("unsharp=5:5:0.460:5:5:0.000")

    return filters


def _as_dict_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, dict)]


def _extract_track_payload(
    track: Dict[str, Any], keys: List[str]
) -> List[Dict[str, Any]]:
    for key in keys:
        entries = _as_dict_list(track.get(key))
        if entries:
            return entries
    return []


def _collect_timeline_tracks(
    timeline_payload: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    video_clips: List[Dict[str, Any]] = []
    effect_entries: List[Dict[str, Any]] = []
    text_entries: List[Dict[str, Any]] = []

    tracks = timeline_payload.get("tracks")
    if isinstance(tracks, dict):
        video_clips.extend(_as_dict_list(tracks.get("video")))
        video_clips.extend(_as_dict_list(tracks.get("videos")))
        effect_entries.extend(_as_dict_list(tracks.get("effects")))
        text_entries.extend(_as_dict_list(tracks.get("text")))
        text_entries.extend(_as_dict_list(tracks.get("captions")))
        text_entries.extend(_as_dict_list(tracks.get("titles")))
        text_entries.extend(_as_dict_list(tracks.get("overlays")))
    elif isinstance(tracks, list):
        for raw_track in tracks:
            if not isinstance(raw_track, dict):
                continue

            kind_raw = (
                raw_track.get("type")
                or raw_track.get("trackType")
                or raw_track.get("track_type")
                or raw_track.get("kind")
                or raw_track.get("name")
                or ""
            )
            kind = str(kind_raw).strip().lower().replace("_", "-")

            track_video_clips = _extract_track_payload(
                raw_track,
                ["clips", "items", "segments", "video"],
            )
            track_effect_entries = _extract_track_payload(
                raw_track,
                ["effects", "clips", "items", "segments"],
            )
            track_text_entries = _extract_track_payload(
                raw_track,
                ["captions", "titles", "text", "overlays", "clips", "items"],
            )

            if kind in {"video", "videos", "media", "main"}:
                video_clips.extend(track_video_clips)
            elif kind in {"effect", "effects", "fx", "filters"}:
                effect_entries.extend(track_effect_entries)
            elif kind in {
                "text",
                "caption",
                "captions",
                "title",
                "titles",
                "overlay",
                "overlays",
                "graphics",
            }:
                text_entries.extend(track_text_entries)

    if not video_clips:
        video_clips.extend(
            _extract_track_payload(
                timeline_payload,
                ["video", "videos", "clips", "items", "segments"],
            )
        )

    if not effect_entries:
        effect_entries.extend(_as_dict_list(timeline_payload.get("effects")))

    if not text_entries:
        text_entries.extend(
            _as_dict_list(timeline_payload.get("textOverlays"))
            + _as_dict_list(timeline_payload.get("text_overlays"))
            + _as_dict_list(timeline_payload.get("captions"))
            + _as_dict_list(timeline_payload.get("titles"))
            + _as_dict_list(timeline_payload.get("overlays"))
        )

    return {
        "video": video_clips,
        "effects": effect_entries,
        "text": text_entries,
    }


def _extract_timeline_clips(
    timeline_payload: Any,
    input_path: str,
    fallback_input_urls: List[str],
    default_duration: float,
) -> List[Dict[str, Any]]:
    if not isinstance(timeline_payload, dict):
        return []

    collected_tracks = _collect_timeline_tracks(timeline_payload)
    raw_video_clips = collected_tracks["video"]
    if not raw_video_clips:
        return []

    effect_entries = collected_tracks["effects"]

    normalized: List[Dict[str, Any]] = []
    fallback_idx = 0

    for clip_index, raw_clip in enumerate(raw_video_clips):
        if not isinstance(raw_clip, dict):
            continue

        media_type = str(
            raw_clip.get("mediaType", raw_clip.get("media_type", "unknown"))
        ).lower()
        if media_type not in {"video", "image"}:
            continue

        state = str(raw_clip.get("state", "linked")).lower()
        source_url = raw_clip.get("sourceUrl", raw_clip.get("source_url"))
        if not isinstance(source_url, str) or not source_url.strip():
            source_url = None
        elif not _looks_like_url(source_url) and not os.path.exists(source_url):
            source_url = None

        if source_url is None and fallback_idx < len(fallback_input_urls):
            source_url = fallback_input_urls[fallback_idx]
            fallback_idx += 1

        if source_url is None and clip_index == 0:
            source_url = input_path

        if source_url is None:
            if state == "pending":
                continue
            continue

        start_sec = _clamp_number(
            raw_clip.get("startSec", raw_clip.get("start_sec", 0)), 0, 7200, 0
        )
        duration_sec = _clamp_number(
            raw_clip.get("durationSec", raw_clip.get("duration_sec", default_duration)),
            0.1,
            7200,
            default_duration,
        )
        trim_in_sec = _clamp_number(
            raw_clip.get("trimInSec", raw_clip.get("trim_in_sec", 0)), 0, 7200, 0
        )

        trim_out_raw = raw_clip.get("trimOutSec", raw_clip.get("trim_out_sec"))
        trim_out_sec: Optional[float] = None
        if trim_out_raw not in (None, ""):
            trim_out_sec = _clamp_number(trim_out_raw, 0, 7200, 0)
            if trim_out_sec <= trim_in_sec:
                trim_out_sec = None

        speed = _clamp_number(raw_clip.get("speed", 1), 0.25, 4.0, 1.0)

        clip_effect_stack: List[Dict[str, Any]] = []
        raw_clip_effect_stack = raw_clip.get(
            "effectStack", raw_clip.get("effect_stack")
        )
        if isinstance(raw_clip_effect_stack, list):
            for stack_effect in raw_clip_effect_stack:
                if isinstance(stack_effect, dict):
                    clip_effect_stack.append(stack_effect)

        clip_id = _safe_clip_name(raw_clip.get("id"), f"clip-{clip_index + 1}")
        for tracked_effect in effect_entries:
            target_id = tracked_effect.get("clipId", tracked_effect.get("clip_id"))
            if isinstance(target_id, str) and target_id.strip() == clip_id:
                clip_effect_stack.append(tracked_effect)

        transition_in_sec = _parse_transition_fade_sec(
            raw_clip.get("transitionIn")
            or raw_clip.get("transition_in")
            or raw_clip.get("inTransition")
            or raw_clip.get("in_transition")
        )
        transition_out_sec = _parse_transition_fade_sec(
            raw_clip.get("transitionOut")
            or raw_clip.get("transition_out")
            or raw_clip.get("outTransition")
            or raw_clip.get("out_transition")
        )

        normalized.append(
            {
                "id": clip_id,
                "label": _safe_clip_name(
                    raw_clip.get("label"), f"Clip {clip_index + 1}"
                ),
                "source": str(source_url),
                "mediaType": media_type,
                "startSec": start_sec,
                "durationSec": duration_sec,
                "trimInSec": trim_in_sec,
                "trimOutSec": trim_out_sec,
                "speed": speed,
                "effectStack": clip_effect_stack,
                "transitionInSec": transition_in_sec,
                "transitionOutSec": transition_out_sec,
            }
        )

    normalized.sort(key=lambda clip: float(clip["startSec"]))
    return normalized


def _extract_timeline_text_overlays(
    timeline_payload: Any,
    default_duration: float,
) -> List[Dict[str, Any]]:
    if not isinstance(timeline_payload, dict):
        return []

    collected_tracks = _collect_timeline_tracks(timeline_payload)
    raw_text_entries = collected_tracks.get("text") or []
    normalized: List[Dict[str, Any]] = []

    for index, raw_entry in enumerate(raw_text_entries):
        if not isinstance(raw_entry, dict):
            continue
        item = _normalize_text_overlay(raw_entry, index, default_duration)
        if item is not None:
            normalized.append(item)

    normalized.sort(key=lambda entry: float(entry["startSec"]))
    return normalized


def _estimate_timeline_duration(
    clips: List[Dict[str, Any]], default_duration: float
) -> float:
    estimated = max(0.1, default_duration)
    for clip in clips:
        try:
            start_sec = float(clip.get("startSec", 0.0))
            duration_sec = max(0.1, float(clip.get("durationSec", default_duration)))
        except (TypeError, ValueError):
            continue
        estimated = max(estimated, start_sec + duration_sec)
    return estimated


def _build_overlay_video_filters(
    text_overlays: List[Dict[str, Any]],
    width: int,
    height: int,
    timeline_duration: float,
    add_tiktok_chrome: bool,
) -> List[str]:
    filters: List[str] = []

    if add_tiktok_chrome:
        top_height = max(72, int(height * 0.1))
        bottom_height = max(92, int(height * 0.12))
        accent_height = max(8, int(height * 0.008))
        filters.append(
            f"drawbox=x=0:y=0:w=iw:h={top_height}:color=0x000000@0.38:t=fill"
        )
        filters.append(
            f"drawbox=x=0:y={height - bottom_height}:w=iw:h={bottom_height}:color=0x000000@0.32:t=fill"
        )
        filters.append(
            f"drawbox=x=0:y=0:w='iw*min(1,max(0,t/{max(0.1, timeline_duration):.3f}))':h={accent_height}:color=0x00e6ff@0.96:t=fill"
        )

    for overlay in text_overlays:
        text_value = str(overlay.get("text", "")).strip()
        if not text_value:
            continue

        start_sec = max(0.0, float(overlay.get("startSec", 0.0)))
        duration_sec = max(0.15, float(overlay.get("durationSec", 1.5)))
        end_sec = min(max(0.1, timeline_duration), start_sec + duration_sec)
        if end_sec <= start_sec:
            continue

        fade_in = min(0.35, max(0.06, duration_sec * 0.22))
        fade_out = min(0.35, max(0.08, duration_sec * 0.22))
        safe_fade_in = max(0.03, min(fade_in, duration_sec / 2.0))
        safe_fade_out = max(0.03, min(fade_out, duration_sec / 2.0))

        align = str(overlay.get("align", "center")).strip().lower()
        base_padding = int(max(18, min(96, width * 0.05)))
        if align == "left":
            base_x_expr = str(base_padding)
        elif align == "right":
            base_x_expr = f"w-text_w-{base_padding}"
        else:
            base_x_expr = "(w-text_w)/2"

        y_base = _text_position_y(
            str(overlay.get("position", "bottom")),
            height,
            float(overlay.get("yPercent", 0.8)),
        )

        motion_strength = _clamp_number(
            overlay.get("motionStrength", 0.85),
            0.0,
            1.0,
            0.85,
        )
        y_offset = int(40 + 80 * motion_strength)
        animation = str(overlay.get("animation", "pop")).strip().lower()
        x_expr = base_x_expr
        if animation in {"punch", "shake"}:
            x_shake = int(8 + 24 * motion_strength)
            x_expr = f"({base_x_expr})+{x_shake}*sin((t-{start_sec:.3f})*42)"

        if animation in {"slide-up", "slideup", "pop", "bounce", "punch"}:
            y_expr = (
                f"if(lt(t,{start_sec:.3f}),{y_base + y_offset},"
                f"if(lt(t,{start_sec + safe_fade_in:.3f}),"
                f"{y_base + y_offset}-({y_offset})*(t-{start_sec:.3f})/{safe_fade_in:.3f},"
                f"{y_base}))"
            )
        else:
            y_expr = str(y_base)

        alpha_expr = (
            f"if(lt(t,{start_sec:.3f}),0,"
            f"if(lt(t,{start_sec + safe_fade_in:.3f}),"
            f"(t-{start_sec:.3f})/{safe_fade_in:.3f},"
            f"if(lt(t,{end_sec - safe_fade_out:.3f}),1,"
            f"if(lt(t,{end_sec:.3f}),"
            f"({end_sec:.3f}-t)/{safe_fade_out:.3f},0))))"
        )

        uppercase = _as_bool(overlay.get("uppercase", False), default=False)
        if uppercase:
            text_value = text_value.upper()
        text_value = _escape_drawtext_text(text_value)

        font_size = int(_clamp_number(overlay.get("fontSize", 64), 20, 190, 64))
        captioner_tiktok_style = _is_captioner_tiktok_style(
            _normalize_text_style(overlay.get("style", "default"))
        ) or _as_bool(overlay.get("captionerTikTokStyle", False), default=False)

        color = _to_ffmpeg_color(
            _normalize_hex_color(
                overlay.get("color", "#000000" if captioner_tiktok_style else "#FFFFFF")
            )
        )
        stroke_color = _to_ffmpeg_color(
            _normalize_hex_color(overlay.get("strokeColor", "#000000"))
        )
        stroke_width = _clamp_number(
            overlay.get("strokeWidth", 0 if captioner_tiktok_style else 4),
            0.0,
            14.0,
            0 if captioner_tiktok_style else 4,
        )

        box_enabled = _as_bool(
            overlay.get("box", True if captioner_tiktok_style else True),
            default=True if captioner_tiktok_style else True,
        )
        box_color = _to_ffmpeg_color(
            _normalize_hex_color(
                overlay.get(
                    "boxColor", "#FFFFFF" if captioner_tiktok_style else "#000000"
                )
            )
        )
        box_opacity = _clamp_number(
            overlay.get("boxOpacity", 1.0 if captioner_tiktok_style else 0.38),
            0.0,
            1.0,
            1.0 if captioner_tiktok_style else 0.38,
        )
        default_box_border = (
            max(10, int(font_size * 0.24)) if captioner_tiktok_style else 12
        )
        box_border_w = int(
            _clamp_number(
                overlay.get("boxBorderW", default_box_border),
                0,
                52,
                default_box_border,
            )
        )

        fontfile = _resolve_fontfile(
            overlay.get("font"),
            bold=_as_bool(overlay.get("bold", True), default=True),
        )
        font_option = (
            f"fontfile='{_escape_drawtext_value(fontfile)}'"
            if isinstance(fontfile, str)
            else "font='Arial'"
        )

        drawtext_parts = [
            font_option,
            f"text='{text_value}'",
            f"fontsize={font_size}",
            f"fontcolor={color}",
            f"bordercolor={stroke_color}",
            f"borderw={stroke_width:.2f}",
            f"box={1 if box_enabled else 0}",
            f"boxcolor={box_color}@{box_opacity:.3f}",
            f"boxborderw={box_border_w}",
            f"x={x_expr}",
            f"y='{y_expr}'",
            f"alpha='{alpha_expr}'",
            f"line_spacing={6 if captioner_tiktok_style else 8}",
            f"enable='between(t,{start_sec:.3f},{end_sec:.3f})'",
        ]
        filters.append(f"drawtext={':'.join(drawtext_parts)}")

    return filters


def _render_timeline_video(
    output_path: str,
    clips: List[Dict[str, Any]],
    text_overlays: List[Dict[str, Any]],
    width: int,
    height: int,
    fps: int,
    scale_filter: str,
    video_codec: str,
    preset: str,
    quality_crf: int,
    brightness: float,
    contrast: float,
    saturation: float,
    gamma: float,
    hue_deg: float,
    grayscale: bool,
    blur_sigma: float,
    denoise_strength: float,
    rotate_deg: int,
    flip_horizontal: bool,
    flip_vertical: bool,
    sharpen_amount: float,
    fade_in_sec: float,
    fade_out_sec: float,
    mute: bool,
    volume: float,
    add_tiktok_chrome: bool,
    captioner_overlay_defaults: Dict[str, Any],
    global_beat_bpm: float,
    global_beat_amount: float,
    global_beat_phase_sec: float,
    global_beat_zoom_factor: float,
    global_beat_shake_factor: float,
) -> Dict[str, Any]:
    if not clips:
        raise ValueError("Cannot render timeline without clips")

    command: List[str] = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    filter_parts: List[str] = []
    concat_labels: List[str] = []
    audio_labels: List[str] = []
    overlay_image_paths: List[str] = []

    timeline_duration = 0.0
    fallback_cursor = 0.0

    for index, clip in enumerate(clips):
        clip_duration = max(0.1, float(clip["durationSec"]))
        clip_start = max(fallback_cursor, float(clip.get("startSec", fallback_cursor)))
        clip_effects = _resolve_clip_effects(clip.get("effectStack"), clip_duration)

        if global_beat_bpm > 0.0:
            if _clamp_number(clip_effects.get("beatBpm", 0.0), 0.0, 220.0, 0.0) <= 0.0:
                clip_effects["beatBpm"] = global_beat_bpm
            if _clamp_number(clip_effects.get("beatAmount", 0.0), 0.0, 1.0, 0.0) <= 0.0:
                clip_effects["beatAmount"] = global_beat_amount
            if (
                _clamp_number(clip_effects.get("beatZoomFactor", 1.0), 0.2, 2.4, 1.0)
                == 1.0
            ):
                clip_effects["beatZoomFactor"] = global_beat_zoom_factor
            if (
                _clamp_number(clip_effects.get("beatShakeFactor", 1.0), 0.2, 2.6, 1.0)
                == 1.0
            ):
                clip_effects["beatShakeFactor"] = global_beat_shake_factor

            beat_period = 60.0 / max(1e-6, global_beat_bpm)
            phase_for_clip = (global_beat_phase_sec + clip_start) % beat_period
            clip_effects["beatPhaseSec"] = phase_for_clip

        speed = max(
            0.25,
            min(
                4.0,
                float(clip.get("speed", 1.0)) * float(clip_effects.get("speed", 1.0)),
            ),
        )
        trim_in_sec = max(0.0, float(clip["trimInSec"]))
        trim_out_sec = clip["trimOutSec"]

        clip_transition_in_sec = _clamp_number(
            clip.get("transitionInSec", 0.0),
            0.0,
            min(5.0, max(0.0, clip_duration - 0.01)),
            0.0,
        )
        clip_transition_out_sec = _clamp_number(
            clip.get("transitionOutSec", 0.0),
            0.0,
            min(5.0, max(0.0, clip_duration - 0.01)),
            0.0,
        )

        clip_fade_in = max(
            _clamp_number(
                clip_effects.get("fadeInSec", 0.0),
                0.0,
                min(5.0, max(0.0, clip_duration - 0.01)),
                0.0,
            ),
            clip_transition_in_sec,
        )
        clip_fade_out = max(
            _clamp_number(
                clip_effects.get("fadeOutSec", 0.0),
                0.0,
                min(5.0, max(0.0, clip_duration - 0.01)),
                0.0,
            ),
            clip_transition_out_sec,
        )

        if trim_out_sec is None:
            source_span = max(0.1, clip_duration * speed)
            trim_end = trim_in_sec + source_span
        else:
            trim_end = max(trim_in_sec + 0.05, float(trim_out_sec))

        clip_video_filters: List[str] = []
        clip_audio_filters: List[str] = []
        clip_motion_filters = _build_clip_motion_video_filters(
            clip_effects=clip_effects,
            clip_duration=clip_duration,
            width=width,
            height=height,
            clip_index=index,
        )

        clip_rotate_deg = int(clip_effects.get("rotateDeg", 0))
        if clip_rotate_deg == 90:
            clip_video_filters.append("transpose=1")
        elif clip_rotate_deg == 180:
            clip_video_filters.extend(["hflip", "vflip"])
        elif clip_rotate_deg == 270:
            clip_video_filters.append("transpose=2")

        if _as_bool(clip_effects.get("flipHorizontal", False), default=False):
            clip_video_filters.append("hflip")
        if _as_bool(clip_effects.get("flipVertical", False), default=False):
            clip_video_filters.append("vflip")

        clip_brightness = _clamp_number(
            clip_effects.get("brightness", 0.0),
            -0.3,
            0.3,
            0.0,
        )
        clip_contrast = _clamp_number(clip_effects.get("contrast", 1.0), 0.5, 2.0, 1.0)
        clip_saturation = _clamp_number(
            clip_effects.get("saturation", 1.0),
            0.0,
            3.0,
            1.0,
        )
        clip_gamma = _clamp_number(clip_effects.get("gamma", 1.0), 0.5, 2.0, 1.0)
        if (
            abs(clip_brightness) > 1e-6
            or abs(clip_contrast - 1.0) > 1e-6
            or abs(clip_saturation - 1.0) > 1e-6
            or abs(clip_gamma - 1.0) > 1e-6
        ):
            clip_video_filters.append(
                f"eq=brightness={clip_brightness:.3f}:contrast={clip_contrast:.3f}:saturation={clip_saturation:.3f}:gamma={clip_gamma:.3f}"
            )

        clip_hue_deg = _clamp_number(
            clip_effects.get("hueDeg", 0.0), -180.0, 180.0, 0.0
        )
        if abs(clip_hue_deg) > 1e-6:
            clip_video_filters.append(f"hue=h={clip_hue_deg:.2f}")
        if _as_bool(clip_effects.get("grayscale", False), default=False):
            clip_video_filters.append("hue=s=0")

        clip_blur_sigma = _clamp_number(
            clip_effects.get("blurSigma", 0.0), 0.0, 8.0, 0.0
        )
        if clip_blur_sigma > 1e-6:
            clip_video_filters.append(f"gblur=sigma={clip_blur_sigma:.2f}")

        clip_denoise_strength = _clamp_number(
            clip_effects.get("denoiseStrength", 0.0),
            0.0,
            10.0,
            0.0,
        )
        if clip_denoise_strength > 1e-6:
            luma_spatial = max(0.1, clip_denoise_strength)
            chroma_spatial = max(0.1, clip_denoise_strength * 0.75)
            luma_tmp = max(0.1, clip_denoise_strength * 1.5)
            chroma_tmp = max(0.1, clip_denoise_strength * 1.125)
            clip_video_filters.append(
                f"hqdn3d={luma_spatial:.2f}:{chroma_spatial:.2f}:{luma_tmp:.2f}:{chroma_tmp:.2f}"
            )

        clip_sharpen_amount = _clamp_number(
            clip_effects.get("sharpenAmount", 0.0),
            0.0,
            2.0,
            0.0,
        )
        if clip_sharpen_amount > 1e-6:
            clip_video_filters.append(
                f"unsharp=5:5:{clip_sharpen_amount:.3f}:5:5:0.000"
            )

        if clip_fade_in > 1e-6:
            clip_video_filters.append(f"fade=t=in:st=0:d={clip_fade_in:.3f}")
            clip_audio_filters.append(f"afade=t=in:st=0:d={clip_fade_in:.3f}")
        if clip_fade_out > 1e-6:
            clip_fade_out_start = max(0.0, clip_duration - clip_fade_out)
            if clip_fade_out_start < clip_duration:
                clip_video_filters.append(
                    f"fade=t=out:st={clip_fade_out_start:.3f}:d={clip_fade_out:.3f}"
                )
                clip_audio_filters.append(
                    f"afade=t=out:st={clip_fade_out_start:.3f}:d={clip_fade_out:.3f}"
                )

        clip_mute = _as_bool(clip_effects.get("mute", False), default=False)
        clip_volume = _clamp_number(clip_effects.get("volume", 1.0), 0.0, 2.0, 1.0)
        if not clip_mute and abs(clip_volume - 1.0) > 1e-6:
            clip_audio_filters.append(f"volume={clip_volume:.3f}")

        if clip.get("mediaType") == "image":
            image_read_duration = max(clip_duration, (trim_end - trim_in_sec) + 0.1)
            command.extend(
                [
                    "-loop",
                    "1",
                    "-t",
                    f"{image_read_duration:.3f}",
                    "-i",
                    str(clip["source"]),
                ]
            )
            clip_filter_parts = [f"[{index}:v]{scale_filter}"]
            clip_filter_parts.extend(clip_motion_filters)
            clip_filter_parts.extend(clip_video_filters)
            clip_filter_parts.extend(
                [
                    "format=pix_fmts=yuv420p",
                    "setsar=1",
                    f"fps={fps}",
                    f"trim=duration={clip_duration:.3f}",
                    "setpts=PTS-STARTPTS",
                ]
            )
            clip_filter = ",".join(clip_filter_parts) + f"[v{index}]"
            audio_filter = (
                f"anullsrc=r=48000:cl=stereo,atrim=duration={clip_duration:.3f},"
                f"asetpts=PTS-STARTPTS[a{index}]"
            )
        else:
            command.extend(["-i", str(clip["source"])])
            clip_filter_parts = [
                f"[{index}:v]trim=start={trim_in_sec:.3f}:end={trim_end:.3f}",
                "setpts=PTS-STARTPTS",
            ]
            if abs(speed - 1.0) > 1e-6:
                clip_filter_parts.append(f"setpts=PTS/{speed:.6f}")
            clip_filter_parts.append(scale_filter)
            clip_filter_parts.extend(clip_motion_filters)
            clip_filter_parts.extend(clip_video_filters)
            clip_filter_parts.extend(
                [
                    "format=pix_fmts=yuv420p",
                    "setsar=1",
                    f"fps={fps}",
                    f"trim=duration={clip_duration:.3f}",
                    "setpts=PTS-STARTPTS",
                ]
            )
            clip_filter = ",".join(clip_filter_parts) + f"[v{index}]"

            if _has_audio_stream(str(clip["source"])):
                audio_filter_parts = [
                    f"[{index}:a]atrim=start={trim_in_sec:.3f}:end={trim_end:.3f}",
                    "asetpts=PTS-STARTPTS",
                ]
                if abs(speed - 1.0) > 1e-6:
                    for atempo_filter in _build_atempo_filters(speed):
                        audio_filter_parts.append(atempo_filter)
                if clip_mute:
                    audio_filter_parts.append("volume=0")
                audio_filter_parts.extend(clip_audio_filters)
                audio_filter_parts.extend(
                    [
                        f"atrim=duration={clip_duration:.3f}",
                        "asetpts=PTS-STARTPTS",
                    ]
                )
                audio_filter = ",".join(audio_filter_parts) + f"[a{index}]"
            else:
                audio_filter = (
                    f"anullsrc=r=48000:cl=stereo,atrim=duration={clip_duration:.3f},"
                    f"asetpts=PTS-STARTPTS[a{index}]"
                )

        filter_parts.append(clip_filter)
        filter_parts.append(audio_filter)
        concat_labels.append(f"[v{index}]")
        audio_labels.append(f"[a{index}]")

        timeline_duration = max(timeline_duration, clip_start + clip_duration)
        fallback_cursor = clip_start + clip_duration

    # Build transition chain if we have multiple clips
    if len(clips) > 1 and any(
        _parse_transition(c.get("transitionIn") or c.get("transitionOut")).get("type")
        != "none"
        for c in clips
    ):
        # Use xfade for video transitions with effects
        v_labels = concat_labels.copy()
        a_labels = audio_labels.copy()

        for i in range(len(clips) - 1):
            transition = _parse_transition(clips[i + 1].get("transitionIn"))
            if transition.get("type") == "none":
                transition = _parse_transition(clips[i].get("transitionOut"))

            if (
                transition.get("type") != "none"
                and transition.get("durationSec", 0) > 0
            ):
                duration = transition["durationSec"]
                trans_type = _get_xfade_transition_name(
                    transition["type"], transition.get("direction", "left")
                )

                # Apply xfade between current and next clip
                next_v = f"[v{i + 1}]"
                next_a = f"[a{i + 1}]"

                # Build xfade filter
                v_out = f"[vt{i}]"
                a_out = f"[at{i}]"

                # Apply xfade with format filter to prevent pixel format issues
                filter_parts.append(
                    f"{v_labels[i]}{next_v}xfade=transition={trans_type}:duration={duration:.3f},"
                    f"format=pix_fmts=yuv420p{v_out}"
                )

                # For audio, use acrossfade
                filter_parts.append(
                    f"{a_labels[i]}{next_a}acrossfade=d={duration:.3f}{a_out}"
                )

                # Update labels for next iteration
                v_labels[i + 1] = v_out
                a_labels[i + 1] = a_out
            else:
                # No transition, just concat these two
                v_out = f"[vc{i}]"
                a_out = f"[ac{i}]"
                filter_parts.append(
                    f"{v_labels[i]}{v_labels[i + 1]}concat=n=2:v=1:a=0{v_out}"
                )
                filter_parts.append(
                    f"{a_labels[i]}{a_labels[i + 1]}concat=n=2:v=0:a=1{a_out}"
                )
                v_labels[i + 1] = v_out
                a_labels[i + 1] = a_out

        # Final output label
        filter_parts.append(f"{v_labels[-1]}setpts=PTS-STARTPTS[vcat]")
        if a_labels:
            filter_parts.append(f"{a_labels[-1]}asetpts=PTS-STARTPTS[acat]")
    else:
        # Simple concat for single clip or no transitions
        filter_parts.append(
            f"{''.join(concat_labels)}concat=n={len(concat_labels)}:v=1:a=0[vcat]"
        )
        if audio_labels:
            filter_parts.append(
                f"{''.join(audio_labels)}concat=n={len(audio_labels)}:v=0:a=1[acat]"
            )

    post_filters: List[str] = []
    if rotate_deg == 90:
        post_filters.append("transpose=1")
    elif rotate_deg == 180:
        post_filters.extend(["hflip", "vflip"])
    elif rotate_deg == 270:
        post_filters.append("transpose=2")

    if flip_horizontal:
        post_filters.append("hflip")
    if flip_vertical:
        post_filters.append("vflip")

    if (
        abs(brightness) > 1e-6
        or abs(contrast - 1.0) > 1e-6
        or abs(saturation - 1.0) > 1e-6
        or abs(gamma - 1.0) > 1e-6
    ):
        post_filters.append(
            f"eq=brightness={brightness:.3f}:contrast={contrast:.3f}:saturation={saturation:.3f}:gamma={gamma:.3f}"
        )

    if abs(hue_deg) > 1e-6:
        post_filters.append(f"hue=h={hue_deg:.2f}")

    if grayscale:
        post_filters.append("hue=s=0")

    if blur_sigma > 1e-6:
        post_filters.append(f"gblur=sigma={blur_sigma:.2f}")

    if denoise_strength > 1e-6:
        luma_spatial = max(0.1, denoise_strength)
        chroma_spatial = max(0.1, denoise_strength * 0.75)
        luma_tmp = max(0.1, denoise_strength * 1.5)
        chroma_tmp = max(0.1, denoise_strength * 1.125)
        post_filters.append(
            f"hqdn3d={luma_spatial:.2f}:{chroma_spatial:.2f}:{luma_tmp:.2f}:{chroma_tmp:.2f}"
        )

    if sharpen_amount > 1e-6:
        post_filters.append(f"unsharp=5:5:{sharpen_amount:.3f}:5:5:0.000")

    safe_fade_in = min(fade_in_sec, max(0.0, timeline_duration - 0.01))
    if safe_fade_in > 1e-6:
        post_filters.append(f"fade=t=in:st=0:d={safe_fade_in:.3f}")

    safe_fade_out = min(fade_out_sec, max(0.0, timeline_duration - 0.01))
    if safe_fade_out > 1e-6:
        fade_out_start = max(0.0, timeline_duration - safe_fade_out)
        if fade_out_start < timeline_duration:
            post_filters.append(
                f"fade=t=out:st={fade_out_start:.3f}:d={safe_fade_out:.3f}"
            )

    drawtext_overlays: List[Dict[str, Any]] = []
    captioner_tiktok_overlays: List[Dict[str, Any]] = []
    for overlay in text_overlays:
        style = _normalize_text_style(overlay.get("style", "default"))
        if _is_captioner_tiktok_style(style) or _as_bool(
            overlay.get("captionerTikTokStyle", False),
            default=False,
        ):
            captioner_tiktok_overlays.append(overlay)
        else:
            drawtext_overlays.append(overlay)

    rendered_captioner_overlays: List[Dict[str, Any]] = []
    overlay_input_start_index = len(clips)
    for overlay in captioner_tiktok_overlays:
        fd, overlay_path = tempfile.mkstemp(prefix="editor_txt_", suffix=".png")
        os.close(fd)

        if not _render_captioner_tiktok_overlay_png(
            output_png_path=overlay_path,
            width=width,
            height=height,
            overlay=overlay,
            captioner_defaults=captioner_overlay_defaults,
        ):
            try:
                os.remove(overlay_path)
            except OSError:
                pass
            raise RuntimeError(
                "Failed to render captioner-style TikTok overlay. "
                "Ensure fonts/TikTokBold.otf exists and is a valid font file."
            )

        overlay_image_paths.append(overlay_path)
        rendered_captioner_overlays.append(overlay)
        command.extend(
            [
                "-loop",
                "1",
                "-t",
                f"{max(0.1, timeline_duration):.3f}",
                "-i",
                overlay_path,
            ]
        )

    overlay_filters = _build_overlay_video_filters(
        text_overlays=drawtext_overlays,
        width=width,
        height=height,
        timeline_duration=max(0.1, timeline_duration),
        add_tiktok_chrome=add_tiktok_chrome,
    )
    if overlay_filters:
        post_filters.extend(overlay_filters)

    post_filters.append(f"fps={fps}")
    filter_parts.append(f"[vcat]{','.join(post_filters)}[vbase]")

    current_video_label = "vbase"
    for overlay_idx, overlay in enumerate(rendered_captioner_overlays):
        input_index = overlay_input_start_index + overlay_idx
        start_sec = max(0.0, float(overlay.get("startSec", 0.0)))
        duration_sec = max(0.15, float(overlay.get("durationSec", 1.5)))

        fade_in = min(0.35, max(0.06, duration_sec * 0.22))
        fade_out = min(0.35, max(0.08, duration_sec * 0.22))
        safe_fade_in = max(0.03, min(fade_in, duration_sec / 2.0))
        safe_fade_out = max(0.03, min(fade_out, duration_sec / 2.0))
        fade_out_start = max(0.0, duration_sec - safe_fade_out)

        base_x_expr = "0"
        y_base = 0

        motion_strength = _clamp_number(
            overlay.get("motionStrength", 0.85),
            0.0,
            1.0,
            0.85,
        )
        y_offset = int(40 + 80 * motion_strength)
        animation = str(overlay.get("animation", "pop")).strip().lower()
        x_expr = base_x_expr
        if animation in {"punch", "shake"}:
            x_shake = int(8 + 24 * motion_strength)
            x_expr = f"{x_shake}*sin((t-{start_sec:.3f})*42)"

        if animation in {"slide-up", "slideup", "pop", "bounce", "punch"}:
            y_expr = (
                f"if(lt(t,{start_sec:.3f}),{y_base + y_offset},"
                f"if(lt(t,{start_sec + safe_fade_in:.3f}),"
                f"{y_base + y_offset}-({y_offset})*(t-{start_sec:.3f})/{safe_fade_in:.3f},"
                f"{y_base}))"
            )
        else:
            y_expr = str(y_base)

        timed_label = f"ovtimed{overlay_idx}"
        out_label = f"vov{overlay_idx}"
        filter_parts.append(
            f"[{input_index}:v]format=rgba,trim=duration={duration_sec:.3f},"
            f"fade=t=in:st=0:d={safe_fade_in:.3f}:alpha=1,"
            f"fade=t=out:st={fade_out_start:.3f}:d={safe_fade_out:.3f}:alpha=1,"
            f"setpts=PTS+{start_sec:.3f}/TB[{timed_label}]"
        )
        filter_parts.append(
            f"[{current_video_label}][{timed_label}]"
            f"overlay=x='{x_expr}':y='{y_expr}':eof_action=pass:shortest=0[{out_label}]"
        )
        current_video_label = out_label

    filter_parts.append(f"[{current_video_label}]null[vout]")

    if mute:
        filter_parts.append(
            f"anullsrc=r=48000:cl=stereo,atrim=duration={max(0.1, timeline_duration):.3f},"
            "asetpts=PTS-STARTPTS[aout]"
        )
    elif audio_labels:
        audio_post_filters: List[str] = [
            "aresample=async=1:min_hard_comp=0.100:first_pts=0"
        ]
        if abs(volume - 1.0) > 1e-6:
            audio_post_filters.append(f"volume={volume:.3f}")
        filter_parts.append(f"[acat]{','.join(audio_post_filters)}[aout]")
    else:
        filter_parts.append(
            f"anullsrc=r=48000:cl=stereo,atrim=duration={max(0.1, timeline_duration):.3f},"
            "asetpts=PTS-STARTPTS[aout]"
        )

    command.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-t",
            f"{max(0.1, timeline_duration):.3f}",
            "-s",
            f"{width}x{height}",
            "-c:v",
            video_codec,
            "-preset",
            preset,
            "-crf",
            str(quality_crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            output_path,
        ]
    )

    try:
        _run_ffmpeg(command)
    finally:
        for overlay_path in overlay_image_paths:
            try:
                os.remove(overlay_path)
            except OSError:
                pass

    return {
        "clipCount": len(clips),
        "durationSec": max(0.1, timeline_duration),
        "textOverlayCount": len(text_overlays),
        "tiktokAdsChrome": add_tiktok_chrome,
    }


def process_editor(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """Process media using editor config and render to MP4."""

    def report(progress: float, message: str) -> None:
        if progress_callback:
            progress_callback(progress, message)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    report(0.05, "Preparing editor render...")

    mode = str(_cfg(config, "mode", default="simple")).lower()
    timeline_json = _cfg(config, "timelineJson", "timeline_json", default="")
    timeline_payload = None
    if isinstance(timeline_json, str) and timeline_json.strip():
        try:
            timeline_payload = json.loads(timeline_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid timeline JSON: {exc.msg}") from exc
    elif isinstance(timeline_json, dict):
        timeline_payload = timeline_json

    if mode != "timeline" and isinstance(timeline_payload, dict):
        mode = "timeline"

    fallback_input_urls_raw = _cfg(config, "inputUrls", "input_urls", default=[])
    fallback_input_urls: List[str] = []
    if isinstance(fallback_input_urls_raw, list):
        for item in fallback_input_urls_raw:
            if not isinstance(item, str):
                continue
            candidate = item.strip()
            if not candidate:
                continue
            if _looks_like_url(candidate) or os.path.exists(candidate):
                fallback_input_urls.append(candidate)

    width = int(_clamp_number(_cfg(config, "width", default=1080), 256, 3840, 1080))
    height = int(_clamp_number(_cfg(config, "height", default=1920), 256, 3840, 1920))
    fps = int(_clamp_number(_cfg(config, "fps", default=30), 12, 60, 30))

    duration_sec = _clamp_number(
        _cfg(config, "durationSec", "duration_sec", default=8),
        1,
        600,
        8,
    )
    trim_start_sec = _clamp_number(
        _cfg(config, "trimStartSec", "trim_start_sec", default=0),
        0,
        600,
        0,
    )

    trim_end_raw = _cfg(config, "trimEndSec", "trim_end_sec", default=None)
    trim_end_sec = None
    if trim_end_raw not in (None, ""):
        trim_end_sec = _clamp_number(trim_end_raw, 0, 600, 0)
        if trim_end_sec <= trim_start_sec:
            trim_end_sec = None

    playback_rate = _clamp_number(
        _cfg(config, "playbackRate", "playback_rate", default=1),
        0.25,
        4.0,
        1.0,
    )
    mute = _as_bool(_cfg(config, "mute", default=False), default=False)
    volume = _clamp_number(_cfg(config, "volume", default=1), 0.0, 2.0, 1.0)

    brightness = _clamp_number(_cfg(config, "brightness", default=0), -0.3, 0.3, 0.0)
    contrast = _clamp_number(_cfg(config, "contrast", default=1), 0.5, 2.0, 1.0)
    saturation = _clamp_number(_cfg(config, "saturation", default=1), 0.0, 3.0, 1.0)
    gamma = _clamp_number(_cfg(config, "gamma", default=1), 0.5, 2.0, 1.0)

    hue_deg = _clamp_number(
        _cfg(config, "hueDeg", "hue_deg", default=0),
        -180.0,
        180.0,
        0.0,
    )
    grayscale = _as_bool(_cfg(config, "grayscale", default=False), default=False)
    blur_sigma = _clamp_number(
        _cfg(config, "blurSigma", "blur_sigma", default=0),
        0.0,
        8.0,
        0.0,
    )
    denoise_strength = _clamp_number(
        _cfg(config, "denoiseStrength", "denoise_strength", default=0),
        0.0,
        10.0,
        0.0,
    )

    rotate_raw = _cfg(config, "rotateDeg", "rotate_deg", default=0)
    try:
        rotate_deg = int(round(float(rotate_raw)))
    except (TypeError, ValueError):
        rotate_deg = 0
    if rotate_deg not in {0, 90, 180, 270}:
        rotate_deg = 0

    flip_horizontal = _as_bool(
        _cfg(config, "flipHorizontal", "flip_horizontal", default=False),
        default=False,
    )
    flip_vertical = _as_bool(
        _cfg(config, "flipVertical", "flip_vertical", default=False),
        default=False,
    )

    sharpen_amount = _clamp_number(
        _cfg(config, "sharpenAmount", "sharpen_amount", default=0),
        0.0,
        2.0,
        0.0,
    )

    fade_in_sec = _clamp_number(
        _cfg(config, "fadeInSec", "fade_in_sec", default=0),
        0.0,
        10.0,
        0.0,
    )
    fade_out_sec = _clamp_number(
        _cfg(config, "fadeOutSec", "fade_out_sec", default=0),
        0.0,
        10.0,
        0.0,
    )

    audio_normalize = _as_bool(
        _cfg(config, "audioNormalize", "audio_normalize", default=False),
        default=False,
    )
    audio_fade_in_sec = _clamp_number(
        _cfg(
            config,
            "audioFadeInSec",
            "audio_fade_in_sec",
            default=fade_in_sec,
        ),
        0.0,
        10.0,
        fade_in_sec,
    )
    audio_fade_out_sec = _clamp_number(
        _cfg(
            config,
            "audioFadeOutSec",
            "audio_fade_out_sec",
            default=fade_out_sec,
        ),
        0.0,
        10.0,
        fade_out_sec,
    )

    highpass_raw = _cfg(config, "audioHighpassHz", "audio_highpass_hz", default=0)
    audio_highpass_hz = 0
    if highpass_raw not in (None, "", 0, "0"):
        audio_highpass_hz = int(_clamp_number(highpass_raw, 20, 300, 80))

    lowpass_raw = _cfg(config, "audioLowpassHz", "audio_lowpass_hz", default=0)
    audio_lowpass_hz = 0
    if lowpass_raw not in (None, "", 0, "0"):
        audio_lowpass_hz = int(_clamp_number(lowpass_raw, 1000, 20000, 12000))

    if audio_lowpass_hz and audio_highpass_hz and audio_lowpass_hz <= audio_highpass_hz:
        adjusted_lowpass = min(20000, audio_highpass_hz + 1000)
        audio_lowpass_hz = (
            adjusted_lowpass if adjusted_lowpass > audio_highpass_hz else 0
        )

    fit = str(_cfg(config, "fit", default="cover")).lower()
    if fit not in {"cover", "contain", "stretch"}:
        fit = "cover"

    background_color = _normalize_hex_color(
        _cfg(config, "backgroundColor", "background_color", default="#000000")
    )
    ffmpeg_bg_color = _to_ffmpeg_color(background_color)

    codec = str(_cfg(config, "videoCodec", "video_codec", default="h264")).lower()
    video_codec = "libx265" if codec in {"h265", "hevc", "libx265"} else "libx264"
    quality_crf = int(
        _clamp_number(_cfg(config, "qualityCrf", "quality_crf", default=20), 16, 35, 20)
    )

    preset = str(_cfg(config, "preset", default="medium")).lower()
    if preset not in VALID_PRESETS:
        preset = "medium"

    if fit == "cover":
        scale_filter = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}"
        )
    elif fit == "contain":
        scale_filter = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={ffmpeg_bg_color}"
        )
    else:
        scale_filter = f"scale={width}:{height}"

    clip_duration = duration_sec
    if trim_end_sec is not None and trim_end_sec > trim_start_sec:
        clip_duration = trim_end_sec - trim_start_sec

    render_duration = clip_duration
    if playback_rate > 0:
        render_duration = clip_duration / playback_rate
    render_duration = max(0.01, render_duration)

    video_filters: List[str] = []

    if rotate_deg == 90:
        video_filters.append("transpose=1")
    elif rotate_deg == 180:
        video_filters.extend(["hflip", "vflip"])
    elif rotate_deg == 270:
        video_filters.append("transpose=2")

    if flip_horizontal:
        video_filters.append("hflip")
    if flip_vertical:
        video_filters.append("vflip")

    if (
        abs(brightness) > 1e-6
        or abs(contrast - 1.0) > 1e-6
        or abs(saturation - 1.0) > 1e-6
        or abs(gamma - 1.0) > 1e-6
    ):
        video_filters.append(
            f"eq=brightness={brightness:.3f}:contrast={contrast:.3f}:saturation={saturation:.3f}:gamma={gamma:.3f}"
        )

    if abs(hue_deg) > 1e-6:
        video_filters.append(f"hue=h={hue_deg:.2f}")

    if grayscale:
        video_filters.append("hue=s=0")

    if blur_sigma > 1e-6:
        video_filters.append(f"gblur=sigma={blur_sigma:.2f}")

    if denoise_strength > 1e-6:
        luma_spatial = max(0.1, denoise_strength)
        chroma_spatial = max(0.1, denoise_strength * 0.75)
        luma_tmp = max(0.1, denoise_strength * 1.5)
        chroma_tmp = max(0.1, denoise_strength * 1.125)
        video_filters.append(
            f"hqdn3d={luma_spatial:.2f}:{chroma_spatial:.2f}:{luma_tmp:.2f}:{chroma_tmp:.2f}"
        )

    if sharpen_amount > 1e-6:
        video_filters.append(f"unsharp=5:5:{sharpen_amount:.3f}:5:5:0.000")

    video_filters.append(scale_filter)
    if abs(playback_rate - 1.0) > 1e-6:
        video_filters.append(f"setpts=PTS/{playback_rate:.6f}")

    if fade_in_sec > 1e-6:
        safe_fade_in = min(fade_in_sec, max(0.0, render_duration - 0.01))
        if safe_fade_in > 1e-6:
            video_filters.append(f"fade=t=in:st=0:d={safe_fade_in:.3f}")

    if fade_out_sec > 1e-6:
        safe_fade_out = min(fade_out_sec, max(0.0, render_duration - 0.01))
        fade_out_start = max(0.0, render_duration - safe_fade_out)
        if safe_fade_out > 1e-6 and fade_out_start < render_duration:
            video_filters.append(
                f"fade=t=out:st={fade_out_start:.3f}:d={safe_fade_out:.3f}"
            )

    video_filters.append(f"fps={fps}")
    vf = ",".join(video_filters)

    def _build_audio_filters() -> List[str]:
        audio_filters: List[str] = []

        if abs(playback_rate - 1.0) > 1e-6:
            audio_filters.extend(_build_atempo_filters(playback_rate))
        if abs(volume - 1.0) > 1e-6:
            audio_filters.append(f"volume={volume:.3f}")

        if audio_highpass_hz > 0:
            audio_filters.append(f"highpass=f={audio_highpass_hz}")
        if audio_lowpass_hz > 0:
            audio_filters.append(f"lowpass=f={audio_lowpass_hz}")

        if audio_normalize:
            audio_filters.append("loudnorm=I=-16:LRA=11:TP=-1.5")

        if audio_fade_in_sec > 1e-6:
            safe_audio_fade_in = min(
                audio_fade_in_sec, max(0.0, render_duration - 0.01)
            )
            if safe_audio_fade_in > 1e-6:
                audio_filters.append(f"afade=t=in:st=0:d={safe_audio_fade_in:.3f}")

        if audio_fade_out_sec > 1e-6:
            safe_audio_fade_out = min(
                audio_fade_out_sec,
                max(0.0, render_duration - 0.01),
            )
            audio_fade_out_start = max(0.0, render_duration - safe_audio_fade_out)
            if safe_audio_fade_out > 1e-6 and audio_fade_out_start < render_duration:
                audio_filters.append(
                    f"afade=t=out:st={audio_fade_out_start:.3f}:d={safe_audio_fade_out:.3f}"
                )

        return audio_filters

    timeline_clips = _extract_timeline_clips(
        timeline_payload,
        input_path,
        fallback_input_urls,
        clip_duration,
    )

    estimated_timeline_duration = _estimate_timeline_duration(
        timeline_clips,
        clip_duration,
    )
    timeline_text_overlays = _extract_timeline_text_overlays(
        timeline_payload,
        estimated_timeline_duration,
    )
    config_text_overlays = _extract_text_overlays_from_config(
        config,
        estimated_timeline_duration,
    )
    tiktok_ads_overlays = _build_tiktok_ads_overlays(
        config,
        estimated_timeline_duration,
    )
    all_text_overlays = (
        timeline_text_overlays + config_text_overlays + tiktok_ads_overlays
    )
    all_text_overlays = _expand_word_by_word_overlays(all_text_overlays)
    all_text_overlays.sort(key=lambda item: float(item.get("startSec", 0.0)))

    tiktok_chrome_enabled = _as_bool(
        _cfg(
            config,
            "tiktokChromeEnabled",
            "tiktok_chrome_enabled",
            "tiktokAdsEnabled",
            "tiktok_ads_enabled",
            default=False,
        ),
        default=False,
    ) or bool(tiktok_ads_overlays)

    style_preset = (
        str(_cfg(config, "stylePreset", "style_preset", default="none")).strip().lower()
    )
    default_beat_detection = style_preset in {
        "tiktok-ads-v7",
        "tiktok_ads_v7",
        "tiktok-v7",
        "remotion-v7",
    }
    enable_audio_beat_detect = _as_bool(
        _cfg(
            config,
            "audioBeatDetect",
            "audio_beat_detect",
            default=default_beat_detection,
        ),
        default=default_beat_detection,
    )

    detected_beat_profile = {
        "bpm": 0.0,
        "amount": 0.0,
        "phaseSec": 0.0,
        "confidence": 0.0,
        "lowBandEnergy": 0.0,
        "highBandEnergy": 0.0,
        "transientDensity": 0.0,
    }
    if mode == "timeline" and timeline_clips and enable_audio_beat_detect:
        report(0.16, "Analyzing audio rhythm...")
        detected_beat_profile = _detect_audio_rhythm_from_timeline_sources(
            input_path=input_path,
            clips=timeline_clips,
        )

    manual_beat_bpm = _clamp_number(
        _cfg(config, "audioBeatBpm", "audio_beat_bpm", default=0.0),
        0.0,
        220.0,
        0.0,
    )
    manual_beat_amount = _clamp_number(
        _cfg(config, "audioBeatAmount", "audio_beat_amount", default=0.0),
        0.0,
        1.0,
        0.0,
    )
    manual_beat_phase = _clamp_number(
        _cfg(config, "audioBeatPhaseSec", "audio_beat_phase_sec", default=0.0),
        -8.0,
        8.0,
        0.0,
    )

    global_beat_bpm = (
        manual_beat_bpm if manual_beat_bpm > 0.0 else detected_beat_profile["bpm"]
    )
    global_beat_amount = (
        manual_beat_amount
        if manual_beat_amount > 0.0
        else detected_beat_profile["amount"]
    )
    global_beat_phase = detected_beat_profile["phaseSec"] + manual_beat_phase
    global_beat_zoom_factor = float(
        _clamp_number(
            0.85 + float(detected_beat_profile.get("lowBandEnergy", 0.0)) * 0.95,
            0.65,
            2.2,
            1.0,
        )
    )
    global_beat_shake_factor = float(
        _clamp_number(
            0.75
            + float(detected_beat_profile.get("highBandEnergy", 0.0)) * 0.95
            + float(detected_beat_profile.get("transientDensity", 0.0)) * 0.65,
            0.65,
            2.4,
            1.0,
        )
    )
    beat_fallback_used = False
    if enable_audio_beat_detect and global_beat_bpm <= 0.0 and default_beat_detection:
        global_beat_bpm = float(
            _clamp_number(
                _cfg(
                    config,
                    "audioBeatFallbackBpm",
                    "audio_beat_fallback_bpm",
                    default=132,
                ),
                80.0,
                180.0,
                132.0,
            )
        )
        fallback_amount = float(
            _clamp_number(
                _cfg(
                    config,
                    "audioBeatFallbackAmount",
                    "audio_beat_fallback_amount",
                    default=0.34,
                ),
                0.12,
                0.95,
                0.34,
            )
        )
        if global_beat_amount <= 0.0:
            global_beat_amount = fallback_amount
        beat_fallback_used = True

    captioner_overlay_defaults = _extract_editor_captioner_defaults(config)

    if mode == "timeline" and timeline_clips:
        report(0.2, "Running timeline render...")
        timeline_result = _render_timeline_video(
            output_path=output_path,
            clips=timeline_clips,
            text_overlays=all_text_overlays,
            width=width,
            height=height,
            fps=fps,
            scale_filter=scale_filter,
            video_codec=video_codec,
            preset=preset,
            quality_crf=quality_crf,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            gamma=gamma,
            hue_deg=hue_deg,
            grayscale=grayscale,
            blur_sigma=blur_sigma,
            denoise_strength=denoise_strength,
            rotate_deg=rotate_deg,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical,
            sharpen_amount=sharpen_amount,
            fade_in_sec=fade_in_sec,
            fade_out_sec=fade_out_sec,
            mute=mute,
            volume=volume,
            add_tiktok_chrome=tiktok_chrome_enabled,
            captioner_overlay_defaults=captioner_overlay_defaults,
            global_beat_bpm=global_beat_bpm,
            global_beat_amount=global_beat_amount,
            global_beat_phase_sec=global_beat_phase,
            global_beat_zoom_factor=global_beat_zoom_factor,
            global_beat_shake_factor=global_beat_shake_factor,
        )

        if not os.path.exists(output_path):
            raise RuntimeError("Editor timeline render finished without an output file")

        if os.path.getsize(output_path) <= 0:
            raise RuntimeError("Editor timeline render produced an empty output file")

        report(1.0, "Editor timeline render complete")
        return {
            "mode": mode,
            "timelineProvided": timeline_payload is not None,
            "timelineApplied": True,
            "timelineClipCount": timeline_result["clipCount"],
            "textOverlayCount": timeline_result.get("textOverlayCount", 0),
            "tiktokAdsChrome": timeline_result.get("tiktokAdsChrome", False),
            "audioBeatProfile": detected_beat_profile,
            "audioBeatFallbackUsed": beat_fallback_used,
            "audioBeatMotion": {
                "zoomFactor": global_beat_zoom_factor,
                "shakeFactor": global_beat_shake_factor,
            },
            "width": width,
            "height": height,
            "fps": fps,
            "durationSec": timeline_result["durationSec"],
            "playbackRate": playback_rate,
            "fit": fit,
            "mute": mute,
            "appliedOptions": {
                "brightness": brightness,
                "contrast": contrast,
                "saturation": saturation,
                "gamma": gamma,
                "hueDeg": hue_deg,
                "grayscale": grayscale,
                "blurSigma": blur_sigma,
                "denoiseStrength": denoise_strength,
                "rotateDeg": rotate_deg,
                "flipHorizontal": flip_horizontal,
                "flipVertical": flip_vertical,
                "sharpenAmount": sharpen_amount,
                "fadeInSec": fade_in_sec,
                "fadeOutSec": fade_out_sec,
            },
        }

    report(0.2, "Running FFmpeg render...")

    command: List[str]
    if _is_image(input_path):
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-i",
            input_path,
            "-t",
            f"{clip_duration:.3f}",
            "-vf",
            vf,
            "-r",
            str(fps),
            "-c:v",
            video_codec,
            "-preset",
            preset,
            "-crf",
            str(quality_crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            output_path,
        ]
    elif _is_audio(input_path):
        audio_filters = _build_audio_filters()

        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={ffmpeg_bg_color}:s={width}x{height}:r={fps}",
            "-i",
            input_path,
            "-shortest",
            "-t",
            f"{clip_duration:.3f}",
            "-c:v",
            video_codec,
            "-preset",
            preset,
            "-crf",
            str(quality_crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
        ]
        if mute:
            command.extend(["-an"])
        elif audio_filters:
            command.extend(["-af", ",".join(audio_filters)])
        command.append(output_path)
    else:
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
        ]
        if trim_start_sec > 0:
            command.extend(["-ss", f"{trim_start_sec:.3f}"])

        command.extend(["-i", input_path])
        command.extend(["-t", f"{clip_duration:.3f}"])
        command.extend(
            [
                "-vf",
                vf,
                "-c:v",
                video_codec,
                "-preset",
                preset,
                "-crf",
                str(quality_crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ]
        )

        if mute:
            command.extend(["-an"])
        else:
            has_audio_stream = _has_audio_stream(input_path)
            audio_filters = _build_audio_filters() if has_audio_stream else []
            if audio_filters and has_audio_stream:
                command.extend(["-af", ",".join(audio_filters)])
            if has_audio_stream:
                command.extend(["-c:a", "aac", "-b:a", "192k"])

        command.append(output_path)

    _run_ffmpeg(command)

    if not os.path.exists(output_path):
        raise RuntimeError("Editor render finished without an output file")

    if os.path.getsize(output_path) <= 0:
        raise RuntimeError("Editor render produced an empty output file")

    report(1.0, "Editor render complete")

    return {
        "mode": mode,
        "timelineProvided": timeline_payload is not None,
        "timelineApplied": False,
        "width": width,
        "height": height,
        "fps": fps,
        "durationSec": clip_duration,
        "playbackRate": playback_rate,
        "fit": fit,
        "appliedOptions": {
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "gamma": gamma,
            "hueDeg": hue_deg,
            "grayscale": grayscale,
            "blurSigma": blur_sigma,
            "denoiseStrength": denoise_strength,
            "rotateDeg": rotate_deg,
            "flipHorizontal": flip_horizontal,
            "flipVertical": flip_vertical,
            "sharpenAmount": sharpen_amount,
            "fadeInSec": fade_in_sec,
            "fadeOutSec": fade_out_sec,
            "audioNormalize": audio_normalize,
            "audioFadeInSec": audio_fade_in_sec,
            "audioFadeOutSec": audio_fade_out_sec,
            "audioHighpassHz": audio_highpass_hz,
            "audioLowpassHz": audio_lowpass_hz,
            "mute": mute,
            "volume": volume,
        },
    }
