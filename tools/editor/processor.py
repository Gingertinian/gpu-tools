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
from typing import Any, Callable, Dict, List, Optional


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
    result = subprocess.run(command, capture_output=True, text=True)
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


def _looks_like_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.startswith("http://") or value.startswith("https://")


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


def _parse_transition_fade_sec(raw_transition: Any) -> float:
    if not isinstance(raw_transition, dict):
        return 0.0

    transition_type = _safe_effect_type(
        raw_transition.get("type")
        or raw_transition.get("transitionType")
        or raw_transition.get("transition_type")
        or raw_transition.get("name")
    )
    if transition_type not in {
        "fade",
        "dissolve",
        "crossfade",
        "dip",
        "diptocolor",
        "fadeblack",
        "fadewhite",
    }:
        return 0.0

    duration_raw = (
        raw_transition.get("durationSec")
        or raw_transition.get("duration_sec")
        or raw_transition.get("duration")
        or _extract_effect_value(raw_transition, "durationSec")
    )
    return _clamp_number(duration_raw, 0.0, 5.0, 0.0)


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

    return resolved


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

    tracks = timeline_payload.get("tracks")
    if isinstance(tracks, dict):
        video_clips.extend(_as_dict_list(tracks.get("video")))
        video_clips.extend(_as_dict_list(tracks.get("videos")))
        effect_entries.extend(_as_dict_list(tracks.get("effects")))
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

            if kind in {"video", "videos", "media", "main"}:
                video_clips.extend(track_video_clips)
            elif kind in {"effect", "effects", "fx", "filters"}:
                effect_entries.extend(track_effect_entries)

    if not video_clips:
        video_clips.extend(
            _extract_track_payload(
                timeline_payload,
                ["video", "videos", "clips", "items", "segments"],
            )
        )

    if not effect_entries:
        effect_entries.extend(_as_dict_list(timeline_payload.get("effects")))

    return {
        "video": video_clips,
        "effects": effect_entries,
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


def _render_timeline_video(
    output_path: str,
    clips: List[Dict[str, Any]],
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
) -> Dict[str, Any]:
    if not clips:
        raise ValueError("Cannot render timeline without clips")

    command: List[str] = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    filter_parts: List[str] = []
    concat_labels: List[str] = []
    audio_labels: List[str] = []

    timeline_duration = 0.0
    fallback_cursor = 0.0

    for index, clip in enumerate(clips):
        clip_duration = max(0.1, float(clip["durationSec"]))
        clip_effects = _resolve_clip_effects(clip.get("effectStack"), clip_duration)
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
            clip_filter_parts.extend(clip_video_filters)
            clip_filter_parts.extend(
                [
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
            clip_filter_parts.extend(clip_video_filters)
            clip_filter_parts.extend(
                [
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

        clip_start = max(fallback_cursor, float(clip.get("startSec", fallback_cursor)))
        timeline_duration = max(timeline_duration, clip_start + clip_duration)
        fallback_cursor = clip_start + clip_duration

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

    post_filters.append(f"fps={fps}")
    filter_parts.append(f"[vcat]{','.join(post_filters)}[vout]")

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

    _run_ffmpeg(command)

    return {
        "clipCount": len(clips),
        "durationSec": max(0.1, timeline_duration),
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

    if mode == "timeline" and timeline_clips:
        report(0.2, "Running timeline render...")
        timeline_result = _render_timeline_video(
            output_path=output_path,
            clips=timeline_clips,
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
