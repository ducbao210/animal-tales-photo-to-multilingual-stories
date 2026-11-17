"""Optional reference-guided character frame generation and sprite assembly."""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from src.contracts.schema import StageError, ensure_under
from src.utils.files import atomic_write_bytes


DEFAULT_POSES = (
    ("happy", "standing with a bright smile"),
    ("surprised", "ears raised and eyes wide in surprise"),
    ("running", "running forward with a joyful expression"),
    ("climbing", "carefully climbing with a focused expression"),
)


def generate_animation_frames(
    reference_image: Path, prompt_template_path: Path, output_dir: Path,
    client: Any, settings: dict[str, Any],
    poses: tuple[tuple[str, str], ...] = DEFAULT_POSES,
) -> list[Path]:
    """Generate four reference-guided frames; provider identity still needs human review."""
    if len(poses) < 4:
        raise ValueError("At least four poses are required")
    template = prompt_template_path.read_text(encoding="utf-8")
    description = settings.get("reference_description", "the same animal character as the reference image")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    for index, (emotion, action) in enumerate(poses, start=1):
        prompt = re.sub(r"\{frame_number\}", str(index), template)
        prompt = re.sub(r"\{reference_description\}", description, prompt)
        prompt = re.sub(r"\{action\}", action, prompt)
        prompt = re.sub(r"\{emotion\}", emotion, prompt)
        if re.search(r"\{[^{}]+\}", prompt):
            raise ValueError("Pose prompt contains an unresolved variable")
        response = client.generate(prompt, output_dir=output_dir, settings={**settings, "reference_image_path": str(reference_image)})
        path = ensure_under(output_dir / f"frame_{index:02d}.png", output_dir)
        try:
            with Image.open(io.BytesIO(response.data)) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as error:
            raise StageError("B", "image_decode", False, f"Pose frame {index} is not a valid image") from error
        atomic_write_bytes(path, response.data)
        frames.append(path)
    return frames


def assemble_animation(frames: list[Path], output_dir: Path, *, duration_ms: int = 350) -> tuple[Path, Path]:
    if len(frames) < 4:
        raise ValueError("At least four frames are required for the animation bonus")
    output_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for path in frames:
        with Image.open(path) as image:
            images.append(image.convert("RGBA"))
    size = (max(image.width for image in images), max(image.height for image in images))
    normalized = []
    for image in images:
        canvas = Image.new("RGBA", size, (255, 255, 255, 0))
        canvas.alpha_composite(image.resize(size))
        normalized.append(canvas)
    gif_buffer = io.BytesIO()
    normalized[0].save(gif_buffer, format="GIF", save_all=True, append_images=normalized[1:], duration=duration_ms, loop=0, disposal=2)
    gif_path = ensure_under(output_dir / "frames.gif", output_dir)
    atomic_write_bytes(gif_path, gif_buffer.getvalue())
    sheet = Image.new("RGBA", (size[0] * len(normalized), size[1]), (255, 255, 255, 0))
    for idx, image in enumerate(normalized):
        sheet.alpha_composite(image, (idx * size[0], 0))
    sheet_buffer = io.BytesIO()
    sheet.save(sheet_buffer, format="PNG")
    sheet_path = ensure_under(output_dir / "sprite_sheet.png", output_dir)
    atomic_write_bytes(sheet_path, sheet_buffer.getvalue())
    return gif_path, sheet_path
