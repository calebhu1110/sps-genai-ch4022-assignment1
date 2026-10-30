"""GAN digit-generation endpoint (Assignment 3).

Use as a router inside the existing app:

    from app.gan.api import router as gan_router
    app.include_router(gan_router)
"""
import io
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from PIL import Image

from .model import Z_DIM, Generator

GENERATOR_PATH = Path(__file__).parent / "generator.pth"

# Load the trained Generator once, when the app starts.
# eval() makes BatchNorm use its running statistics (required for inference).
_generator = Generator(Z_DIM)
_generator.load_state_dict(torch.load(GENERATOR_PATH, map_location="cpu"))
_generator.eval()

router = APIRouter(tags=["gan"])


@router.get("/generate_digit", response_class=Response,
            responses={200: {"content": {"image/png": {}}}})
def generate_digit(
    seed: int | None = Query(None, description="Optional random seed (same seed -> same digit)"),
    scale: int = Query(8, ge=1, le=32, description="Upscale factor for the 28x28 image"),
):
    """Generate one handwritten-digit image from random noise; returns a PNG."""
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    z = torch.randn(1, Z_DIM, generator=gen)  # (1, 100)

    with torch.no_grad():
        img = _generator(z)[0, 0]             # (28, 28), values in [-1, 1]

    img = ((img + 1) / 2).clamp(0, 1)          # [-1, 1] -> [0, 1]
    pixels = (img * 255).byte().numpy()        # -> 0..255
    image = Image.fromarray(pixels, mode="L")
    if scale > 1:
        image = image.resize((28 * scale, 28 * scale), Image.NEAREST)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
