import numpy as np
import polars as pl
from PIL import Image

from caloreye.data.triage_probe import (
    hard_reasons,
    mark_exact_duplicates,
    probe_candidate,
    score_pixels
)

def test_score_pixels_blank_image_has_no_sharpness():
    image = np.full((32, 32, 3), 120, dtype=np.uint8)

    facts = score_pixels(image)

    assert facts["luma_std"] == 0
    assert facts["sharpness"] == 0
    assert facts["text_density"] == 0

def test_hard_reasons_checks_geometry_and_blankness():
    facts = {"decodes": True, "short_side": 100, "aspect": 5.0, "luma_std": 0.0}

    reasons = hard_reasons(facts)

    assert reasons == ["tiny", "extreme_aspect", "blank"]

def test_probe_candidate_transposes_exif_and_decodes(tmp_path):
    image_path = tmp_path / "food.jpg"
    image = Image.new("RGB", (200, 300), (120, 80, 50))
    exif = Image.Exif()
    exif[274] = 6
    image.save(image_path, exif=exif)
    candidate = {
        "candidate_id": "image1",
        "source_id": "6",
        "local_path": str(image_path)
    }

    facts = probe_candidate(candidate)

    assert facts["decodes"]
    assert (facts["width"], facts["height"]) == (300, 200)
    assert facts["short_side"] == 200

def test_mark_exact_duplicates_keeps_largest_resolution():
    frame = pl.DataFrame({
        "candidate_id": ["small", "large"],
        "original_sha256": ["sha", "sha"],
        "width": [100, 200],
        "height": [100, 200],
        "hard_reasons": [[], []]
    }).lazy()

    rows = mark_exact_duplicates(frame).collect().sort("candidate_id")

    assert rows["hard_reasons"].to_list() == [[], ["exact_duplicate"]]
