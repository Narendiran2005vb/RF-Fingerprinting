"""Fast source extraction from a 2-D CFAR detection mask."""

import numpy as np
from scipy import ndimage


def extract_sources(
    detection,
    power,
    freqs_hz,
    times_s,
    min_pixels=10,
    closing_size=None,
):
    """
    Fast connected-component extraction.

    Important:
    This version avoids np.where(labels == old_id) for every component.
    That operation can become extremely slow when CFAR produces many
    small detections.

    Returns:
        sources      : list of dictionaries
        source_labels: compact 2-D label image
        source_ids   : [1, 2, ..., N]
        num_sources  : N
    """
    detection = np.asarray(detection, dtype=bool)
    power = np.asarray(power)

    if detection.ndim != 2 or power.shape != detection.shape:
        raise ValueError("detection and power must be matching 2-D arrays.")

    # Do not perform morphological closing by default.
    # On a large spectrogram it can be expensive and is not required
    # for the basic CFAR validation.
    if closing_size is not None:
        structure = np.ones(closing_size, dtype=bool)
        mask = ndimage.binary_closing(detection, structure=structure)
    else:
        mask = detection

    # Connected-component labeling.
    labels, n = ndimage.label(
        mask,
        structure=np.ones((3, 3), dtype=np.uint8)
    )

    if n == 0:
        return [], np.zeros_like(labels, dtype=np.int32), \
               np.empty(0, dtype=np.int32), 0

    # Count pixels in every component ONCE.
    counts = np.bincount(labels.ravel())

    # Keep only components meeting the minimum area.
    keep = np.flatnonzero(counts >= min_pixels)
    keep = keep[keep != 0]  # label 0 = background

    if keep.size == 0:
        return [], np.zeros_like(labels, dtype=np.int32), \
               np.empty(0, dtype=np.int32), 0

    # Compact labels:
    # old label -> new source ID
    lut = np.zeros(n + 1, dtype=np.int32)
    lut[keep] = np.arange(1, keep.size + 1, dtype=np.int32)
    source_labels = lut[labels]

    sources = []

    # find_objects is much faster than scanning the whole image for
    # every component. It returns bounding boxes.
    objects = ndimage.find_objects(labels)

    for source_id, old_id in enumerate(keep, start=1):
        sl = objects[old_id - 1]
        if sl is None:
            continue

        fslice, tslice = sl

        # Restrict work to this component's bounding box.
        local_labels = labels[fslice, tslice]
        local_power = power[fslice, tslice]

        component_mask = local_labels == old_id
        fi_local, ti_local = np.nonzero(component_mask)

        if fi_local.size == 0:
            continue

        fi = fi_local + fslice.start
        ti = ti_local + tslice.start

        component_power = local_power[component_mask]
        peak_index = int(np.argmax(component_power))

        peak_f = float(freqs_hz[fi[peak_index]])
        peak_t = float(times_s[ti[peak_index]])
        peak_power = float(component_power[peak_index])

        f_low = float(freqs_hz[fi.min()])
        f_high = float(freqs_hz[fi.max()])
        t_start = float(times_s[ti.min()])
        t_end = float(times_s[ti.max()])

        sources.append({
            "source_id": source_id,
            "peak_frequency_hz": peak_f,
            "peak_time_s": peak_t,
            "peak_power_linear": peak_power,
            "frequency_low_hz": f_low,
            "frequency_high_hz": f_high,
            "bandwidth_hz": f_high - f_low,
            "time_start_s": t_start,
            "time_end_s": t_end,
            "duration_s": t_end - t_start,
            "detection_pixels": int(counts[old_id]),
        })

    source_ids = np.arange(
        1, len(sources) + 1, dtype=np.int32
    )

    return sources, source_labels, source_ids, len(sources)
