import numpy as np
import cv2
from skimage import color, util, morphology, measure, exposure
from skimage.segmentation import watershed
from scipy import ndimage as ndi


def _remove_small_objects(binary, min_size):
    try:
        return morphology.remove_small_objects(binary, max_size=min_size)
    except TypeError:
        return morphology.remove_small_objects(binary, min_size=min_size)


def _remove_small_holes(binary, max_hole_size):
    try:
        return morphology.remove_small_holes(binary, max_size=max_hole_size)
    except TypeError:
        return morphology.remove_small_holes(binary, area_threshold=max_hole_size)


def _detect_footer_top(image):
    """Return the first row of a bottom microscope footer, or image height."""
    h, w = image.shape[:2]
    if h < 200 or w < 200:
        return h

    row_mean = image.mean(axis=1)
    row_std = image.std(axis=1)
    row_bright = (image > 190).mean(axis=1)

    body_end = max(1, int(h * 0.75))
    body_mean = float(np.median(row_mean[:body_end]))
    body_std = float(np.median(row_std[:body_end]))
    body_bright = float(np.median(row_bright[:body_end]))

    footer_like = (
        ((row_mean < body_mean - 10) & (row_std < body_std + 5))
        | (row_bright > body_bright + 0.12)
        | (row_std > body_std + 22)
    )

    window = int(np.clip(h // 180, 15, 45))
    smooth = np.convolve(footer_like.astype(float), np.ones(window) / window, mode="same")
    rows = np.arange(h)
    candidates = np.flatnonzero((rows >= int(h * 0.82)) & (smooth > 0.35))
    if candidates.size == 0:
        return h

    footer_top = int(candidates[0])
    footer_height = h - footer_top
    if footer_height < max(30, int(h * 0.01)) or footer_height > int(h * 0.18):
        return h
    return footer_top


def _border_ratio(binary, valid_height):
    h, w = binary.shape[:2]
    valid_height = int(np.clip(valid_height, 1, h))
    y_bottom = valid_height - 1
    if valid_height == 1:
        values = binary[0, :]
    else:
        values = np.concatenate([
            binary[0, :],
            binary[y_bottom, :],
            binary[1:y_bottom, 0],
            binary[1:y_bottom, w - 1],
        ])
    return float(values.mean()) if values.size else 0.0


def _postprocess_candidate(raw_binary, footer_top, closing_radius, opening_radius):
    binary = raw_binary.astype(bool, copy=True)
    if footer_top < binary.shape[0]:
        binary[footer_top:, :] = False

    binary = _remove_small_objects(binary, 500)

    if closing_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (closing_radius * 2 + 1, closing_radius * 2 + 1),
        )
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    if opening_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (opening_radius * 2 + 1, opening_radius * 2 + 1),
        )
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)

    if footer_top < binary.shape[0]:
        binary[footer_top:, :] = False

    max_hole_size = int(np.clip(binary.size // 2000, 256, 10000))
    binary = _remove_small_holes(binary, max_hole_size)

    if footer_top < binary.shape[0]:
        binary[footer_top:, :] = False

    return binary


def _score_candidate(binary, valid_height):
    h, w = binary.shape[:2]
    valid_height = int(np.clip(valid_height, 1, h))
    valid_binary = binary[:valid_height, :]
    valid_pixels = max(1, valid_binary.size)
    foreground_ratio = float(valid_binary.sum()) / valid_pixels

    if foreground_ratio <= 0:
        return float("inf")

    labeled, _ = ndi.label(valid_binary)
    sizes = np.bincount(labeled.ravel())
    component_sizes = sizes[1:]
    component_count = len(component_sizes)
    largest_ratio = (float(component_sizes.max()) / valid_pixels) if component_count else 0.0
    border = _border_ratio(binary, valid_height)

    score = abs(foreground_ratio - 0.16)
    score += largest_ratio * 2.5
    score += border * 0.75
    expected_component_limit = max(50.0, valid_pixels / 10000.0)
    if component_count > expected_component_limit:
        score += ((component_count - expected_component_limit) / expected_component_limit) * 0.75
    if foreground_ratio > 0.50:
        score += (foreground_ratio - 0.50) * 4.0
    if foreground_ratio < 0.002:
        score += (0.002 - foreground_ratio) * 20.0
    return score


def image_kmeans(image, k=2, separation_strength=0):
    """
    Segments the image using K-means clustering.
    Matches MATLAB 'imagekmeans.m' and 'loadEMimages.m' logic.
    """
    # 1. Preprocessing — Contrast Stretching
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    p1, p99 = np.percentile(image, (1, 99))
    if p99 <= p1:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    image_adj = exposure.rescale_intensity(image, in_range=(p1, p99), out_range=np.uint8)
    image_adj = cv2.medianBlur(image_adj, 3)

    # 2. K-means clustering (k=2: foreground vs background)
    # Use a deterministic sample for large camera JPGs, then apply centers to full image.
    k = max(2, int(k))
    max_kmeans_pixels = 750_000
    step = max(1, int(np.ceil(np.sqrt(image_adj.size / max_kmeans_pixels))))
    data = image_adj[::step, ::step].reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    cv2.setRNGSeed(0)
    _, _, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = centers.reshape(-1)

    if k == 2:
        threshold = float(np.mean(centers))
        if centers[0] <= centers[1]:
            segmented_image = np.where(image_adj <= threshold, 0, 1)
        else:
            segmented_image = np.where(image_adj <= threshold, 1, 0)
    else:
        distances = np.abs(image_adj.astype(np.float32)[..., None] - centers[None, None, :])
        segmented_image = np.argmin(distances, axis=2)

    # 3. Foreground selection. Border voting alone breaks on exported camera JPGs
    # with bottom metadata strips, so score each class after light morphology.
    h, w = image_adj.shape[:2]
    footer_top = _detect_footer_top(image_adj)

    separation_strength = int(np.clip(separation_strength, -6, 6))
    merge_closing_radius = [1, 3, 5, 7, 9, 11, 13]
    split_opening_radius = [3, 4, 5, 6, 7, 8, 9]
    if separation_strength < 0:
        closing_radius = merge_closing_radius[-separation_strength]
        opening_radius = 1
    else:
        closing_radius = 1
        opening_radius = split_opening_radius[separation_strength]

    best_binary = None
    best_score = float("inf")
    for label_idx in range(k):
        candidate = _postprocess_candidate(
            segmented_image == label_idx,
            footer_top,
            closing_radius,
            opening_radius,
        )
        score = _score_candidate(candidate, footer_top)
        if score < best_score:
            best_score = score
            best_binary = candidate

    binary = best_binary if best_binary is not None else np.zeros((h, w), dtype=bool)
    return binary.astype(np.uint8)

def compute_hu_moments(image):
    """
    Computes Hu moments for a binary or grayscale image.
    """
    moments = measure.moments_central(image)
    hu_moments = measure.moments_hu(moments)
    return hu_moments

def masking(img, region_coords):
    """
    Helper function to create a masked image from region coordinates.
    """
    mask = np.zeros(img.shape, dtype=bool)
    # region_coords is (N, 2) array of (row, col)
    mask[region_coords[:, 0], region_coords[:, 1]] = True
    return img & mask

def ruecs(img_input, area_threshold=25, cnt=0):
    """
    Recursive Ultimate Erosion of Convex Shapes (rUECS).
    
    Args:
        img_input: Binary image (numpy array) or a list of dictionaries representing particles.
        area_threshold: Minimum area to keep.
        cnt: Iteration counter (used in recursion).
    """
    
    # Initialize if input is an image
    if not isinstance(img_input, list):
        img_bool = img_input > 0
        area = np.sum(img_bool)
        
        particle = {
            'image': img_bool,
            'init_area': area,
            'area': area,
            'cnt': cnt,
            'isconvex': False,
            'keep': True
        }
        img_list = [particle]
    else:
        img_list = img_input

    # Structuring elements
    se1 = morphology.disk(1)
    se2 = np.ones((2, 2), dtype=np.uint8)
    
    queue = list(img_list)
    final_markers = []
    
    while queue:
        current_particle = queue.pop(0)
        
        if not current_particle['keep']:
            final_markers.append(current_particle)
            continue
            
        if current_particle['isconvex']:
            final_markers.append(current_particle)
            continue
            
        image = current_particle['image']
        
        # Check if empty
        if not np.any(image):
            current_particle['keep'] = False
            final_markers.append(current_particle)
            continue

        # Label to handle connected components
        label_img = measure.label(image)
        if np.max(label_img) == 0:
             current_particle['keep'] = False
             final_markers.append(current_particle)
             continue
             
        regions = measure.regionprops(label_img)
        # Assuming single object or taking the first major one
        s = regions[0] 
        
        # Convexity criteria
        # Solidity = Area / ConvexArea
        # Defect = 1 - Solidity
        # Original: (1 - Area/ConvexArea > 0.1) -> Solidity < 0.9
        
        is_convex = True
        if s.area_convex == 0:
            convexity_defect = 0
        else:
            convexity_defect = 1.0 - (s.area / s.area_convex)
        
        # Perimeter ratio check (approx)
        # Using convex_image perimeter
        ch_perimeter = 0
        if s.image_convex.ndim == 2:
             ch_perimeter = measure.perimeter(s.image_convex)
        
        perimeter_ratio = 0
        if s.perimeter > 0:
            perimeter_ratio = ch_perimeter / s.perimeter
        
        if (convexity_defect > 0.1) or (perimeter_ratio < 0.9):
            is_convex = False
            
        if not is_convex:
            # Erode
            if current_particle['cnt'] % 2 == 1:
                se = se1
            else:
                se = se2
                
            eroded = morphology.erosion(image, se)
            opened = morphology.opening(eroded, se1)
            
            if not np.any(opened):
                current_particle['keep'] = False
                final_markers.append(current_particle)
                continue
                
            label_eroded = measure.label(opened)
            regions_eroded = measure.regionprops(label_eroded)
            
            if not regions_eroded:
                current_particle['keep'] = False
                final_markers.append(current_particle)
                continue
                
            first_region = regions_eroded[0]
            
            # Update current particle
            current_particle['image'] = masking(opened, first_region.coords)
            current_particle['cnt'] += 1
            current_particle['area'] = first_region.area
            
            # Check area threshold
            if (current_particle['area'] < 0.1 * current_particle['init_area']) or (current_particle['area'] < area_threshold):
                current_particle['keep'] = False
            
            # Split case
            if len(regions_eroded) > 1:
                current_particle['init_area'] = first_region.area
                queue.insert(0, current_particle) 
                
                for i in range(1, len(regions_eroded)):
                    sub_region = regions_eroded[i]
                    sub_img = masking(opened, sub_region.coords)
                    
                    new_particle = {
                        'image': sub_img,
                        'init_area': sub_region.area,
                        'area': sub_region.area,
                        'cnt': current_particle['cnt'],
                        'isconvex': False,
                        'keep': True
                    }
                    
                    # Recurse
                    sub_markers = ruecs([new_particle], area_threshold, current_particle['cnt'])
                    queue.extend(sub_markers)

            else:
                queue.insert(0, current_particle)
                
        else:
            current_particle['isconvex'] = True
            final_markers.append(current_particle)

    # Final filtering
    filtered_markers = []
    for m in final_markers:
        if m['area'] < area_threshold:
            m['keep'] = False
        if m['keep']:
            filtered_markers.append(m)
            
    return filtered_markers

def dilmarkers(markers, original_shape):
    """
    Dilates markers back to their original size.
    Returns: dilated_markers (list), overlay (RGB)
    """
    if not markers:
        # handle case where original_shape might be image or tuple
        shape = original_shape if isinstance(original_shape, tuple) else original_shape.shape[:2]
        return [], np.zeros(shape + (3,), dtype=np.uint8)
        
    se1 = morphology.disk(1)
    se2 = np.ones((2, 2), dtype=np.uint8)
    
    dilated_markers = []
    
    # Determine shape
    if isinstance(original_shape, np.ndarray):
        bg_image = original_shape
        if bg_image.ndim == 2:
            bg_image = color.gray2rgb(bg_image)
        elif bg_image.shape[2] == 4:
            bg_image = color.rgba2rgb(bg_image)
        if bg_image.dtype != np.uint8:
            bg_image = util.img_as_ubyte(bg_image)
        image_shape = bg_image.shape[:2]
    else:
        image_shape = original_shape
        bg_image = np.zeros(image_shape + (3,), dtype=np.uint8)

    for marker in markers:
        m_img = marker['image']
        cnt = marker['cnt']
        
        # Dilate
        dilated = m_img.copy()
        for j in range(cnt, 0, -1):
            if j % 2 == 1:
                se = se1
            else:
                se = se2
                
            # Keep dilation constrained to the image size (it is same size masks)
            dilated = morphology.dilation(dilated, se)
            
        dilated_markers.append(dilated)
        
    # We mainly need the dilated markers list
    # Constructing overlay omitted for speed unless needed, but returning dummy if needed
    # The user code might check return tuple.

    return dilated_markers, bg_image # returning image as overlay placeholder


def split_clump(crop_bool, min_marker_area, separation_strength=0):
    """
    Split a fused / low-solidity binary region into individual convex particles
    using a marker-controlled watershed on the Euclidean distance transform.

    This is the primary clump separator. Compared with recursive erosion
    (``ruecs``) it is far faster (the heavy steps are C-implemented) and more
    robust on dense nanorod aggregates, because seeds are extended (h-) maxima of
    the distance map: the whole ridge of one elongated rod collapses to a single
    regional maximum, so a single rod is NOT cut into pieces, while two touching
    rods separated by a neck become two seeds.

    Args:
        crop_bool: 2-D boolean array — one connected component (the clump).
        min_marker_area: discard watershed basins smaller than this (px).
        separation_strength: -6..+6. Higher = split more aggressively
            (more seeds); lower = merge more. 0 is the balanced default.

    Returns:
        List of 2-D boolean masks (same shape as ``crop_bool``), one per
        particle. Returns ``[crop_bool]`` when the region should stay whole.
    """
    crop_bool = np.ascontiguousarray(crop_bool, dtype=bool)
    if not crop_bool.any():
        return []

    # Bound worst-case cost: a binarisation failure can flood into one giant
    # region, where the distance transform / morphological reconstruction /
    # watershed would otherwise take many seconds. For oversized regions we run
    # the split on a downscaled copy and map the labels back. Normal particles
    # and clumps are far below this budget, so their results are unchanged.
    max_work_px = 4_000_000
    scale = 1
    work = crop_bool
    if crop_bool.size > max_work_px:
        scale = int(np.ceil(np.sqrt(crop_bool.size / max_work_px)))
        downscaled = crop_bool[::scale, ::scale]
        if downscaled.any():
            work = np.ascontiguousarray(downscaled)
        else:
            scale = 1

    dist = ndi.distance_transform_edt(work)
    # Light smoothing suppresses spurious ridge bumps that would over-segment.
    dist = ndi.gaussian_filter(dist, 1.0)

    peak = float(dist.max())
    # Region too thin to contain more than one particle core — keep as-is.
    if peak < 1.5:
        return [crop_bool]

    # h controls how deep a valley between two cores must be before they are
    # treated as separate seeds. Tie it to the user's separation slider so the
    # same control that tunes binarisation also tunes clump splitting.
    separation_strength = int(np.clip(separation_strength, -6, 6))
    h_frac = float(np.clip(0.45 - separation_strength * 0.04, 0.18, 0.72))
    h = max(0.75, peak * h_frac)

    markers, n = ndi.label(morphology.h_maxima(dist, h))
    if n <= 1:
        return [crop_bool]

    ws = watershed(-dist, markers, mask=work)

    if scale > 1:
        # Upsample the label map back to full resolution and re-constrain it to
        # the exact original mask.
        ws = np.repeat(np.repeat(ws, scale, axis=0), scale, axis=1)
        ws = ws[:crop_bool.shape[0], :crop_bool.shape[1]]
        ws = ws * crop_bool

    masks = []
    for label_idx in range(1, n + 1):
        mask = ws == label_idx
        if int(mask.sum()) >= min_marker_area:
            masks.append(mask)

    return masks or [crop_bool]
