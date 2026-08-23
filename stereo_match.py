import cv2
import numpy as np

# =========================================================================
# GLOBAL CONSTANTS & CONFIGURATION
# =========================================================================
BLOCK_SIZE = 13          # slightly larger ZNCC window -> more robust to noise
RADIUS = BLOCK_SIZE // 2
SEARCH_RANGE_X = 300
VERTICAL_TOLERANCE = 6   # +/- px around the epiline to absorb small F estimation error

MAX_DISPLAY_LINES = 40
MIN_LINE_SPACING = 25

PALETTE = [
    (255, 80, 80), (80, 220, 255), (80, 255, 120), (255, 200, 60),
    (255, 90, 220), (140, 200, 255), (0, 165, 255), (200, 255, 0),
    (180, 105, 255), (60, 255, 200),
]

# Shared state canvas elements
left_frame, right_frame = None, None
right_interactive_canvas = None
f_matrix = None

def compute_zncc_metric(patch_1, patch_2):
    """
    Computes Zero-Mean Normalized Cross-Correlation to robustly match
    local window illumination variations across the stereo frame pair.
    """
    if patch_1.shape != patch_2.shape:
        return -1.0

    mean_removed_1 = patch_1 - np.mean(patch_1)
    mean_removed_2 = patch_2 - np.mean(patch_2)

    numerator = np.sum(mean_removed_1 * mean_removed_2)
    denominator = np.sqrt(np.sum(mean_removed_1**2) * np.sum(mean_removed_2**2))

    if denominator == 0:
        return -1.0
    return numerator / denominator

def map_epipolar_match(click_x, click_y):
    """
    Traces the geometric epipolar line across the right viewport, calculates a localized
    ZNCC sliding block correlation, and highlights the corresponding point along with its score.
    """
    global left_frame, right_frame, right_interactive_canvas, f_matrix

    right_interactive_canvas = right_frame.copy()
    img_w = right_frame.shape[1]

    pt_homogeneous = np.array([[[click_x, click_y]]], dtype=np.float32)
    epiline_coords = cv2.computeCorrespondEpilines(pt_homogeneous, 1, f_matrix)
    a_coeff, b_coeff, c_coeff = epiline_coords[0][0]

    y_at_x0 = int(-c_coeff / b_coeff)
    y_at_xmax = int((-c_coeff - a_coeff * img_w) / b_coeff)

    cv2.line(right_interactive_canvas, (0, y_at_x0), (img_w, y_at_xmax), (0, 0, 0), 3, lineType=cv2.LINE_AA)
    cv2.line(right_interactive_canvas, (0, y_at_x0), (img_w, y_at_xmax), (0, 0, 255), 1, lineType=cv2.LINE_AA)

    gray_l = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
    h_max, w_max = gray_l.shape

    if click_y - RADIUS < 0 or click_y + RADIUS >= h_max or click_x - RADIUS < 0 or click_x + RADIUS >= w_max:
        print("[Boundary Warning] Click point patch overflows matrix dimensions.")
        return

    target_patch = gray_l[click_y - RADIUS : click_y + RADIUS + 1, click_x - RADIUS : click_x + RADIUS + 1]

    highest_correlation = -1.0
    best_matching_x, best_matching_y = None, None

    scan_start = max(RADIUS, click_x - SEARCH_RANGE_X)
    scan_end = min(gray_r.shape[1] - RADIUS, click_x + SEARCH_RANGE_X)

    for test_x in range(scan_start, scan_end):
        calculated_y = int(round((-a_coeff * test_x - c_coeff) / b_coeff))

        for vertical_offset in range(-VERTICAL_TOLERANCE, VERTICAL_TOLERANCE + 1):
            test_y = calculated_y + vertical_offset

            if test_y - RADIUS < 0 or test_y + RADIUS >= gray_r.shape[0]:
                continue

            sample_patch = gray_r[test_y - RADIUS : test_y + RADIUS + 1, test_x - RADIUS : test_x + RADIUS + 1]
            current_zncc = compute_zncc_metric(target_patch, sample_patch)

            if current_zncc > highest_correlation:
                highest_correlation = current_zncc
                best_matching_x, best_matching_y = test_x, test_y

    if best_matching_x is not None and highest_correlation > 0.25:
        arm_length = 14

        cv2.line(right_interactive_canvas, (best_matching_x - arm_length, best_matching_y), (best_matching_x + arm_length, best_matching_y), (0, 0, 0), 4)
        cv2.line(right_interactive_canvas, (best_matching_x, best_matching_y - arm_length), (best_matching_x, best_matching_y + arm_length), (0, 0, 0), 4)
        cv2.line(right_interactive_canvas, (best_matching_x - arm_length, best_matching_y), (best_matching_x + arm_length, best_matching_y), (0, 255, 0), 2)
        cv2.line(right_interactive_canvas, (best_matching_x, best_matching_y - arm_length), (best_matching_x, best_matching_y + arm_length), (0, 255, 0), 2)

        print(f"[Core Engine] Resolved Correspondence Point p': ({best_matching_x}, {best_matching_y}) | ZNCC: {highest_correlation:.4f}")
    else:
        print("[Match Notification] Surface texture profile too uniform. Click an edge or distinct feature.")

    cv2.imshow("Right Viewport (Stereo Match)", right_interactive_canvas)

def register_mouse_clicks(action, mouse_x, mouse_y, flags, parameters):
    if action == cv2.EVENT_LBUTTONDOWN:
        print(f"\n[Input Capture] Point p selected on Left Frame: ({mouse_x}, {mouse_y})")
        map_epipolar_match(mouse_x, mouse_y)

def main():
    global left_frame, right_frame, right_interactive_canvas, f_matrix

    print("[1/4] Loading photographic source assets...")
    img_1 = cv2.imread('photo1.jpeg')
    img_2 = cv2.imread('photo2.jpeg')

    if img_1 is None or img_2 is None:
        print("\n[IO Error] Execution halted. Verify photo1.jpeg and photo2.jpeg exist in the directory.")
        return

    src_h, src_w = img_1.shape[:2]
    left_frame = img_1.copy()
    right_frame = cv2.resize(img_2, (src_w, src_h), interpolation=cv2.INTER_AREA)

    monitor_max_h = 800
    if left_frame.shape[0] > monitor_max_h:
        resize_factor = monitor_max_h / left_frame.shape[0]
        left_frame = cv2.resize(left_frame, (0, 0), fx=resize_factor, fy=resize_factor)
        right_frame = cv2.resize(right_frame, (0, 0), fx=resize_factor, fy=resize_factor)

    right_interactive_canvas = right_frame.copy()

    mono_l = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    mono_r = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    print("[2/4] Initializing SIFT (Scale-Invariant Feature Transform)...")
    # More features, and slightly stricter contrast/edge thresholds -> higher quality keypoints
    sift_processor = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.03, edgeThreshold=12)
    keypoints_l, descriptors_l = sift_processor.detectAndCompute(mono_l, None)
    keypoints_r, descriptors_r = sift_processor.detectAndCompute(mono_r, None)

    brute_force_matcher = cv2.BFMatcher()
    raw_knn_matches = brute_force_matcher.knnMatch(descriptors_l, descriptors_r, k=2)

    # Stricter Lowe ratio test (0.75). No rigid vertical-band pre-filter here --
    # that heuristic assumes the two shots are near-parallel/rectified and silently
    # discards valid matches on more general camera motion. RANSAC/MAGSAC below
    # does proper geometric outlier rejection instead, which is both more accurate
    # and works for any two-view geometry, not just near-horizontal baselines.
    filtered_matches = []
    points_left_space = []
    points_right_space = []

    for best_match, second_best in raw_knn_matches:
        if best_match.distance < 0.75 * second_best.distance:
            filtered_matches.append(best_match)
            points_left_space.append(keypoints_l[best_match.queryIdx].pt)
            points_right_space.append(keypoints_r[best_match.trainIdx].pt)

    points_left_space = np.array(points_left_space, dtype=np.float32).reshape(-1, 2)
    points_right_space = np.array(points_right_space, dtype=np.float32).reshape(-1, 2)

    print("[3/4] Modeling Fundamental Matrix architecture via MAGSAC++ filtering...")
    # MAGSAC++ gives a more accurate, threshold-insensitive fundamental matrix
    # compared to plain RANSAC; falls back gracefully if unavailable.
    f_matrix, consensus_mask = cv2.findFundamentalMat(points_left_space, points_right_space, cv2.USAC_MAGSAC, 1.0, 0.999)

    if f_matrix is None:
        f_matrix, consensus_mask = cv2.findFundamentalMat(points_left_space, points_right_space, cv2.FM_RANSAC, 2.5, 0.99)

    if f_matrix is None:
        f_matrix, consensus_mask = cv2.findFundamentalMat(points_left_space, points_right_space, cv2.FM_8POINT)

    if f_matrix is None:
        print("[Fatal Mathematical Error] Epipolar framework geometry collapsed.")
        return

    geometric_inliers = [filtered_matches[i] for i in range(len(filtered_matches)) if consensus_mask[i][0] == 1]
    print(f"[Diagnostics] Inliers: {len(geometric_inliers)} / {len(filtered_matches)}")

    # Rank by descriptor match quality (lowest distance = most reliable) so the
    # cleanest matches get priority when we cap how many lines to draw
    geometric_inliers.sort(key=lambda match_obj: match_obj.distance)

    h_l, w_l = left_frame.shape[:2]
    h_r, w_r = right_frame.shape[:2]

    panoramic_verification_canvas = np.zeros((max(h_l, h_r), w_l + w_r, 3), dtype=np.uint8)
    panoramic_verification_canvas[0:h_l, 0:w_l] = left_frame
    panoramic_verification_canvas[0:h_r, w_l:w_l+w_r] = right_frame

    # Cap total lines and enforce spacing so they stay readable instead of overlapping
    chosen_matches = []
    chosen_points = []
    for candidate_match in geometric_inliers:
        p = keypoints_l[candidate_match.queryIdx].pt
        if all(np.hypot(p[0] - cp[0], p[1] - cp[1]) > MIN_LINE_SPACING for cp in chosen_points):
            chosen_matches.append(candidate_match)
            chosen_points.append(p)
        if len(chosen_matches) >= MAX_DISPLAY_LINES:
            break

    print(f"[Visual Generator] Rendering {len(chosen_matches)} clean, well-spaced match lines...")

    for i, inline_match in enumerate(chosen_matches):
        raw_left_xy = tuple(map(int, keypoints_l[inline_match.queryIdx].pt))
        raw_right_xy = tuple(map(int, keypoints_r[inline_match.trainIdx].pt))
        shifted_right_xy = (raw_right_xy[0] + w_l, raw_right_xy[1])
        color = PALETTE[i % len(PALETTE)]

        cv2.line(panoramic_verification_canvas, raw_left_xy, shifted_right_xy, (0, 0, 0), 3, lineType=cv2.LINE_AA)
        cv2.line(panoramic_verification_canvas, raw_left_xy, shifted_right_xy, color, 1, lineType=cv2.LINE_AA)
        cv2.circle(panoramic_verification_canvas, raw_left_xy, 6, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(panoramic_verification_canvas, raw_left_xy, 4, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(panoramic_verification_canvas, shifted_right_xy, 6, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(panoramic_verification_canvas, shifted_right_xy, 4, color, -1, lineType=cv2.LINE_AA)

    print("\n[Verification Open] Displaying clean SIFT feature matching vectors.")
    print("--> Press ANY key while focusing the window to move into the interactive environment.")
    cv2.imshow("Custom Multi-Colored SIFT Feature Match Verification", panoramic_verification_canvas)
    cv2.waitKey(0)
    cv2.destroyWindow("Custom Multi-Colored SIFT Feature Match Verification")

    # --- [4/4] LAUNCH ACTIVE INTERACTIVE MATRIX WORKSPACE ---
    cv2.imshow("Left Viewport (Click Coordinates Here)", left_frame)
    cv2.imshow("Right Viewport (Stereo Match)", right_interactive_canvas)

    # Pick the starting point from the best-quality SIFT inlier itself (already
    # sorted by ascending descriptor distance = best match first) instead of a
    # fixed guessed screen position. A hardcoded (33%, 40%) point has no
    # guarantee of landing on a distinctive feature -- it could just as easily
    # fall on flat sky or a blank wall, giving a weak/inaccurate first result.
    # Starting from a keypoint SIFT + MAGSAC already agreed on guarantees the
    # very first ZNCC search has strong texture to lock onto.
    initial_x, initial_y = None, None
    for candidate_match in geometric_inliers:
        px, py = keypoints_l[candidate_match.queryIdx].pt
        px, py = int(round(px)), int(round(py))
        margin = RADIUS + VERTICAL_TOLERANCE + 2
        if margin <= px < w_l - margin and margin <= py < h_l - margin:
            initial_x, initial_y = px, py
            break

    # Fallback in the unlikely case no inlier has enough margin from the border
    if initial_x is None:
        initial_x = int(left_frame.shape[1] * 0.33)
        initial_y = int(left_frame.shape[0] * 0.40)

    map_epipolar_match(initial_x, initial_y)

    cv2.setMouseCallback("Left Viewport (Click Coordinates Here)", register_mouse_clicks)

    print("\n[Workspace Active] Click anywhere on the left viewport to update matching results.")
    print("--> Press 'ESC' smoothly inside the display window frames to terminate gracefully.")

    while True:
        runtime_key = cv2.waitKey(1) & 0xFF
        if runtime_key == 27:
            print("[System Status] Workspace closed safely.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()