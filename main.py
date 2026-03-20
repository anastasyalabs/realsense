import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os
import time

# =========================
# CONFIG
# =========================
WIDTH = 640
HEIGHT = 480
FPS = 15

CALIBRATION_FRAMES = 20
CALIBRATION_JSON = "calibration.json"
REFERENCE_DEPTH_NPY = "reference_depth.npy"
OUTPUT_DIR = "output"

MIN_OBJECT_HEIGHT_MM = 5.0
MAX_VIS_HEIGHT_MM = 300.0
MIN_CONTOUR_AREA = 300

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# CALIBRATION SAVE / LOAD
# =========================
def save_calibration_to_json(a, b, c, rmse, valid_ratio, width, height, filename=CALIBRATION_JSON):
    calibration_data = {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "rmse_mm": float(rmse),
        "valid_ratio": float(valid_ratio),
        "width": int(width),
        "height": int(height),
        "timestamp": int(time.time()),
        "reference_depth_file": REFERENCE_DEPTH_NPY
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(calibration_data, f, indent=4, ensure_ascii=False)

    print(f"[INFO] Kalibrācija saglabāta failā: {os.path.abspath(filename)}")


def load_calibration_from_json(filename=CALIBRATION_JSON):
    if not os.path.exists(filename):
        print("[INFO] Kalibrācijas JSON fails nav atrasts.")
        return None

    with open(filename, "r", encoding="utf-8") as f:
        calibration_data = json.load(f)

    print(f"[INFO] Kalibrācija ielādēta no faila: {os.path.abspath(filename)}")
    return calibration_data


def save_reference_depth(reference_depth_mm, filename=REFERENCE_DEPTH_NPY):
    np.save(filename, reference_depth_mm.astype(np.float32))
    print(f"[INFO] References dziļuma karte saglabāta: {os.path.abspath(filename)}")


def load_reference_depth(filename=REFERENCE_DEPTH_NPY):
    if not os.path.exists(filename):
        print("[INFO] References dziļuma NPY fails nav atrasts.")
        return None

    reference_depth_mm = np.load(filename).astype(np.float32)
    print(f"[INFO] References dziļuma karte ielādēta: {os.path.abspath(filename)}")
    return reference_depth_mm


# =========================
# MATH / SURFACE MODEL
# =========================
def fit_plane(depth_mm):
    print("[PROCESS] Starting plane fitting...")

    h, w = depth_mm.shape
    ys, xs = np.mgrid[0:h, 0:w]

    x = xs.flatten().astype(np.float32)
    y = ys.flatten().astype(np.float32)
    z = depth_mm.flatten().astype(np.float32)

    valid = z > 0
    x = x[valid]
    y = y[valid]
    z = z[valid]

    if len(z) < 1000:
        raise RuntimeError("Nepietiek derīgu dziļuma punktu plaknes pielāgošanai.")

    A = np.column_stack((x, y, np.ones_like(x)))
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

    a, b, c = coeffs
    z_pred = a * x + b * y + c

    rmse = np.sqrt(np.mean((z - z_pred) ** 2))
    valid_ratio = len(z) / (h * w)

    print("[PROCESS] Plane fitting finished")
    return float(a), float(b), float(c), float(rmse), float(valid_ratio)


def build_reference_plane(width, height, a, b, c):
    ys, xs = np.mgrid[0:height, 0:width]
    reference_plane = a * xs + b * ys + c
    return reference_plane.astype(np.float32)


def compute_height_map(depth_mm, reference_surface_mm):
    height_map = reference_surface_mm - depth_mm
    height_map[depth_mm <= 0] = 0
    height_map[reference_surface_mm <= 0] = 0
    height_map[height_map < 0] = 0
    return height_map.astype(np.float32)


def visualize_height_map(height_map, max_height_mm=MAX_VIS_HEIGHT_MM):
    vis = np.clip(height_map, 0, max_height_mm)
    vis = (vis / max_height_mm * 255.0).astype(np.uint8)
    return vis


# =========================
# SEGMENTATION / CONTOURS
# =========================
def segment_objects(height_map, min_height_mm=MIN_OBJECT_HEIGHT_MM):
    mask = np.zeros_like(height_map, dtype=np.uint8)
    mask[height_map > min_height_mm] = 255
    return mask


def clean_mask(mask):
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
    return cleaned


def refine_mask_with_rgb(mask, color_image):
    """
    RGB is secondary data.
    This keeps the height-map mask as primary, then sharpens edges with RGB gradients.
    """
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Dilate edges slightly so they can help define borders
    edge_kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, edge_kernel, iterations=1)

    # Keep original object regions, but strengthen visible boundaries
    refined = mask.copy()
    refined = cv2.bitwise_or(refined, cv2.bitwise_and(edges_dilated, mask))
    refined = clean_mask(refined)
    return refined


def find_object_contours(mask, min_area=MIN_CONTOUR_AREA):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            filtered.append(cnt)

    return filtered


def draw_contours_with_info(image, contours, height_map=None):
    result = image.copy()

    for idx, cnt in enumerate(contours, start=1):
        cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)

        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)

        label = f"Obj {idx}"

        if height_map is not None:
            object_mask = np.zeros(height_map.shape, dtype=np.uint8)
            cv2.drawContours(object_mask, [cnt], -1, 255, thickness=-1)
            object_heights = height_map[object_mask == 255]

            if object_heights.size > 0:
                max_h = float(np.max(object_heights))
                mean_h = float(np.mean(object_heights))
                label = f"Obj {idx} | max={max_h:.1f}mm avg={mean_h:.1f}mm"

        cv2.putText(
            result,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

    return result


# =========================
# SAVE OUTPUTS
# =========================
def save_outputs(base_name, color_image, depth_mm, height_map, mask, contour_image):
    color_path = os.path.join(OUTPUT_DIR, f"{base_name}_color.png")
    depth_path = os.path.join(OUTPUT_DIR, f"{base_name}_depth.npy")
    height_path = os.path.join(OUTPUT_DIR, f"{base_name}_height.npy")
    height_vis_path = os.path.join(OUTPUT_DIR, f"{base_name}_height_vis.png")
    mask_path = os.path.join(OUTPUT_DIR, f"{base_name}_mask.png")
    contours_path = os.path.join(OUTPUT_DIR, f"{base_name}_contours.png")

    cv2.imwrite(color_path, color_image)
    np.save(depth_path, depth_mm.astype(np.float32))
    np.save(height_path, height_map.astype(np.float32))
    cv2.imwrite(height_vis_path, visualize_height_map(height_map))
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(contours_path, contour_image)

    print(f"[INFO] Saglabāts: {color_path}")
    print(f"[INFO] Saglabāts: {depth_path}")
    print(f"[INFO] Saglabāts: {height_path}")
    print(f"[INFO] Saglabāts: {height_vis_path}")
    print(f"[INFO] Saglabāts: {mask_path}")
    print(f"[INFO] Saglabāts: {contours_path}")


# =========================
# REALSENSE HELPERS
# =========================
def get_aligned_frames(pipeline, align, depth_scale):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        return None, None, None

    depth_raw = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_mm = depth_raw.astype(np.float32) * depth_scale * 1000.0

    return depth_frame, color_frame, depth_mm, color_image


def warmup_camera(pipeline, num_frames=30):
    print(f"[PROCESS] Warming up camera ({num_frames} frames)")
    for _ in range(num_frames):
        pipeline.wait_for_frames()
    print("[PROCESS] Camera warm-up finished")


# =========================
# CALIBRATION ROUTINE
# =========================
def run_calibration(pipeline, align, depth_scale):
    print("\n[PROCESS] Calibration started")
    print("[INFO] Nodrošini, ka kamera redz TUKŠU VIRSMU bez objektiem")

    frames_list = []

    for i in range(CALIBRATION_FRAMES):
        _, _, depth_mm, _ = get_aligned_frames(pipeline, align, depth_scale)
        if depth_mm is None:
            print(f"[WARNING] Calibration frame {i+1} nav derīgs")
            continue

        valid_ratio_frame = float(np.mean(depth_mm > 0))
        mean_depth_valid = float(np.mean(depth_mm[depth_mm > 0])) if np.any(depth_mm > 0) else 0.0

        print(
            f"[INFO] Calibration frame {i+1}/{CALIBRATION_FRAMES} | "
            f"valid_ratio={valid_ratio_frame:.3f} | mean_depth={mean_depth_valid:.2f} mm"
        )

        if valid_ratio_frame < 0.80:
            print("[WARNING] Pārāk maz derīgu dziļuma pikseļu, kadrs izlaists")
            continue

        frames_list.append(depth_mm)

    if len(frames_list) < 3:
        raise RuntimeError("Kalibrācijai savākts pārāk maz derīgu kadru.")

    print("[PROCESS] Stacking calibration frames")
    depth_stack = np.stack(frames_list, axis=0)

    print("[PROCESS] Computing median depth reference")
    reference_depth_mm = np.median(depth_stack, axis=0).astype(np.float32)

    print("[PROCESS] Fitting plane to median reference")
    a, b, c, rmse, valid_ratio = fit_plane(reference_depth_mm)

    save_reference_depth(reference_depth_mm, REFERENCE_DEPTH_NPY)
    save_calibration_to_json(
        a=a,
        b=b,
        c=c,
        rmse=rmse,
        valid_ratio=valid_ratio,
        width=WIDTH,
        height=HEIGHT,
        filename=CALIBRATION_JSON
    )

    print("\n=== CALIBRATION RESULT ===")
    print(f"a coefficient: {a}")
    print(f"b coefficient: {b}")
    print(f"c coefficient: {c} mm")
    print(f"RMSE (quality): {rmse:.2f} mm")
    print(f"Valid pixel ratio: {valid_ratio:.3f}")
    print("==========================\n")

    return {
        "a": a,
        "b": b,
        "c": c,
        "rmse_mm": rmse,
        "valid_ratio": valid_ratio,
        "width": WIDTH,
        "height": HEIGHT
    }, reference_depth_mm


# =========================
# MAIN
# =========================
def main():
    print("[PROCESS] Creating RealSense pipeline")

    pipeline = rs.pipeline()
    config = rs.config()

    print("[PROCESS] Enabling streams")
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    print("[PROCESS] Starting camera")
    profile = pipeline.start(config)
    print("[PROCESS] Camera started successfully")

    warmup_camera(pipeline, 30)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"[INFO] Depth scale: {depth_scale}")

    align = rs.align(rs.stream.color)
    print("[PROCESS] Alignment object created")

    calibration = load_calibration_from_json(CALIBRATION_JSON)
    reference_depth_mm = load_reference_depth(REFERENCE_DEPTH_NPY)

    a = b = c = None
    reference_plane_mm = None

    if calibration is not None:
        a = calibration["a"]
        b = calibration["b"]
        c = calibration["c"]

        if calibration["width"] == WIDTH and calibration["height"] == HEIGHT:
            reference_plane_mm = build_reference_plane(WIDTH, HEIGHT, a, b, c)
            print("[INFO] References plakne izveidota no JSON koeficientiem")
        else:
            print("[WARNING] Kalibrācijas izšķirtspēja neatbilst pašreizējai straumei")

    print("\nControls:")
    print("  C - calibrate on empty surface")
    print("  S - save current RGB/depth/height/mask/contours")
    print("  R - reload calibration from disk")
    print("  P - toggle reference mode (median map / plane)")
    print("  Q or ESC - exit\n")

    use_reference_depth_map = True

    try:
        while True:
            result = get_aligned_frames(pipeline, align, depth_scale)
            if result[0] is None:
                print("[WARNING] Invalid frame received")
                continue

            _, _, depth_mm, color_image = result

            display_color = color_image.copy()
            height_map = None
            height_vis = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
            object_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
            contours = []

            reference_surface_mm = None

            # Prefer median reference map if available
            if use_reference_depth_map and reference_depth_mm is not None:
                if reference_depth_mm.shape == depth_mm.shape:
                    reference_surface_mm = reference_depth_mm
                else:
                    print("[WARNING] reference_depth shape mismatch")
            elif reference_plane_mm is not None:
                if reference_plane_mm.shape == depth_mm.shape:
                    reference_surface_mm = reference_plane_mm
                else:
                    print("[WARNING] reference_plane shape mismatch")

            if reference_surface_mm is not None:
                height_map = compute_height_map(depth_mm, reference_surface_mm)
                height_vis = visualize_height_map(height_map, MAX_VIS_HEIGHT_MM)

                object_mask = segment_objects(height_map, MIN_OBJECT_HEIGHT_MM)
                object_mask = clean_mask(object_mask)

                # Secondary refinement with RGB
                object_mask = refine_mask_with_rgb(object_mask, color_image)

                contours = find_object_contours(object_mask, MIN_CONTOUR_AREA)
                display_color = draw_contours_with_info(color_image, contours, height_map)

                mode_text = "REF: median map" if (use_reference_depth_map and reference_depth_mm is not None) else "REF: plane"
                cv2.putText(
                    display_color,
                    mode_text,
                    (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
            else:
                cv2.putText(
                    display_color,
                    "No calibration loaded. Press C.",
                    (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

            cv2.imshow("Color", color_image)
            cv2.imshow("Height Map", height_vis)
            cv2.imshow("Object Mask", object_mask)
            cv2.imshow("Contours", display_color)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                print("[PROCESS] Exit requested")
                break

            elif key == ord('c') or key == ord('C'):
                try:
                    calibration, reference_depth_mm = run_calibration(pipeline, align, depth_scale)
                    a = calibration["a"]
                    b = calibration["b"]
                    c = calibration["c"]
                    reference_plane_mm = build_reference_plane(WIDTH, HEIGHT, a, b, c)
                    use_reference_depth_map = True
                except Exception as e:
                    print(f"[ERROR] Calibration failed: {e}")

            elif key == ord('r') or key == ord('R'):
                calibration = load_calibration_from_json(CALIBRATION_JSON)
                reference_depth_mm = load_reference_depth(REFERENCE_DEPTH_NPY)

                if calibration is not None:
                    a = calibration["a"]
                    b = calibration["b"]
                    c = calibration["c"]
                    reference_plane_mm = build_reference_plane(WIDTH, HEIGHT, a, b, c)
                    print("[INFO] Kalibrācija pārlādēta")
                else:
                    a = b = c = None
                    reference_plane_mm = None
                    print("[WARNING] Kalibrācija nav pieejama")

            elif key == ord('p') or key == ord('P'):
                use_reference_depth_map = not use_reference_depth_map
                print(f"[INFO] Reference mode changed: {'median map' if use_reference_depth_map else 'plane'}")

            elif key == ord('s') or key == ord('S'):
                if height_map is None:
                    print("[WARNING] Nav kalibrācijas. Saglabāt height_map nevar.")
                else:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    save_outputs(
                        base_name=timestamp,
                        color_image=color_image,
                        depth_mm=depth_mm,
                        height_map=height_map,
                        mask=object_mask,
                        contour_image=display_color
                    )

    finally:
        print("[PROCESS] Stopping camera")
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[PROCESS] Program finished")


if __name__ == "__main__":
    main()