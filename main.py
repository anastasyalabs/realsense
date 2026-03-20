import pyrealsense2 as rs  # Intel RealSense Python bibliotēka darbam ar kameru
import numpy as np  # NumPy bibliotēka matricām, skaitļošanai un masīvu apstrādei
import cv2  # OpenCV bibliotēka attēlu apstrādei, logiem, kontūrām un morfoloģijai
import json  # JSON bibliotēka kalibrācijas parametru saglabāšanai failā
import os  # OS bibliotēka darbam ar mapēm un failu eksistences pārbaudi
import time  # Laika bibliotēka timestamp un failu nosaukumu ģenerēšanai
from datetime import datetime  # Ērtai cilvēkam saprotama datuma/laika saglabāšanai

# =========================
# CONFIG
# =========================
WIDTH = 640  # Kameras straumes platums pikseļos
HEIGHT = 480  # Kameras straumes augstums pikseļos
FPS = 15  # Kadru skaits sekundē

CALIBRATION_FRAMES = 20  # Cik kadrus izmantot kalibrācijai tukšai virsmai
CALIBRATION_JSON = "calibration.json"  # Faila nosaukums, kur glabāt plaknes koeficientus un kalibrācijas metadatus
REFERENCE_DEPTH_NPY = "reference_depth.npy"  # Faila nosaukums references virsmas dziļuma kartei
OUTPUT_DIR = "output"  # Mape, kur saglabāt rezultātu attēlus un .npy failus

MIN_OBJECT_HEIGHT_MM = 5.0  # Minimālais augstums mm, virs kura pikseli uzskatām par objektu
MAX_VIS_HEIGHT_MM = 300.0  # Maksimālais augstums mm, ko rādīt vizualizācijā kā baltu
MIN_CONTOUR_AREA = 300  # Minimālā kontūras platība pikseļos, lai ignorētu sīkus trokšņus

os.makedirs(OUTPUT_DIR, exist_ok=True)  # Izveido output mapi, ja tās vēl nav; ja ir, kļūda netiek mesta


# =========================
# CALIBRATION SAVE / LOAD
# =========================
def save_calibration_to_json(a, b, c, rmse, valid_ratio, width, height, filename=CALIBRATION_JSON):
    now = datetime.now()  # Paņem pašreizējo lokālo datumu un laiku lasāmā formā

    calibration_data = {  # Izveido vārdnīcu ar visiem kalibrācijas datiem
        "a": float(a),  # Plaknes koeficients pa x asi
        "b": float(b),  # Plaknes koeficients pa y asi
        "c": float(c),  # Plaknes nobīde; aptuvenais attālums līdz virsmai
        "rmse_mm": float(rmse),  # Plaknes pielāgošanas kļūda mm
        "valid_ratio": float(valid_ratio),  # Derīgo dziļuma pikseļu proporcija
        "width": int(width),  # Izmantotais platums kalibrācijas laikā
        "height": int(height),  # Izmantotais augstums kalibrācijas laikā
        "timestamp": int(time.time()),  # Unix timestamp sekundēs kopš 1970. gada
        "timestamp_readable": now.strftime("%Y-%m-%d %H:%M:%S"),  # Cilvēkam lasāms datums/laiks
        "reference_depth_file": REFERENCE_DEPTH_NPY  # Norāde uz failu ar references dziļuma karti
    }

    with open(filename, "w", encoding="utf-8") as f:  # Atver JSON failu rakstīšanai UTF-8 kodējumā
        json.dump(calibration_data, f, indent=4, ensure_ascii=False)  # Saglabā vārdnīcu kā formatētu JSON

    print(f"[INFO] Kalibrācija saglabāta failā: {os.path.abspath(filename)}")  # Izdrukā, kur fails ir saglabāts


def load_calibration_from_json(filename=CALIBRATION_JSON):
    if not os.path.exists(filename):  # Pārbauda, vai kalibrācijas JSON fails eksistē
        print("[INFO] Kalibrācijas JSON fails nav atrasts.")  # Paziņo, ka fails nav atrasts
        return None  # Atgriež None, lai programma zinātu, ka kalibrācijas nav

    with open(filename, "r", encoding="utf-8") as f:  # Atver JSON failu lasīšanai
        calibration_data = json.load(f)  # Nolasa JSON saturu Python vārdnīcā

    print(f"[INFO] Kalibrācija ielādēta no faila: {os.path.abspath(filename)}")  # Paziņo, ka kalibrācija ielādēta
    return calibration_data  # Atgriež ielādēto vārdnīcu


def save_reference_depth(reference_depth_mm, filename=REFERENCE_DEPTH_NPY):
    np.save(filename, reference_depth_mm.astype(np.float32))  # Saglabā references dziļuma karti .npy failā float32 formātā
    print(f"[INFO] References dziļuma karte saglabāta: {os.path.abspath(filename)}")  # Paziņo par saglabāšanu


def load_reference_depth(filename=REFERENCE_DEPTH_NPY):
    if not os.path.exists(filename):  # Pārbauda, vai references dziļuma fails eksistē
        print("[INFO] References dziļuma NPY fails nav atrasts.")  # Ja nav, izdrukā paziņojumu
        return None  # Atgriež None

    reference_depth_mm = np.load(filename).astype(np.float32)  # Ielādē .npy masīvu un pārvērš uz float32
    print(f"[INFO] References dziļuma karte ielādēta: {os.path.abspath(filename)}")  # Paziņo, ka karte ielādēta
    return reference_depth_mm  # Atgriež references dziļuma karti


# =========================
# MATH / SURFACE MODEL
# =========================
def fit_plane(depth_mm):
    print("[PROCESS] Starting plane fitting...")  # Paziņo, ka sākas plaknes pielāgošana

    h, w = depth_mm.shape  # Dabū dziļuma attēla augstumu un platumu
    ys, xs = np.mgrid[0:h, 0:w]  # Izveido y un x koordinātu režģus katram pikselim

    x = xs.flatten().astype(np.float32)  # Saplacina x koordinātas 1D masīvā
    y = ys.flatten().astype(np.float32)  # Saplacina y koordinātas 1D masīvā
    z = depth_mm.flatten().astype(np.float32)  # Saplacina dziļuma vērtības 1D masīvā

    valid = z > 0  # Izveido masku tikai derīgiem dziļuma pikseļiem (0 parasti nozīmē nederīgs)
    x = x[valid]  # Atlasa tikai x koordinātas derīgajiem pikseļiem
    y = y[valid]  # Atlasa tikai y koordinātas derīgajiem pikseļiem
    z = z[valid]  # Atlasa tikai derīgās dziļuma vērtības

    if len(z) < 1000:  # Ja derīgo punktu ir pārāk maz
        raise RuntimeError("Nepietiek derīgu dziļuma punktu plaknes pielāgošanai.")  # Met kļūdu

    A = np.column_stack((x, y, np.ones_like(x)))  # Izveido lineārās sistēmas matricu plaknei z = a*x + b*y + c
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)  # Atrisina mazāko kvadrātu uzdevumu
    a, b, c = coeffs  # Izvelk plaknes koeficientus

    z_pred = a * x + b * y + c  # Aprēķina plaknes prognozētās dziļuma vērtības derīgajiem punktiem
    rmse = np.sqrt(np.mean((z - z_pred) ** 2))  # Aprēķina vidējo kvadrātisko kļūdu mm
    valid_ratio = len(z) / (h * w)  # Derīgo pikseļu proporcija pret visu attēlu

    print("[PROCESS] Plane fitting finished")  # Paziņo, ka plaknes pielāgošana pabeigta
    return float(a), float(b), float(c), float(rmse), float(valid_ratio)  # Atgriež koeficientus un kvalitātes rādītājus


def build_reference_plane(width, height, a, b, c):
    ys, xs = np.mgrid[0:height, 0:width]  # Izveido koordinātas visam attēlam
    reference_plane = a * xs + b * ys + c  # Aprēķina katram pikselim teorētisko virsmas dziļumu pēc plaknes vienādojuma
    return reference_plane.astype(np.float32)  # Atgriež plakni float32 formātā


def compute_height_map(depth_mm, reference_surface_mm):
    height_map = reference_surface_mm - depth_mm  # Augstums = references virsmas dziļums - aktuālais dziļums
    height_map[depth_mm <= 0] = 0  # Ja aktuālais dziļums nav derīgs, augstumu iestata uz 0
    height_map[reference_surface_mm <= 0] = 0  # Ja reference nav derīga, arī augstumu iestata uz 0
    height_map[height_map < 0] = 0  # Visas negatīvās vērtības (zem virsmas) nogriež uz 0
    return height_map.astype(np.float32)  # Atgriež augstuma karti float32 formātā


def visualize_height_map(height_map, max_height_mm=MAX_VIS_HEIGHT_MM):
    vis = np.clip(height_map, 0, max_height_mm)  # Ierobežo augstuma vērtības diapazonā 0..max_height_mm
    vis = (vis / max_height_mm * 255.0).astype(np.uint8)  # Pārvērš mm uz 8-bit grayscale 0..255
    return vis  # Atgriež attēlojamu augstuma karti


# =========================
# SEGMENTATION / CONTOURS
# =========================
def segment_objects(height_map, min_height_mm=MIN_OBJECT_HEIGHT_MM):
    mask = np.zeros_like(height_map, dtype=np.uint8)  # Izveido melnu masku tādā pašā izmērā kā height_map
    mask[height_map > min_height_mm] = 255  # Visi pikseļi virs sliekšņa kļūst balti = objekts
    return mask  # Atgriež bināro objektu masku


def clean_mask(mask):
    kernel_open = np.ones((3, 3), np.uint8)  # Mazs kodols trokšņu noņemšanai
    kernel_close = np.ones((5, 5), np.uint8)  # Nedaudz lielāks kodols caurumu aizpildīšanai

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)  # OPEN noņem sīkus baltus trokšņus
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)  # CLOSE aizpilda mazus melnus caurumus objektos
    return cleaned  # Atgriež attīrītu masku


def refine_mask_with_rgb(mask, color_image):
    """
    RGB is secondary data.
    This keeps the height-map mask as primary, then sharpens edges with RGB gradients.
    """
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)  # Pārveido krāsu attēlu pelēktoņu formātā
    edges = cv2.Canny(gray, 50, 150)  # Atrod malas ar Canny algoritmu

    edge_kernel = np.ones((3, 3), np.uint8)  # Kodols malu paplašināšanai
    edges_dilated = cv2.dilate(edges, edge_kernel, iterations=1)  # Padara malas biezākas un vieglāk izmantojamas

    refined = mask.copy()  # Izveido maskas kopiju, ko uzlabosim
    refined = cv2.bitwise_or(refined, cv2.bitwise_and(edges_dilated, mask))  # Patur tikai tās RGB malas, kas atrodas objektu maskā
    refined = clean_mask(refined)  # Atkārtoti attīra masku pēc malu pievienošanas
    return refined  # Atgriež uzlabotu masku


def find_object_contours(mask, min_area=MIN_CONTOUR_AREA):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Atrod ārējās kontūras binārajā maskā
    filtered = []  # Izveido tukšu sarakstu derīgajām kontūrām

    for cnt in contours:  # Iet cauri katrai atrastajai kontūrai
        area = cv2.contourArea(cnt)  # Aprēķina kontūras laukumu
        if area >= min_area:  # Ja laukums ir pietiekami liels
            filtered.append(cnt)  # Pievieno kontūru derīgo sarakstam

    return filtered  # Atgriež filtrētas kontūras


def draw_contours_with_info(image, contours, height_map=None):
    result = image.copy()  # Izveido kopiju no RGB attēla, lai uz tās zīmētu rezultātus

    for idx, cnt in enumerate(contours, start=1):  # Iet cauri kontūrām, numurējot objektus no 1
        cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)  # Uzzīmē zaļu kontūru ap objektu

        x, y, w, h = cv2.boundingRect(cnt)  # Aprēķina kontūras bounding box
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Uzzīmē zilu taisnstūri ap objektu

        label = f"Obj {idx}"  # Sākotnējais objekta nosaukums

        if height_map is not None:  # Ja pieejama height_map, var aprēķināt objekta augstumu
            object_mask = np.zeros(height_map.shape, dtype=np.uint8)  # Izveido tukšu masku objekta laukuma iegūšanai
            cv2.drawContours(object_mask, [cnt], -1, 255, thickness=-1)  # Aizpilda kontūras iekšpusi baltā krāsā
            object_heights = height_map[object_mask == 255]  # Paņem height_map vērtības tikai objekta iekšienē

            if object_heights.size > 0:  # Ja objektā ir vismaz viens pikselis
                max_h = float(np.max(object_heights))  # Maksimālais augstums objektā
                mean_h = float(np.mean(object_heights))  # Vidējais augstums objektā
                label = f"Obj {idx} | max={max_h:.1f}mm avg={mean_h:.1f}mm"  # Sagatavo uzrakstu ar augstuma informāciju

        cv2.putText(  # Uzraksta informāciju uz attēla
            result,  # Attēls, uz kura zīmēt tekstu
            label,  # Teksta saturs
            (x, max(20, y - 8)),  # Teksta pozīcija virs objekta; neļauj aiziet pārāk augstu
            cv2.FONT_HERSHEY_SIMPLEX,  # Fonta veids
            0.45,  # Fonta izmērs
            (0, 255, 255),  # Teksta krāsa (dzeltenīga)
            1,  # Līnijas biezums
            cv2.LINE_AA  # Antialiasing gludākam tekstam
        )

    return result  # Atgriež attēlu ar kontūrām, taisnstūriem un tekstu


# =========================
# SAVE OUTPUTS
# =========================
def save_outputs(base_name, color_image, depth_mm, height_map, mask, contour_image):
    color_path = os.path.join(OUTPUT_DIR, f"{base_name}_color.png")  # Ceļš RGB attēla saglabāšanai
    depth_path = os.path.join(OUTPUT_DIR, f"{base_name}_depth.npy")  # Ceļš dziļuma matricas saglabāšanai
    height_path = os.path.join(OUTPUT_DIR, f"{base_name}_height.npy")  # Ceļš height_map saglabāšanai
    height_vis_path = os.path.join(OUTPUT_DIR, f"{base_name}_height_vis.png")  # Ceļš height_map vizualizācijai
    mask_path = os.path.join(OUTPUT_DIR, f"{base_name}_mask.png")  # Ceļš objekta maskas saglabāšanai
    contours_path = os.path.join(OUTPUT_DIR, f"{base_name}_contours.png")  # Ceļš gala kontūru attēlam

    cv2.imwrite(color_path, color_image)  # Saglabā RGB attēlu PNG failā
    np.save(depth_path, depth_mm.astype(np.float32))  # Saglabā dziļuma matricu mm kā .npy
    np.save(height_path, height_map.astype(np.float32))  # Saglabā augstuma karti kā .npy
    cv2.imwrite(height_vis_path, visualize_height_map(height_map))  # Saglabā augstuma kartes grayscale vizualizāciju
    cv2.imwrite(mask_path, mask)  # Saglabā bināro objektu masku
    cv2.imwrite(contours_path, contour_image)  # Saglabā attēlu ar kontūrām un marķējumiem

    print(f"[INFO] Saglabāts: {color_path}")  # Izdrukā, ka RGB attēls saglabāts
    print(f"[INFO] Saglabāts: {depth_path}")  # Izdrukā, ka dziļuma .npy saglabāts
    print(f"[INFO] Saglabāts: {height_path}")  # Izdrukā, ka height_map .npy saglabāts
    print(f"[INFO] Saglabāts: {height_vis_path}")  # Izdrukā, ka height_map vizualizācija saglabāta
    print(f"[INFO] Saglabāts: {mask_path}")  # Izdrukā, ka maska saglabāta
    print(f"[INFO] Saglabāts: {contours_path}")  # Izdrukā, ka kontūru attēls saglabāts


# =========================
# REALSENSE HELPERS
# =========================
def get_aligned_frames(pipeline, align, depth_scale):
    frames = pipeline.wait_for_frames()  # Sagaida nākamo kadru komplektu no kameras
    aligned_frames = align.process(frames)  # Pielīdzina depth kadru color koordinātu sistēmai

    depth_frame = aligned_frames.get_depth_frame()  # Izņem pielīdzināto depth kadru
    color_frame = aligned_frames.get_color_frame()  # Izņem color kadru

    if not depth_frame or not color_frame:  # Ja kāds no kadriem nav derīgs
        return None, None, None, None  # Atgriež None vērtības, lai izlaistu šo iterāciju

    depth_raw = np.asanyarray(depth_frame.get_data())  # Pārvērš depth kadru NumPy masīvā
    color_image = np.asanyarray(color_frame.get_data())  # Pārvērš color kadru NumPy masīvā
    depth_mm = depth_raw.astype(np.float32) * depth_scale * 1000.0  # Pārvērš raw dziļumu no sensora vienībām milimetros

    return depth_frame, color_frame, depth_mm, color_image  # Atgriež kadrus un apstrādātos masīvus


def warmup_camera(pipeline, num_frames=30):
    print(f"[PROCESS] Warming up camera ({num_frames} frames)")  # Paziņo, ka sāk sensora iesildīšanu
    for _ in range(num_frames):  # Atkārto norādīto kadru skaitu
        pipeline.wait_for_frames()  # Nolasa un ignorē kadru, lai kamera stabilizētos
    print("[PROCESS] Camera warm-up finished")  # Paziņo, ka iesildīšana pabeigta


# =========================
# CALIBRATION ROUTINE
# =========================
def run_calibration(pipeline, align, depth_scale):
    print("\n[PROCESS] Calibration started")  # Paziņo, ka sākas kalibrācija
    print("[INFO] Nodrošini, ka kamera redz TUKŠU VIRSMU bez objektiem")  # Norāde lietotājam notīrīt galdu

    frames_list = []  # Saraksts, kur glabāt kalibrācijas depth kadrus

    for i in range(CALIBRATION_FRAMES):  # Iet cauri visiem kalibrācijas kadriem
        _, _, depth_mm, _ = get_aligned_frames(pipeline, align, depth_scale)  # Iegūst vienu pielīdzinātu depth kadru mm
        if depth_mm is None:  # Ja kadrs nav derīgs
            print(f"[WARNING] Calibration frame {i+1} nav derīgs")  # Paziņo par kļūdainu kadru
            continue  # Pāriet pie nākamā kadra

        valid_ratio_frame = float(np.mean(depth_mm > 0))  # Aprēķina, kāda daļa pikseļu šajā kadrā ir derīgi
        mean_depth_valid = float(np.mean(depth_mm[depth_mm > 0])) if np.any(depth_mm > 0) else 0.0  # Aprēķina vidējo derīgo dziļumu

        print(  # Izdrukā diagnostiku par katru kalibrācijas kadru
            f"[INFO] Calibration frame {i+1}/{CALIBRATION_FRAMES} | "
            f"valid_ratio={valid_ratio_frame:.3f} | mean_depth={mean_depth_valid:.2f} mm"
        )

        if valid_ratio_frame < 0.80:  # Ja derīgo pikseļu ir pārāk maz
            print("[WARNING] Pārāk maz derīgu dziļuma pikseļu, kadrs izlaists")  # Paziņo, ka kadrs neder
            continue  # Neiekļauj šo kadru kalibrācijā

        frames_list.append(depth_mm)  # Pievieno derīgo kadru sarakstam

    if len(frames_list) < 3:  # Ja pēc filtrēšanas ir palicis pārāk maz kadru
        raise RuntimeError("Kalibrācijai savākts pārāk maz derīgu kadru.")  # Met kļūdu

    print("[PROCESS] Stacking calibration frames")  # Paziņo, ka kadrus apvienos vienā masīvā
    depth_stack = np.stack(frames_list, axis=0)  # Izveido 3D masīvu (kadra_indekss, y, x)

    print("[PROCESS] Computing median depth reference")  # Paziņo, ka rēķinās mediānu
    reference_depth_mm = np.median(depth_stack, axis=0).astype(np.float32)  # Aprēķina mediānas references dziļuma karti

    print("[PROCESS] Fitting plane to median reference")  # Paziņo, ka plakni pielāgos mediānas virsmai
    a, b, c, rmse, valid_ratio = fit_plane(reference_depth_mm)  # Aprēķina plaknes koeficientus un kvalitātes rādītājus

    save_reference_depth(reference_depth_mm, REFERENCE_DEPTH_NPY)  # Saglabā references dziļuma karti failā
    save_calibration_to_json(  # Saglabā koeficientus un metadatus JSON failā
        a=a,
        b=b,
        c=c,
        rmse=rmse,
        valid_ratio=valid_ratio,
        width=WIDTH,
        height=HEIGHT,
        filename=CALIBRATION_JSON
    )

    print("\n=== CALIBRATION RESULT ===")  # Skaidri nodala kalibrācijas rezultātu bloku
    print(f"a coefficient: {a}")  # Izdrukā a koeficientu
    print(f"b coefficient: {b}")  # Izdrukā b koeficientu
    print(f"c coefficient: {c} mm")  # Izdrukā c koeficientu mm
    print(f"RMSE (quality): {rmse:.2f} mm")  # Izdrukā RMSE kvalitātes rādītāju
    print(f"Valid pixel ratio: {valid_ratio:.3f}")  # Izdrukā derīgo pikseļu proporciju
    print("==========================\n")  # Noslēdz rezultātu bloku

    return {  # Atgriež kalibrācijas datus vārdnīcā
        "a": a,
        "b": b,
        "c": c,
        "rmse_mm": rmse,
        "valid_ratio": valid_ratio,
        "width": WIDTH,
        "height": HEIGHT
    }, reference_depth_mm  # Atgriež arī references dziļuma karti


# =========================
# MAIN
# =========================
def main():
    print("[PROCESS] Creating RealSense pipeline")  # Paziņo, ka veido RealSense pipeline

    pipeline = rs.pipeline()  # Izveido RealSense datu pipeline objektu
    config = rs.config()  # Izveido konfigurācijas objektu kamerai

    print("[PROCESS] Enabling streams")  # Paziņo, ka aktivizēs straumes
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)  # Aktivizē depth straumi ar doto izšķirtspēju un FPS
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)  # Aktivizē color straumi BGR formātā

    print("[PROCESS] Starting camera")  # Paziņo, ka ieslēdz kameru
    profile = pipeline.start(config)  # Startē straumes un iegūst profilu
    print("[PROCESS] Camera started successfully")  # Paziņo, ka kamera veiksmīgi startējusi

    warmup_camera(pipeline, 30)  # Stabilizē kameru, nolasot pirmos 30 kadrus

    depth_sensor = profile.get_device().first_depth_sensor()  # Iegūst dziļuma sensoru no ierīces
    depth_scale = depth_sensor.get_depth_scale()  # Iegūst dziļuma mēroga koeficientu, lai raw datus pārvērstu metros
    print(f"[INFO] Depth scale: {depth_scale}")  # Izdrukā dziļuma mērogu

    align = rs.align(rs.stream.color)  # Izveido align objektu, lai pielīdzinātu depth color attēlam
    print("[PROCESS] Alignment object created")  # Paziņo, ka align objekts izveidots

    calibration = load_calibration_from_json(CALIBRATION_JSON)  # Mēģina ielādēt saglabāto kalibrācijas JSON
    reference_depth_mm = load_reference_depth(REFERENCE_DEPTH_NPY)  # Mēģina ielādēt references dziļuma karti

    a = b = c = None  # Sākotnēji plaknes koeficienti nav definēti
    reference_plane_mm = None  # Sākotnēji references plakne nav izveidota

    if calibration is not None:  # Ja JSON kalibrācija tika atrasta
        a = calibration["a"]  # Nolasa a koeficientu
        b = calibration["b"]  # Nolasa b koeficientu
        c = calibration["c"]  # Nolasa c koeficientu

        if calibration["width"] == WIDTH and calibration["height"] == HEIGHT:  # Pārbauda, vai kalibrācijas izmērs sakrīt ar pašreizējo straumi
            reference_plane_mm = build_reference_plane(WIDTH, HEIGHT, a, b, c)  # Izveido plaknes references dziļuma karti
            print("[INFO] References plakne izveidota no JSON koeficientiem")  # Paziņo par veiksmīgu plaknes izveidi
        else:
            print("[WARNING] Kalibrācijas izšķirtspēja neatbilst pašreizējai straumei")  # Brīdina par neatbilstošu izšķirtspēju

    print("\nControls:")  # Izdrukā vadības taustiņu sarakstu
    print("  C - calibrate on empty surface")  # C taustiņš palaiž kalibrāciju
    print("  S - save current RGB/depth/height/mask/contours")  # S saglabā pašreizējo rezultātu komplektu
    print("  R - reload calibration from disk")  # R ielādē kalibrāciju no diska no jauna
    print("  P - toggle reference mode (median map / plane)")  # P pārslēdz starp median references karti un plakni
    print("  Q or ESC - exit\n")  # Q vai ESC aizver programmu

    use_reference_depth_map = True  # Pēc noklusējuma izmanto precīzāko median references dziļuma karti

    try:
        while True:  # Bezgalīgs galvenais cikls tiešraides apstrādei
            result = get_aligned_frames(pipeline, align, depth_scale)  # Iegūst vienu pielīdzinātu kadru komplektu
            if result[0] is None:  # Ja depth frame nav derīgs
                print("[WARNING] Invalid frame received")  # Paziņo par nederīgu kadru
                continue  # Izlaiž šo iterāciju

            _, _, depth_mm, color_image = result  # Paņem tikai depth_mm un color_image no atgrieztā rezultāta

            display_color = color_image.copy()  # Izveido kopiju no krāsu attēla, uz kuras zīmēt rezultātus
            height_map = None  # Sākotnēji augstuma karte nav aprēķināta
            height_vis = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)  # Izveido melnu attēlu height_map vizualizācijai
            object_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)  # Izveido melnu objektu masku
            contours = []  # Tukšs kontūru saraksts

            reference_surface_mm = None  # Pašreizējā references virsma vēl nav izvēlēta

            if use_reference_depth_map and reference_depth_mm is not None:  # Ja ieslēgts median map režīms un karte ir pieejama
                if reference_depth_mm.shape == depth_mm.shape:  # Pārbauda, vai references kartei ir pareizais izmērs
                    reference_surface_mm = reference_depth_mm  # Izmanto references dziļuma karti kā virsmu
                else:
                    print("[WARNING] reference_depth shape mismatch")  # Brīdina par izmēru neatbilstību
            elif reference_plane_mm is not None:  # Citādi, ja ir pieejama plaknes reference
                if reference_plane_mm.shape == depth_mm.shape:  # Pārbauda izmēru sakritību
                    reference_surface_mm = reference_plane_mm  # Izmanto plakni kā virsmu
                else:
                    print("[WARNING] reference_plane shape mismatch")  # Brīdina par izmēru neatbilstību

            if reference_surface_mm is not None:  # Ja kāda reference virsma ir pieejama
                height_map = compute_height_map(depth_mm, reference_surface_mm)  # Aprēķina augstuma karti attiecībā pret virsmu
                height_vis = visualize_height_map(height_map, MAX_VIS_HEIGHT_MM)  # Pārvērš augstuma karti grayscale attēlā

                object_mask = segment_objects(height_map, MIN_OBJECT_HEIGHT_MM)  # Atrod pikseļus, kas ir augstāki par slieksni
                object_mask = clean_mask(object_mask)  # Attīra masku no trokšņiem

                object_mask = refine_mask_with_rgb(object_mask, color_image)  # Papildus uzlabo maskas malas ar RGB informāciju

                contours = find_object_contours(object_mask, MIN_CONTOUR_AREA)  # Atrod filtrētas objektu kontūras
                display_color = draw_contours_with_info(color_image, contours, height_map)  # Uzzīmē kontūras un augstuma informāciju uz RGB attēla

                mode_text = "REF: median map" if (use_reference_depth_map and reference_depth_mm is not None) else "REF: plane"  # Sagatavo tekstu par pašreizējo references režīmu
                cv2.putText(  # Uzraksta references režīmu uz attēla
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
                cv2.putText(  # Ja kalibrācijas nav, parāda paziņojumu uz attēla
                    display_color,
                    "No calibration loaded. Press C.",
                    (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

            cv2.imshow("Color", color_image)  # Parāda neapstrādāto RGB attēlu
            cv2.imshow("Height Map", height_vis)  # Parāda augstuma kartes vizualizāciju
            cv2.imshow("Object Mask", object_mask)  # Parāda bināro objektu masku
            cv2.imshow("Contours", display_color)  # Parāda gala rezultātu ar kontūrām

            key = cv2.waitKey(1) & 0xFF  # Nolasa nospiesto taustiņu, ja tāds ir

            if key == ord('q') or key == 27:  # Ja nospiests Q vai ESC
                print("[PROCESS] Exit requested")  # Paziņo par iziešanu
                break  # Iziet no galvenā cikla

            elif key == ord('c') or key == ord('C'):  # Ja nospiests C
                try:
                    calibration, reference_depth_mm = run_calibration(pipeline, align, depth_scale)  # Palaiž kalibrāciju un dabū references dziļuma karti
                    a = calibration["a"]  # Atjauno a koeficientu
                    b = calibration["b"]  # Atjauno b koeficientu
                    c = calibration["c"]  # Atjauno c koeficientu
                    reference_plane_mm = build_reference_plane(WIDTH, HEIGHT, a, b, c)  # Izveido plaknes references karti
                    use_reference_depth_map = True  # Pēc kalibrācijas atgriežas median map režīmā
                except Exception as e:
                    print(f"[ERROR] Calibration failed: {e}")  # Ja kalibrācija neizdodas, izdrukā kļūdu

            elif key == ord('r') or key == ord('R'):  # Ja nospiests R
                calibration = load_calibration_from_json(CALIBRATION_JSON)  # Pārlādē JSON kalibrāciju
                reference_depth_mm = load_reference_depth(REFERENCE_DEPTH_NPY)  # Pārlādē references dziļuma karti

                if calibration is not None:  # Ja kalibrācija veiksmīgi ielādēta
                    a = calibration["a"]  # Atjauno a
                    b = calibration["b"]  # Atjauno b
                    c = calibration["c"]  # Atjauno c
                    reference_plane_mm = build_reference_plane(WIDTH, HEIGHT, a, b, c)  # Atjauno references plakni
                    print("[INFO] Kalibrācija pārlādēta")  # Paziņo par veiksmīgu pārlādi
                else:
                    a = b = c = None  # Ja kalibrācijas nav, notīra koeficientus
                    reference_plane_mm = None  # Notīra plakni
                    print("[WARNING] Kalibrācija nav pieejama")  # Paziņo, ka kalibrācija nav pieejama

            elif key == ord('p') or key == ord('P'):  # Ja nospiests P
                use_reference_depth_map = not use_reference_depth_map  # Pārslēdz starp median map un plane režīmu
                print(f"[INFO] Reference mode changed: {'median map' if use_reference_depth_map else 'plane'}")  # Izdrukā pašreizējo režīmu

            elif key == ord('s') or key == ord('S'):  # Ja nospiests S
                if height_map is None:  # Ja nav aprēķināta augstuma karte, tātad nav kalibrācijas
                    print("[WARNING] Nav kalibrācijas. Saglabāt height_map nevar.")  # Paziņo, ka saglabāt nevar
                else:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")  # Izveido timestamp failu prefiksam
                    save_outputs(  # Saglabā pašreizējo rezultātu komplektu failos
                        base_name=timestamp,
                        color_image=color_image,
                        depth_mm=depth_mm,
                        height_map=height_map,
                        mask=object_mask,
                        contour_image=display_color
                    )

    finally:
        print("[PROCESS] Stopping camera")  # Paziņo, ka kamera tiks apturēta
        pipeline.stop()  # Aptur RealSense pipeline
        cv2.destroyAllWindows()  # Aizver visus OpenCV logus
        print("[PROCESS] Program finished")  # Paziņo, ka programma beigusies


if __name__ == "__main__":  # Šis bloks izpildās tikai tad, ja failu palaiž tieši
    main()  # Izsauc galveno funkciju