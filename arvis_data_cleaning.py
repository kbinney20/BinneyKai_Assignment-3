### Importing required  libraries
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import spectral.io.envi as envi

## Connecting to the AVIRIS data folder/loading in data
FOLDER = Path(
    r"C:\Users\KaiBi\OneDrive\Documents\School\UnityENVM\ESCI_610\Project_genisis"
    r"\AVIRIS-Classic_L2_Reflectance_2154_1-20261030_023851"
)
HDR_FILE = FOLDER / "f170222t01p00r14_corr_v1f4_img.hdr"   # text header (metadata)
BIN_FILE = FOLDER / "f170222t01p00r14_corr_v1f4_img.bin"   # raw pixel data

## Checking the data 
img = envi.open(HDR_FILE, BIN_FILE)
print(img)
print(img.shape)

### Choosing the bands of interest
BANDS = {"green": 20, "nir": 53, "swir": 93,
         "swir2": 136, "nir2": 49, "b900": 57, "b970": 64}

### Creating RAW and Cleaned data folders
RAW = Path("raw_bands")          # untouched copies of the data
CLEAN = Path("cleaned_bands")    # cleaned versions
RAW.mkdir(exist_ok=True)
CLEAN.mkdir(exist_ok=True)

### Creating a log file/definition to record the cleaning process
# ---------- cleaning log ----------
log = []
def add_log(column, issue, action, justification, impact):
    log.append({"Column": column, "Issue": issue, "Action Taken": action,
                "Justification": justification, "Impact": impact})

### Inspecting the datas metadata and reading header information
img = envi.open(HDR_FILE, BIN_FILE)
meta = img.metadata
print("shape (lines, samples, bands):", img.shape)

# =====================================================
# 1. DATA INSPECTION

### Checking the data type and units
print("\n--- Units/format ---")
for key in ["data type", "reflectance scale factor", "data ignore value", "wavelength units"]:
    print(key, "->", meta.get(key))
add_log("All bands (reflectance)",
        "Checked units: float32, no scale factor in header, wavelength units = "
        f"{meta.get('wavelength units')}",
        "No conversion needed",
        "Values are already unitless reflectance (0 to ~1) and wavelengths are in nm",
        "None")

# --- Inconsistent formatting: dates ---
flight_id = HDR_FILE.name.split("t")[0]                      # 'f170222'
acq_date = datetime.strptime(flight_id[1:], "%y%m%d").date().isoformat()
folder_date = datetime.strptime(FOLDER.name.split("-")[-1][:8], "%Y%m%d").date().isoformat()
print("\nAcquisition date (from flight ID):", acq_date, "| folder stamp:", folder_date)
add_log("File/folder name dates",
        f"Two different dates: flight ID gives {acq_date}, folder name gives {folder_date}",
        f"Recorded acquisition date as {acq_date} (YYYY-MM-DD); treated folder date as the product/download date",
        "AVIRIS-Classic flight IDs encode the acquisition date as fYYMMDD",
        "Only affects metadata and citation, not pixel values")

# --- Duplicates: wavelength table ---
wl = np.array(meta["wavelength"], dtype=float)
n_dup_wl = len(wl) - len(np.unique(wl))
backwards = [(int(i), int(i) + 1) for i in np.where(np.diff(wl) < 0)[0]]
print("\nDuplicate wavelengths:", n_dup_wl, "| band pairs where wavelength decreases:", backwards)
add_log("wavelength (band table)",
        f"{n_dup_wl} duplicate wavelengths; wavelength decreases at band pairs {backwards} (detector overlaps)",
        "Left table as is; selected bands by index, not by nearest-wavelength search",
        "Overlaps are a normal instrument feature, and index lookup avoids picking the wrong band of a pair",
        "None, as long as indices are used")


# --- Read in raw bands and save an untouched copy ---
raw = {}
for name, idx in BANDS.items():
    raw[name] = img.read_band(idx)
    raw_file = RAW / f"{name}_raw.npy"
    if not raw_file.exists():                 # never overwrite the raw copy
        np.save(raw_file, raw[name])


### 2. Identifying missing data

# Identifying missing data: one mask for the whole scene, built from the green band
print("\n--- Missing value inspection ---")
for name in BANDS:
    b = raw[name]
    print(f"{name}: min {b.min():.3f}  max {b.max():.3f}  mean {b.mean():.3f}  "
          f"NaN count {int(np.isnan(b).sum())}  -9999 count {int((b == -9999).sum())}")

nodata = raw["green"] < -0.5
print(f"\nNo-data fraction: {nodata.mean():.3f}  "
      f"(values: {np.unique(np.round(raw['green'][nodata], 3))[:5]})")
for name in BANDS:
    same = np.array_equal(nodata, raw[name] < -0.5)
    print(f"  {name}: no-data footprint matches green = {same}")
    if not same:
        print("  WARNING: footprint differs, do not use the shared mask for", name)

all_match = all(np.array_equal(nodata, raw[n] < -0.5) for n in BANDS)
print("All bands share the same footprint:", all_match)

# Looking for Duplicates: repeated image lines (lines with any valid data) ---
# using green band as the reference for valid data
g = raw["green"]
has_data = (g > -0.5).any(axis=1)
lines = g[has_data]
n_dup_lines = len(lines) - len(np.unique(lines, axis=0))
print("\nLines checked:", len(lines), "| duplicate lines:", n_dup_lines)
add_log("green (line duplicates)",
        f"{n_dup_lines} duplicated lines among {len(lines)} lines containing data",
        "None" if n_dup_lines == 0 else "Flagged only",
        "Each line is a separate scan of the ground, so identical lines would be unusual",
        "None")

# Looking for Outliers: IQR + physical limits ---
print("\n--- Outliers (valid pixels only) ---")
stats = {}
for name in BANDS:
    v = raw[name][~nodata]
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_high = int((v > hi_f).sum())
    n_low = int((v < lo_f).sum())
    n_iqr = n_high + n_low
    n_imp = int(((v < -0.05) | (v > 1.2)).sum())
    stats[name] = (q1, q3, iqr, lo_f, hi_f, n_iqr, n_imp, len(v), n_high, n_low)
    print(f"{name}: Q1={q1:.3f} Q3={q3:.3f} IQR={iqr:.3f} fences=({lo_f:.3f}, {hi_f:.3f}) "
          f"| IQR outliers={n_iqr} ({n_iqr/len(v):.2%}) high={n_high} low={n_low} "
          f"| impossible={n_imp}")

### 3. Cleaning the data

# Define the cleaning function once (uses the shared no-data mask built earlier)
def clean(band):
    band = band.astype("float32").copy()          # copy, so raw stays untouched
    band[nodata] = np.nan                         # no-data pixels (value -1)
    band[(band < -0.05) | (band > 1.2)] = np.nan  # physically impossible reflectance
    return band

# Missing data -> flag as NaN
add_log("All bands",
        f"{nodata.mean():.1%} of pixels are no-data (value -1), the corners outside the flight strip. "
        "Identified because raw band means were negative (about -0.35) with a minimum of -1, "
        "a fill value rather than real reflectance. No NaN or -9999 values were present. "
        f"Footprint identical across all seven bands: {all_match}",
        "Replaced with NaN using one shared mask. Not removed or imputed",
        "The gaps are structural (never imaged), not random. Imputing would invent data. "
        "NaN keeps the 5626 x 1242 grid aligned across bands. Identical footprints justify a single mask",
        f"Analysis uses the remaining {(~nodata).mean():.1%} of the scene")

# Clean each band, save it, and log the impossible-value and IQR decisions
for name in BANDS:
    q1, q3, iqr, lo_f, hi_f, n_iqr, n_imp, n_valid, n_high, n_low = stats[name]
    label = f"{name} (band {BANDS[name]}, {wl[BANDS[name]]:.0f} nm)"

    arr = clean(raw[name])
    np.save(CLEAN / f"{name}.npy", arr)
    print("saved", name, "band", BANDS[name],
          "valid fraction:", round(float((~np.isnan(arr)).mean()), 4))

    add_log(label,
            f"{n_imp} valid-area pixels outside physical reflectance range (<-0.05 or >1.2)",
            "Set to NaN",
            "Values substantially outside the expected reflectance range (0-1) were considered nonphysical for this analysis and treated as artifacts.",
            f"{n_imp/n_valid:.4%} of valid pixels removed")

    add_log(label,
            f"IQR method (Q1={q1:.3f}, Q3={q3:.3f}, IQR={iqr:.3f}) flags {n_iqr} pixels "
            f"({n_iqr/n_valid:.2%}) outside [{lo_f:.3f}, {hi_f:.3f}]: "
            f"{n_high} above the upper fence, {n_low} below the lower fence",
            "Kept (not removed or capped)",
            "Distribution is right-skewed: a large dark mass (water/shadow) pulls Q1 near zero, "
            "so the IQR rule flags bright but physically plausible surfaces (soil, built areas, "
            "bright vegetation) that are real land cover. Values beyond the physical limits "
            "are handled separately in the previous entry",
            "No change to the data. Preserves the contrast the water indices depend on")

np.save(CLEAN / "nodata_mask.npy", nodata)

# SAVE THE LOG
log_df = pd.DataFrame(log)
log_df.to_csv("cleaning_log.csv", index=False)
print("\nSaved cleaning_log.csv with", len(log_df), "entries")
print(log_df.to_string())



