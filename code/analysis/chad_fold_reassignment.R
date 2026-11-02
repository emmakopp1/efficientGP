# ============================================================================
# Chad CV fold re-assignment (STEP 3)
# ============================================================================
# Rewrites cv_ind in data/new/chad/chad-weekly-with-features.csv so that:
#   - fold 0 is excluded from testing (always in train): N'Djamena (capital)
#     and Logone Oriental (poorly calibrated by the GP model across repeated
#     CV runs regardless of which fold it was placed in; excluded from
#     testing rather than left to distort whichever fold it lands in)
#   - Batha and Guera are manually swapped between quadrants: Batha moves
#     from its natural North quadrant to East, and Guera moves from its
#     natural East quadrant to South (both sit close to quadrant boundaries;
#     see STEP 8 in visualise_tcd_result.R), so that South and East each
#     keep 5 regions after Logone Oriental (South) is removed into fold 0
#   - folds 1 to 5 each contain exactly one region from each of the 4
#     geographic quadrants (North, South, East, West), mirroring the
#     Nigeria nc/ne/nw/s x cv_ind design
#
# Region -> fold assignment within each quadrant is randomised with a fixed
# seed, so the result is reproducible across runs (unlike the original Chad
# aggregation script, which re-shuffled on every execution with no seed).
#
# All other columns of chad-weekly-with-features.csv are left untouched;
# only cv_ind is overwritten.
# ============================================================================

library(tidyverse)
library(here)

DATA_DIR <- file.path(here(), "data", "new", "chad")
CAPITAL <- "Ndjamzna" # N'Djamena, as spelled in the Chad CSVs
EXCLUDED_REGION <- "Logone Oriental" # poorly calibrated across folds, excluded from testing
SWAP_TO_EAST <- "Batha" # North region swapped into East (was swapped to South)
SWAP_TO_SOUTH <- "Guera" # East region swapped into South (was swapped to East)
SEED <- 42

df_static <- read_csv(file.path(DATA_DIR, "chad-static-data.csv"), show_col_types = FALSE)
df_weekly <- read_csv(file.path(DATA_DIR, "chad-weekly-with-features.csv"), show_col_types = FALSE)

# ---- Quadrant classification (same logic as STEP 8 in visualise_tcd_result.R) ----
df_geo <- df_static |>
  filter(!(adm1_name %in% c("Tibesti", CAPITAL)))

lat_center <- median(df_geo$Lat)
lon_center <- median(df_geo$Lon)

df_geo <- df_geo |>
  mutate(
    dlat = Lat - lat_center,
    dlon = Lon - lon_center,
    quadrant = case_when(
      abs(dlat) > abs(dlon) & dlat > 0 ~ "North",
      abs(dlat) > abs(dlon) & dlat <= 0 ~ "South",
      dlon > 0 ~ "East",
      TRUE ~ "West"
    ),
    # manual override: Batha and Guera are swapped between quadrants (Batha
    # North -> East, Guera East -> South) to keep 5 regions per quadrant
    # after removing Logone Oriental (South) into fold 0
    quadrant = case_when(
      adm1_name == SWAP_TO_EAST  ~ "East",
      adm1_name == SWAP_TO_SOUTH ~ "South",
      TRUE ~ quadrant
    )
  )

# ---- Fold 0: capital + excluded region ----
fold0_regions <- c(CAPITAL, EXCLUDED_REGION)

# ---- Folds 1-5: one region per quadrant, randomised with a fixed seed ----
df_remain <- df_geo |> filter(!(adm1_name %in% fold0_regions))

set.seed(SEED)
fold_map <- df_remain |>
  group_by(quadrant) |>
  mutate(cv_ind = sample(seq_len(n()))) |>
  ungroup() |>
  select(adm1_name, quadrant, cv_ind)

fold_map <- bind_rows(
  fold_map,
  tibble(adm1_name = fold0_regions, quadrant = NA_character_, cv_ind = 0)
) |>
  arrange(cv_ind, adm1_name)

cat("Region -> fold assignment:\n")
print(fold_map, n = Inf)
cat("\n")
print(count(fold_map, cv_ind))

# ---- Sanity checks ----
stopifnot(
  "Every region must be assigned exactly once" =
    n_distinct(fold_map$adm1_name) == nrow(fold_map),
  "All regions present in chad-weekly-with-features.csv must be assigned" =
    setequal(fold_map$adm1_name, unique(df_weekly$adm1_name)),
  "Each of folds 1-5 must contain exactly one region per quadrant" =
    fold_map |>
      filter(cv_ind != 0) |>
      count(cv_ind, quadrant) |>
      pull(n) |>
      (\(x) all(x == 1))()
)

# ---- Overwrite cv_ind in the weekly dataset, leave everything else as-is ----
region_to_fold <- set_names(fold_map$cv_ind, fold_map$adm1_name)

df_weekly_updated <- df_weekly |>
  mutate(cv_ind = unname(region_to_fold[adm1_name]))

write_csv(df_weekly_updated, file.path(DATA_DIR, "chad-weekly-with-features.csv"))

cat("\nUpdated cv_ind written to:", file.path(DATA_DIR, "chad-weekly-with-features.csv"), "\n")
