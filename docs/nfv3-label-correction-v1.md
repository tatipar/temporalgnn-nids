# NF-v3 Infiltration Label Correction v1

## Status

**Frozen after automated audit and manual sample review.**

This record applies to the two NF-v3 day-level inputs used for the
Infiltration campaign:

- `cicids2018v3_wed2802.csv`
- `cicids2018v3_thu0103.csv`

The raw input CSVs remain immutable. The corrected CSV, manifest, audit JSON,
and manual-review samples are stored in Google Drive under:

```text
/content/drive/MyDrive/nids-fair-retrain/corrected_data/infiltration_v1/
```

## Rule source and binary target

Rules are implemented in `code/python/utils/nfv3_relabel.py` with version
`cse-cic-ids2018-infiltration-v1`. They are transcribed from
`docs/improved_cse_cic_ids2018_documentation_infiltration.md`.

`binary_target` is the only target used for model training and evaluation:

- `1`: confirmed attack activity, including documented Dropbox Download,
  victim-attacker communication, and NMAP Portscan; all non-benign attacks
  outside the historical `Infilteration` class remain positive.
- `0`: benign traffic, historical `Infilteration` flows that do not match a
  confirmed documented rule, and all `Attempted` flows.

`Attack`, `Label`, `label_corrected_detail`, `binary_target`,
`correction_rule`, and `attempted_category` are labels or provenance metadata
and must never be used as model features.

## Attempted-flow policy

Following the corrected-dataset authors' recommendation, `Attempted` flows
are binary benign and are never a separate machine-learning class.

- Category 4 Dropbox artefacts were relabelled to benign.
- Category 0 was evaluated with `IN_BYTES == 0`, using the documented
  source-to-destination direction. It matched zero rows.
- The `OUT_BYTES == 0` counterfactual also matched zero rows and changed zero
  binary targets. Therefore the corrected output is invariant to this
  directional choice for these inputs.

## Audited outcome

Total flows: **4,180,260**.

| Original target | Corrected target | Flows | Interpretation |
|---:|---:|---:|---|
| 0 | 0 | 3,931,453 | Remained benign |
| 0 | 1 | 60,655 | Original false negatives corrected |
| 1 | 0 | 136,755 | Original false positives corrected |
| 1 | 1 | 51,397 | Remained attack |

The correction changed **197,410 flows (4.7224%)**. Original positives were
188,152 (4.50%); corrected positives are 112,052 (2.68%).

The automated audit passed all checks:

- corrected CSV row count matches the manifest;
- corrected CSV SHA-256 matches the manifest;
- binary and rule counts match the manifest;
- `source_row_id` provenance is contiguous and ordered;
- stored corrected labels replay exactly from the frozen rules.

Manual samples were reviewed for every correction rule:

- `attempted_category_4_to_benign`;
- `confirmed_dropbox_download`;
- `confirmed_nmap_portscan`;
- `confirmed_victim_attacker_communication`;
- `old_infilteration_to_benign`;
- `unchanged`.

## Artifact hashes

Copy the values below verbatim from `nfv3_corrected.manifest.json` after the
final Drive artifact is retained. Do not edit the CSV or regenerate it after
recording these values.

```text
Wednesday input SHA-256: "bebab480e2578b31fa307efdc0dbee1ecc6f9d85406d582a1eaf8242703aba55"
Thursday input SHA-256:  "1d5847a0af491ca52aaf4bb665cbd75353b250687e035644fbe05c6afad6de60"
Corrected CSV SHA-256:  "f8b29797f233e25d51e35493f7309704c3d5bef7eec9238288a1dca0f14e1e36"
Rule version:           cse-cic-ids2018-infiltration-v1
```

Any change to an input CSV, correction rule, attempted-flow policy, or this
binary-target definition creates a new dataset version and requires new
graphs, scaler, hyperparameter selection, and results.
