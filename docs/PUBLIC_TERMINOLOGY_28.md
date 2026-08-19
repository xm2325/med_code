# Public terminology boundary: CTCAE v6.0 / MedDRA 28.0

The repository may use the public **NCI CTCAE v6.0 Clean Copy** workbook. Its
clean-copy worksheet contains a subset with the following source columns:

- `CTCAE v6.0 MedDRA 28.0 LLT Code`
- `CTCAE v6.0 MedDRA 28.0 SOC`
- `CTCAE v6.0 MedDRA 28.0 Term`
- grades 1 through 5, `Definition`, and `Navigational Note`

This is a public CTCAE subset carrying MedDRA 28.0 identifiers. It must never be
described as a public or complete MedDRA 28.0 distribution. The importer retains
the source row, URL, workbook SHA-256, source text, and the `28.0` version pin.
It does not infer or copy an LLT into the Parent PT fields.

## Public workbook import

```python
from cohortcoder.meddra_terminology import NCICTCAEV6Importer

result = NCICTCAEV6Importer("/secure/input/ctcae-v6.0.xlsx").load()
print(result.manifest.to_dict())
print(result.records[0].to_dict())
```

An imported public-only record has `pt_*`, `hlt_*`, `hlgt_*`, and `soc_code`
set to `null`, with `hierarchy_status=PUBLIC_SUBSET_ONLY`. The importer emits a
runtime manifest with the actual workbook fingerprint and row count. The JSON
under `examples/terminology_sources/` is only a source declaration template and
does not claim a digest before a file is imported.

## Optional licensed hierarchy

If an authorised MedDRA 28.0 distribution is available at runtime, the optional
importer can enrich public rows using explicit paths to:

- `llt.asc` for LLT to PT
- `pt.asc` for PT names
- `mdhier.asc` for each PT hierarchy path

```python
from cohortcoder.meddra_terminology import MedDRAAsciiHierarchyImporter

hierarchy = MedDRAAsciiHierarchyImporter(
    terminology_version="28.0",
    llt_path="/secure/meddra/28.0/llt.asc",
    pt_path="/secure/meddra/28.0/pt.asc",
    mdhier_path="/secure/meddra/28.0/mdhier.asc",
)
records = hierarchy.enrich(result.records)
```

Only the primary hierarchy path (`primary_soc_fg=Y`) is attached to the compact
record. The original CTCAE SOC text remains unchanged. Missing LLTs remain
public-only; incomplete or contradictory PT paths fail closed. MedDRA 26.1 and
28.0 inputs cannot be combined.

The `.asc` files, the full derived index, and any subscriber credentials are
runtime inputs. They are excluded from Git and must not be copied into fixtures,
demo HTML, logs, or public API payloads beyond the minimum terms permitted by the
organisation's subscription.

Chinese aliases used elsewhere in the demo are locally curated demo metadata
(`local_demo_curated`), not NCI or MedDRA source terminology.
