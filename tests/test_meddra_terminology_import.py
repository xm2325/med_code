from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from cohortcoder.meddra_terminology import (
    MEDDRA_HIERARCHY_ENRICHED,
    PUBLIC_SUBSET_ONLY,
    MedDRAAsciiHierarchyImporter,
    NCICTCAEV6Importer,
)


def _write_minimal_ctcae_xlsx(path: Path, *, include_definition: bool = True) -> None:
    headers = [
        "CTCAE v6.0 MedDRA 28.0 LLT Code",
        "CTCAE v6.0 MedDRA 28.0 SOC",
        "CTCAE v6.0 MedDRA 28.0 Term",
        "Grade 1 \u00a0\u00a0",
        "Grade 2 \u00a0\u00a0",
        "Grade 3 \u00a0\u00a0",
        "Grade 4 \u00a0\u00a0",
        "Grade 5 \u00a0\u00a0",
    ]
    if include_definition:
        headers.extend(["Definition", "Navigational Note"])
    row = [
        "10002272",
        "Blood and lymphatic system disorders",
        "Anemia",
        "Mild description",
        "Moderate description",
        "Severe description",
        "Life-threatening description",
        "Death",
    ]
    if include_definition:
        row.extend(["Public test definition", "-"])

    def xml_row(number: int, values: list[str]) -> str:
        cells = "".join(
            f'<c r="{chr(65 + index)}{number}" t="inlineStr"><is><t>{value}</t></is></c>'
            for index, value in enumerate(values)
        )
        return f'<row r="{number}">{cells}</row>'

    content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>"""
    root_relationships = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""
    workbook = """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="CTCAE v6.0 Clean Copy" sheetId="1" r:id="rId1"/></sheets>
</workbook>"""
    workbook_relationships = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"""
    worksheet = f"""<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
{xml_row(1, headers)}{xml_row(2, row)}
</sheetData></worksheet>"""
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_relationships)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_relationships)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)


def test_public_ctcae_import_preserves_source_fields_without_inventing_pt(tmp_path: Path) -> None:
    workbook = tmp_path / "ctcae-v6.0.xlsx"
    _write_minimal_ctcae_xlsx(workbook)

    result = NCICTCAEV6Importer(workbook).load()

    assert result.manifest.terminology_version == "28.0"
    assert result.manifest.scope == "NCI_CTCAE_V6_PUBLIC_SUBSET"
    assert result.manifest.record_count == 1
    assert result.manifest.source_sha256 == sha256(workbook.read_bytes()).hexdigest()
    record = result.records[0]
    assert record.llt_code == "10002272"
    assert record.llt_term == "Anemia"
    assert record.soc_term == "Blood and lymphatic system disorders"
    assert record.navigational_note == "-"
    assert record.grades[0].description == "Mild description"
    assert record.pt_code is None
    assert record.pt_term is None
    assert record.hierarchy_status == PUBLIC_SUBSET_ONLY


def test_public_import_requires_the_official_clean_copy_headers(tmp_path: Path) -> None:
    workbook = tmp_path / "ctcae-v6.0.xlsx"
    _write_minimal_ctcae_xlsx(workbook, include_definition=False)

    with pytest.raises(ValueError, match="missing required headers.*Definition"):
        NCICTCAEV6Importer(workbook).load()


def _write_ascii_hierarchy(directory: Path) -> tuple[Path, Path, Path]:
    llt = directory / "llt.asc"
    pt = directory / "pt.asc"
    hierarchy = directory / "mdhier.asc"
    llt.write_text("10002272$Anemia$10002272$$$$$$$$Y$\n", encoding="utf-8")
    pt.write_text("10002272$Anemia$$10005329$$$$$$$$\n", encoding="utf-8")
    hierarchy.write_text(
        "10002272$10018923$10018921$10005329$Anemia$Fixture HLT$Fixture HLGT$"
        "Blood and lymphatic system disorders$Blood$$10005329$Y$\n",
        encoding="utf-8",
    )
    return llt, pt, hierarchy


def test_optional_ascii_importer_adds_only_supplied_28_hierarchy(tmp_path: Path) -> None:
    workbook = tmp_path / "ctcae-v6.0.xlsx"
    _write_minimal_ctcae_xlsx(workbook)
    record = NCICTCAEV6Importer(workbook).load().records[0]
    llt, pt, hierarchy = _write_ascii_hierarchy(tmp_path)

    importer = MedDRAAsciiHierarchyImporter(
        terminology_version="28.0",
        llt_path=llt,
        pt_path=pt,
        mdhier_path=hierarchy,
    )
    enriched = importer.enrich([record])[0]

    assert enriched.pt_code == "10002272"
    assert enriched.pt_term == "Anemia"
    assert enriched.hlt_code == "10018923"
    assert enriched.hlgt_code == "10018921"
    assert enriched.soc_code == "10005329"
    assert enriched.soc_term == record.soc_term
    assert enriched.hierarchy_status == MEDDRA_HIERARCHY_ENRICHED
    assert enriched.hierarchy_source_sha256 == importer.source_sha256


def test_ascii_importer_rejects_cross_version_or_implicit_path(tmp_path: Path) -> None:
    llt, pt, hierarchy = _write_ascii_hierarchy(tmp_path)
    with pytest.raises(ValueError, match="only MedDRA 28.0"):
        MedDRAAsciiHierarchyImporter(
            terminology_version="26.1",
            llt_path=llt,
            pt_path=pt,
            mdhier_path=hierarchy,
        )

    renamed = tmp_path / "not-llt.asc"
    renamed.write_text(llt.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="explicit path to llt.asc"):
        MedDRAAsciiHierarchyImporter(
            terminology_version="28.0",
            llt_path=renamed,
            pt_path=pt,
            mdhier_path=hierarchy,
        )
