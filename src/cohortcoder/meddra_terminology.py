"""Versioned importers for the public CTCAE subset and optional MedDRA hierarchy.

The NCI workbook is a public CTCAE artefact that exposes a subset of MedDRA 28.0
LLT codes, terms, and SOC names.  It is not a full MedDRA distribution.  The
optional ASCII importer deliberately requires user-supplied MedDRA 28.0 files;
the files and their derived full terminology must not be committed to this
repository.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
import re
from typing import Iterable, Iterator
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile


CTCAE_V6_MEDDRA_VERSION = "28.0"
CTCAE_V6_SHEET = "CTCAE v6.0 Clean Copy"
CTCAE_V6_SOURCE_NAME = "NCI CTCAE v6.0 Clean Copy"
CTCAE_V6_SOURCE_URL = (
    "https://dctd.cancer.gov/research/ctep-trials/trial-development/ctcae-v6.0.xlsx"
)

_HEADER_LLT_CODE = "CTCAE v6.0 MedDRA 28.0 LLT Code"
_HEADER_SOC = "CTCAE v6.0 MedDRA 28.0 SOC"
_HEADER_TERM = "CTCAE v6.0 MedDRA 28.0 Term"
_REQUIRED_HEADERS = (
    _HEADER_LLT_CODE,
    _HEADER_SOC,
    _HEADER_TERM,
    "Grade 1",
    "Grade 2",
    "Grade 3",
    "Grade 4",
    "Grade 5",
    "Definition",
    "Navigational Note",
)

PUBLIC_SUBSET_ONLY = "PUBLIC_SUBSET_ONLY"
MEDDRA_HIERARCHY_ENRICHED = "MEDDRA_28_0_HIERARCHY_ENRICHED"


@dataclass(frozen=True)
class GradeDefinition:
    grade: int
    description: str | None

    def __post_init__(self) -> None:
        if self.grade not in range(1, 6):
            raise ValueError("grade must be an integer from 1 through 5")


@dataclass(frozen=True)
class PublicTerminologyRecord:
    """One unmodified row from the public CTCAE workbook, plus optional hierarchy."""

    llt_code: str
    llt_term: str
    soc_term: str
    definition: str | None
    grades: tuple[GradeDefinition, ...]
    navigational_note: str | None = None
    pt_code: str | None = None
    pt_term: str | None = None
    hlt_code: str | None = None
    hlt_term: str | None = None
    hlgt_code: str | None = None
    hlgt_term: str | None = None
    soc_code: str | None = None
    source_name: str = CTCAE_V6_SOURCE_NAME
    source_url: str = CTCAE_V6_SOURCE_URL
    terminology_version: str = CTCAE_V6_MEDDRA_VERSION
    source_sha256: str = ""
    source_row: int | None = None
    hierarchy_status: str = PUBLIC_SUBSET_ONLY
    hierarchy_source_sha256: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"\d{8}", self.llt_code):
            raise ValueError(f"invalid MedDRA LLT code: {self.llt_code!r}")
        if not self.llt_term.strip() or not self.soc_term.strip():
            raise ValueError("LLT term and SOC term must be non-empty")
        if self.terminology_version != CTCAE_V6_MEDDRA_VERSION:
            raise ValueError("public CTCAE records must remain pinned to MedDRA 28.0")
        if self.source_sha256 and not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
        if self.hierarchy_status == PUBLIC_SUBSET_ONLY:
            hierarchy_values = (
                self.pt_code,
                self.pt_term,
                self.hlt_code,
                self.hlt_term,
                self.hlgt_code,
                self.hlgt_term,
                self.soc_code,
                self.hierarchy_source_sha256,
            )
            if any(value is not None for value in hierarchy_values):
                raise ValueError("public-subset records cannot contain inferred hierarchy fields")
        elif self.hierarchy_status == MEDDRA_HIERARCHY_ENRICHED:
            hierarchy_values = (
                self.pt_code,
                self.pt_term,
                self.hlt_code,
                self.hlt_term,
                self.hlgt_code,
                self.hlgt_term,
                self.soc_code,
                self.hierarchy_source_sha256,
            )
            if any(value is None or not str(value).strip() for value in hierarchy_values):
                raise ValueError("enriched records require a complete primary hierarchy path")
            if not re.fullmatch(r"[0-9a-f]{64}", str(self.hierarchy_source_sha256)):
                raise ValueError("hierarchy_source_sha256 must be a lowercase SHA-256 digest")
        else:
            raise ValueError(f"unsupported hierarchy_status: {self.hierarchy_status!r}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TerminologyManifest:
    source_name: str
    source_url: str
    terminology_version: str
    scope: str
    source_sha256: str
    record_count: int
    worksheet: str
    required_headers: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CTCAEImportResult:
    records: tuple[PublicTerminologyRecord, ...]
    manifest: TerminologyManifest


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_cell(value: object) -> str:
    return str(value or "").replace("\xa0", " ").strip()


def _optional_cell(value: object) -> str | None:
    text = _normalise_cell(value)
    return text or None


def _column_number(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference.upper())
    if not letters:
        raise ValueError(f"invalid XLSX cell reference: {reference!r}")
    result = 0
    for char in letters.group(0):
        result = result * 26 + ord(char) - ord("A") + 1
    return result - 1


def _xlsx_rows(path: Path, sheet_name: str) -> Iterator[list[str]]:
    """Read plain cell values from an XLSX with only the standard library."""

    spreadsheet_ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    relationship_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    package_ns = "{http://schemas.openxmlformats.org/package/2006/relationships}"
    try:
        archive = ZipFile(path)
    except (BadZipFile, OSError) as exc:
        raise ValueError(f"not a readable XLSX file: {path}") from exc

    with archive:
        try:
            workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
            relationships = ElementTree.fromstring(
                archive.read("xl/_rels/workbook.xml.rels")
            )
        except (KeyError, ElementTree.ParseError) as exc:
            raise ValueError(f"invalid XLSX workbook structure: {path}") from exc

        relationship_targets = {
            relation.attrib["Id"]: relation.attrib["Target"]
            for relation in relationships.findall(f"{package_ns}Relationship")
        }
        worksheet_path: str | None = None
        for sheet in workbook.findall(f".//{spreadsheet_ns}sheet"):
            if sheet.attrib.get("name") == sheet_name:
                target = relationship_targets.get(sheet.attrib[f"{relationship_ns}id"])
                if target:
                    worksheet_path = "xl/" + target.lstrip("/")
                break
        if worksheet_path is None:
            raise ValueError(f"required worksheet {sheet_name!r} not found")

        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = [
                "".join(node.text or "" for node in item.iter(f"{spreadsheet_ns}t"))
                for item in root.findall(f"{spreadsheet_ns}si")
            ]

        worksheet = ElementTree.fromstring(archive.read(worksheet_path))
        for row in worksheet.findall(f".//{spreadsheet_ns}row"):
            values: dict[int, str] = {}
            for cell in row.findall(f"{spreadsheet_ns}c"):
                column = _column_number(cell.attrib.get("r", ""))
                cell_type = cell.attrib.get("t")
                if cell_type == "inlineStr":
                    value = "".join(
                        node.text or "" for node in cell.iter(f"{spreadsheet_ns}t")
                    )
                else:
                    value_node = cell.find(f"{spreadsheet_ns}v")
                    value = value_node.text if value_node is not None else ""
                    if cell_type == "s" and value:
                        value = shared_strings[int(value)]
                values[column] = value or ""
            if values:
                yield [values.get(index, "") for index in range(max(values) + 1)]


class NCICTCAEV6Importer:
    """Import the official public CTCAE v6.0 clean-copy XLSX without altering rows."""

    def __init__(self, source_path: str | Path, *, sheet_name: str = CTCAE_V6_SHEET):
        self.source_path = Path(source_path).expanduser()
        self.sheet_name = sheet_name

    def load(self) -> CTCAEImportResult:
        if not self.source_path.is_file():
            raise FileNotFoundError(f"CTCAE workbook not found: {self.source_path}")
        if self.source_path.suffix.casefold() != ".xlsx":
            raise ValueError("CTCAE source_path must identify an .xlsx file")

        rows = iter(_xlsx_rows(self.source_path, self.sheet_name))
        try:
            headers = [_normalise_cell(value) for value in next(rows)]
        except StopIteration as exc:
            raise ValueError("CTCAE clean-copy worksheet is empty") from exc
        missing = [header for header in _REQUIRED_HEADERS if header not in headers]
        if missing:
            raise ValueError(f"CTCAE workbook is missing required headers: {missing}")
        columns = {header: index for index, header in enumerate(headers)}
        source_hash = _file_sha256(self.source_path)

        def cell(row: list[str], header: str) -> str:
            index = columns.get(header)
            return row[index] if index is not None and index < len(row) else ""

        records: list[PublicTerminologyRecord] = []
        seen_codes: set[str] = set()
        for source_row, row in enumerate(rows, start=2):
            code = _normalise_cell(cell(row, _HEADER_LLT_CODE))
            if not code and not any(_normalise_cell(value) for value in row):
                continue
            if not re.fullmatch(r"\d{8}", code):
                raise ValueError(f"row {source_row} has an invalid LLT code: {code!r}")
            if code in seen_codes:
                raise ValueError(f"duplicate LLT code at row {source_row}: {code}")
            seen_codes.add(code)
            grades = tuple(
                GradeDefinition(
                    grade=grade,
                    description=_optional_cell(cell(row, f"Grade {grade}")),
                )
                for grade in range(1, 6)
            )
            records.append(
                PublicTerminologyRecord(
                    llt_code=code,
                    llt_term=_normalise_cell(cell(row, _HEADER_TERM)),
                    soc_term=_normalise_cell(cell(row, _HEADER_SOC)),
                    definition=_optional_cell(cell(row, "Definition")),
                    grades=grades,
                    navigational_note=_optional_cell(cell(row, "Navigational Note")),
                    source_sha256=source_hash,
                    source_row=source_row,
                )
            )

        manifest = TerminologyManifest(
            source_name=CTCAE_V6_SOURCE_NAME,
            source_url=CTCAE_V6_SOURCE_URL,
            terminology_version=CTCAE_V6_MEDDRA_VERSION,
            scope="NCI_CTCAE_V6_PUBLIC_SUBSET",
            source_sha256=source_hash,
            record_count=len(records),
            worksheet=self.sheet_name,
            required_headers=_REQUIRED_HEADERS,
        )
        return CTCAEImportResult(records=tuple(records), manifest=manifest)


@dataclass(frozen=True)
class _Hierarchy:
    pt_code: str
    pt_term: str
    hlt_code: str
    hlt_term: str
    hlgt_code: str
    hlgt_term: str
    soc_code: str
    soc_term: str


def _asc_rows(path: Path, minimum_fields: int) -> Iterator[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\r\n").split("$")
            if fields and fields[-1] == "":
                fields.pop()
            if len(fields) < minimum_fields:
                raise ValueError(
                    f"{path.name} row {row_number} has {len(fields)} fields; "
                    f"expected at least {minimum_fields}"
                )
            yield fields


class MedDRAAsciiHierarchyImporter:
    """Read a user-supplied MedDRA 28.0 hierarchy without persisting its contents."""

    def __init__(
        self,
        *,
        terminology_version: str,
        llt_path: str | Path,
        pt_path: str | Path,
        mdhier_path: str | Path,
    ) -> None:
        if terminology_version != CTCAE_V6_MEDDRA_VERSION:
            raise ValueError("only MedDRA 28.0 hierarchy files can enrich this public subset")
        self.terminology_version = terminology_version
        self.llt_path = self._validated_path(llt_path, "llt.asc")
        self.pt_path = self._validated_path(pt_path, "pt.asc")
        self.mdhier_path = self._validated_path(mdhier_path, "mdhier.asc")

    @staticmethod
    def _validated_path(value: str | Path, expected_name: str) -> Path:
        path = Path(value).expanduser()
        if path.name.casefold() != expected_name:
            raise ValueError(f"expected an explicit path to {expected_name}, got {path.name!r}")
        if not path.is_file():
            raise FileNotFoundError(f"required MedDRA hierarchy file not found: {path}")
        return path

    @property
    def source_sha256(self) -> str:
        digest = sha256()
        for path in (self.llt_path, self.pt_path, self.mdhier_path):
            digest.update(path.name.encode("ascii"))
            digest.update(bytes.fromhex(_file_sha256(path)))
        return digest.hexdigest()

    def enrich(
        self, records: Iterable[PublicTerminologyRecord]
    ) -> tuple[PublicTerminologyRecord, ...]:
        llt_to_pt: dict[str, str] = {}
        for fields in _asc_rows(self.llt_path, 3):
            llt_code, pt_code = fields[0].strip(), fields[2].strip()
            if llt_code in llt_to_pt and llt_to_pt[llt_code] != pt_code:
                raise ValueError(f"LLT {llt_code} maps to more than one PT")
            llt_to_pt[llt_code] = pt_code

        pt_terms = {fields[0].strip(): fields[1].strip() for fields in _asc_rows(self.pt_path, 4)}
        primary_hierarchy: dict[str, _Hierarchy] = {}
        for fields in _asc_rows(self.mdhier_path, 12):
            if fields[11].strip().upper() != "Y":
                continue
            pt_code = fields[0].strip()
            hierarchy = _Hierarchy(
                pt_code=pt_code,
                pt_term=fields[4].strip(),
                hlt_code=fields[1].strip(),
                hlt_term=fields[5].strip(),
                hlgt_code=fields[2].strip(),
                hlgt_term=fields[6].strip(),
                soc_code=fields[3].strip(),
                soc_term=fields[7].strip(),
            )
            if pt_code in primary_hierarchy and primary_hierarchy[pt_code] != hierarchy:
                raise ValueError(f"PT {pt_code} has more than one primary hierarchy path")
            primary_hierarchy[pt_code] = hierarchy

        hierarchy_hash = self.source_sha256
        enriched: list[PublicTerminologyRecord] = []
        for record in records:
            if record.terminology_version != self.terminology_version:
                raise ValueError(
                    f"record {record.llt_code} uses MedDRA {record.terminology_version}, "
                    f"not {self.terminology_version}"
                )
            pt_code = llt_to_pt.get(record.llt_code)
            if pt_code is None:
                enriched.append(record)
                continue
            pt_term = pt_terms.get(pt_code)
            hierarchy = primary_hierarchy.get(pt_code)
            if not pt_term or hierarchy is None:
                raise ValueError(f"incomplete hierarchy for LLT {record.llt_code} / PT {pt_code}")
            if hierarchy.pt_term != pt_term:
                raise ValueError(f"PT term disagreement for {pt_code}")
            enriched.append(
                replace(
                    record,
                    pt_code=pt_code,
                    pt_term=pt_term,
                    hlt_code=hierarchy.hlt_code,
                    hlt_term=hierarchy.hlt_term,
                    hlgt_code=hierarchy.hlgt_code,
                    hlgt_term=hierarchy.hlgt_term,
                    soc_code=hierarchy.soc_code,
                    hierarchy_status=MEDDRA_HIERARCHY_ENRICHED,
                    hierarchy_source_sha256=hierarchy_hash,
                )
            )
        return tuple(enriched)
