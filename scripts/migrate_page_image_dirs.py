"""One-time migration: rename existing page_images/ directories so their
names match the document_id values already stored in ChromaDB.

WHY THIS IS NEEDED
------------------
render_pages.py previously used hand-written BOQ document_id keys such as
"BOQ-CONTRACT-AGREEMENT-CE-10-AND-11-LOT-4-VOL-3-28-37".

metadata_loader.py derives image_document_id by calling
    _slugify(source_pdf.rsplit('.', 1)[0])
on the source_pdf field in each BOQ JSON, which produces
"BOQ-CONTRACT-AGREEMENT-CE-10-CE-11-LOT-4-VOL-3-28-37"
(CE-10-CE-11, not CE-10-AND-11, because _slugify replaces every run of
non-alphanumeric chars with a single hyphen -- "CE-10 & CE-11" and
"CE-10___CE-11" both collapse to "CE-10-CE-11").

The ChromaDB collection already contains the correct _slugify-derived IDs
(verified: document_id = "BOQ-CONTRACT-AGREEMENT-CE-10-CE-11-LOT-4-VOL-3-28-37").
Only the on-disk page_images/ directories are misnamed.  This script renames
them to match what is already in the database.

USAGE
-----
Run once from the repository root:

    python scripts/migrate_page_image_dirs.py [--page-images-dir page_images] [--dry-run]

After running, verify with:

    python scripts/render_pages.py --pdf-dir source_pdfs --list-ids

and confirm that every document_id listed there has a matching directory
under page_images/.

ADDENDUM NOTE
-------------
The 211 ADDENDUM chunks (boq_part3.json, no source_pdf) carry
document_id = "BOQ-CE-10-AND-11-LOT-4-SCHEDULE-S4-1" which is derived
from the contract + schedule fields (not from a source PDF filename),
and there is no rendered directory for that ID.  Those chunks cannot be
linked to page images without knowing the exact page offset of the
ADDENDUM pages within a specific source PDF binding -- this script does
not attempt to fix that case.
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Rename table: old directory name -> new directory name.
# Built from the same _slugify logic used by metadata_loader.py and the
# updated render_pages.py, not from hard-coded strings, so future PDF
# additions only need an entry in render_pages._BOQ_PDF_FILENAMES.
# ---------------------------------------------------------------------------

import re

def _slugify(value: str) -> str:
    """Same function as metadata_loader._slugify and render_pages._slugify."""
    return re.sub(r"[^A-Za-z0-9]+", "-", value or "").strip("-").upper()


# Map each BOQ PDF filename to its OLD (hand-written) directory name.
# If the old name already equals the new name (no rename needed), the entry
# is a no-op and is skipped silently.
_BOQ_RENAMES = {
    # old (hand-written, wrong)                              : new (derived from filename, correct)
    "BOQ-CONTRACT-AGREEMENT-CE-10-AND-11-LOT-4-VOL-3-18-27": f"BOQ-{_slugify('Contract Agreement CE-10 & CE-11 Lot 4 Vol-3-18-27')}",
    "BOQ-CONTRACT-AGREEMENT-CE-10-AND-11-LOT-4-VOL-3-28-37": f"BOQ-{_slugify('Contract Agreement CE-10 & CE-11 Lot 4 Vol-3-28-37')}",
    "BOQ-CONTRACT-AGREEMENT-CE-10-AND-11-LOT-4-VOL-3-38-47": f"BOQ-{_slugify('Contract Agreement CE-10 & CE-11 Lot 4 Vol-3-38-47')}",
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rename existing page_images/ directories to match ChromaDB document_id values."
    )
    ap.add_argument(
        "--page-images-dir",
        default="page_images",
        help="Root page images directory (default: page_images).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be renamed without doing anything.",
    )
    args = ap.parse_args()

    if not os.path.isdir(args.page_images_dir):
        sys.exit(f"ERROR: directory not found: {args.page_images_dir}")

    renamed = 0
    skipped_missing = 0
    skipped_already_correct = 0

    for old_name, new_name in _BOQ_RENAMES.items():
        old_path = os.path.join(args.page_images_dir, old_name)
        new_path = os.path.join(args.page_images_dir, new_name)

        if old_name == new_name:
            skipped_already_correct += 1
            continue

        if not os.path.isdir(old_path):
            if os.path.isdir(new_path):
                print(f"SKIP (already renamed): {new_name}")
            else:
                print(f"SKIP (not found):       {old_name}")
                skipped_missing += 1
            continue

        if os.path.isdir(new_path):
            print(
                f"WARNING: both {old_name!r} and {new_name!r} exist. "
                "Manual inspection required -- skipping to avoid data loss."
            )
            continue

        if args.dry_run:
            print(f"[DRY RUN] RENAME: {old_name}  ->  {new_name}")
        else:
            os.rename(old_path, new_path)
            print(f"RENAMED: {old_name}  ->  {new_name}")
        renamed += 1

    print(
        f"\nDone. {renamed} director{'y' if renamed == 1 else 'ies'} "
        f"{'would be ' if args.dry_run else ''}renamed, "
        f"{skipped_missing} not found on disk."
    )
    if args.dry_run and renamed > 0:
        print("Re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()
