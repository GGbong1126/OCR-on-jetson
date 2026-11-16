"""Explicit, provenance-bound field policies; never rewrite OCR output."""
import unicodedata


def page_number_equal(left, right):
    """Ignore U+0020 only; preserve every other code point and its order."""
    left, right = left.replace(" ", ""), right.replace(" ", "")
    valid = bool(left) and any(c.isdecimal() for c in left) and all(
        c.isdecimal() or unicodedata.category(c)[0] in ("P", "S") for c in left)
    return valid and left == right


def validate_annotations(annotation, provenance, rows):
    if annotation.get("schema_version") != 1:
        raise ValueError("Unsupported field annotation schema")
    for key in ("image_sha256", "source_report_sha256", "source_branch", "crop_set_sha256"):
        if not provenance.get(key) or annotation.get(key) != provenance[key]:
            raise ValueError(f"Field annotation provenance mismatch: {key}")
    fields = annotation.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ValueError("Expected nonempty explicit field annotations")
    by_index = {r["index"]: r for r in rows}
    roles = {}
    for field in fields:
        index = field.get("index")
        if type(index) is not int or index not in by_index or index in roles:
            raise ValueError("Unknown or duplicate annotated index")
        row = by_index[index]
        if field.get("role") != "page_number":
            raise ValueError("Unsupported annotated field role")
        if any(field.get(k) != row[k] for k in ("box", "input_shape")):
            raise ValueError("Annotated crop geometry mismatch")
        if not page_number_equal(row["text"], row["text"]):
            raise ValueError("Annotated source is not digits/symbols with ASCII spaces")
        roles[index] = "page_number"
    return roles
