"""
Task 8 — PageIndex Vectorless RAG.

Module này upload các tài liệu Markdown (sau khi chuyển tạm sang PDF), lưu lại
``doc_id`` để không upload lặp, rồi dùng retrieval API của PageIndex làm fallback
cho hybrid search.

PageIndex retrieval là API legacy nhưng vẫn được SDK hỗ trợ. Response của các
phiên bản SDK từng có cả hai dạng ``relevant_contents: list[dict]`` và
``relevant_contents: list[list[dict]]``; parser bên dưới hỗ trợ cả hai.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Iterator

from .env_utils import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"
PDF_CACHE_DIR = PROJECT_ROOT / "data" / "_tmp_pdf"
DOC_IDS_CACHE = PROJECT_ROOT / "pageindex_doc_ids.json"

POLL_INTERVAL_SECONDS = 2.0
POLL_TIMEOUT_SECONDS = 60.0

# Import tùy chọn để các task local vẫn chạy khi người dùng chưa cấu hình
# PageIndex. Đồng thời giữ tên này ở module scope để có thể mock trong unit test.
try:  # SDK hiện tại
    from pageindex import PageIndexClient as PageIndexClient
except ImportError:  # SDK cũ được dùng trong starter code
    try:
        from pageindex.client import PageIndexClient as PageIndexClient
    except ImportError:
        PageIndexClient = None  # type: ignore[assignment,misc]


def _configured_api_key() -> str:
    """Lấy API key mới nhất, kể cả khi env được set sau lúc import module."""
    return os.getenv("PAGEINDEX_API_KEY", "").strip() or PAGEINDEX_API_KEY.strip()


def _make_client():
    """Khởi tạo PageIndex client với lỗi hướng dẫn dễ hiểu."""
    api_key = _configured_api_key()
    if not api_key:
        raise RuntimeError(
            "Chưa cấu hình PAGEINDEX_API_KEY. Hãy thêm key vào file .env."
        )
    if PageIndexClient is None:
        raise RuntimeError(
            "Chưa cài PageIndex SDK. Chạy: pip install -U pageindex"
        )
    return PageIndexClient(api_key=api_key)


def _load_doc_ids() -> dict[str, str]:
    """Đọc cache và bỏ qua entry hỏng thay vì làm hỏng toàn bộ fallback."""
    if not DOC_IDS_CACHE.exists():
        return {}

    try:
        payload = json.loads(DOC_IDS_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(f"Không đọc được cache PageIndex: {exc}", RuntimeWarning)
        return {}

    # Chấp nhận cả format phẳng {path: doc_id} và format có key "documents".
    if isinstance(payload, dict) and isinstance(payload.get("documents"), dict):
        payload = payload["documents"]
    if not isinstance(payload, dict):
        return {}

    doc_ids: dict[str, str] = {}
    for document, value in payload.items():
        doc_id = value.get("doc_id") if isinstance(value, dict) else value
        if isinstance(document, str) and isinstance(doc_id, str) and doc_id.strip():
            doc_ids[document] = doc_id.strip()
    return doc_ids


def _save_doc_ids(doc_ids: dict[str, str]) -> None:
    """Ghi cache theo kiểu atomic để tránh file JSON dở dang khi bị ngắt."""
    DOC_IDS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    temp_path = DOC_IDS_CACHE.with_name(f"{DOC_IDS_CACHE.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(
            json.dumps(doc_ids, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(DOC_IDS_CACHE)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _find_unicode_fonts() -> tuple[Path | None, Path | None]:
    """Tìm font Unicode phổ biến; trả về (regular, bold)."""
    windir = Path(os.environ.get("WINDIR", "C:/Windows"))
    regular_candidates = (
        PROJECT_ROOT / "assets" / "DejaVuSans.ttf",
        windir / "Fonts" / "arial.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    bold_candidates = (
        PROJECT_ROOT / "assets" / "DejaVuSans-Bold.ttf",
        windir / "Fonts" / "arialbd.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    )
    regular = next((path for path in regular_candidates if path.is_file()), None)
    bold = next((path for path in bold_candidates if path.is_file()), None)
    return regular, bold


def _latin1_safe(text: str) -> str:
    """Fallback có thể đọc được khi máy không có font Unicode."""
    text = text.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("latin-1", errors="ignore").decode("latin-1")


def _markdown_to_pdf(markdown_path: Path, pdf_path: Path) -> Path:
    """Chuyển Markdown sang PDF đơn giản, giữ heading để PageIndex đọc cấu trúc."""
    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise RuntimeError(
            "Chưa cài fpdf2 để chuyển Markdown sang PDF. Chạy: pip install fpdf2"
        ) from exc

    content = markdown_path.read_text(encoding="utf-8", errors="replace")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_title(markdown_path.stem)
    pdf.add_page()

    regular_font, bold_font = _find_unicode_fonts()
    font_family = "Helvetica"
    has_bold = True
    use_unicode = regular_font is not None
    if regular_font is not None:
        font_family = "PageIndexUnicode"
        pdf.add_font(font_family, style="", fname=str(regular_font))
        has_bold = bold_font is not None
        if bold_font is not None:
            pdf.add_font(font_family, style="B", fname=str(bold_font))

    for raw_line in content.splitlines():
        line = raw_line.replace("\x00", "").replace("\t", "    ").rstrip()
        if not line:
            pdf.ln(3)
            continue

        heading = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            level = len(heading.group(1))
            line = heading.group(2)
            font_size = max(11, 18 - (level - 1) * 2)
            style = "B" if has_bold else ""
            line_height = font_size * 0.48
        else:
            font_size = 10
            style = ""
            line_height = 5

        rendered_line = line if use_unicode else _latin1_safe(line)
        if not rendered_line:
            continue
        pdf.set_font(font_family, style=style, size=font_size)
        # wrapmode=CHAR tránh lỗi với URL/token dài không có khoảng trắng.
        pdf.multi_cell(
            0,
            line_height,
            rendered_line,
            new_x="LMARGIN",
            new_y="NEXT",
            wrapmode="CHAR",
        )
        if heading:
            pdf.ln(1)

    pdf.output(str(pdf_path))
    return pdf_path


def _pdf_path_for(markdown_path: Path) -> Path:
    """Tạo tên PDF không đụng nhau cho các file trùng tên ở thư mục khác."""
    relative_name = markdown_path.relative_to(STANDARDIZED_DIR).as_posix()
    digest = hashlib.sha1(relative_name.encode("utf-8")).hexdigest()[:10]
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", markdown_path.stem).strip("_")
    return PDF_CACHE_DIR / f"{safe_stem or 'document'}-{digest}.pdf"


def _extract_identifier(response: Any, *keys: str) -> str | None:
    """Lấy ID từ response dict hoặc object của các phiên bản SDK khác nhau."""
    for key in keys:
        value = response.get(key) if isinstance(response, dict) else getattr(response, key, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def upload_documents(client: Any | None = None) -> dict[str, str]:
    """
    Upload toàn bộ Markdown trong ``data/standardized`` lên PageIndex.

    Các file đã có trong ``pageindex_doc_ids.json`` sẽ được bỏ qua. Hàm trả về
    mapping ``đường_dẫn_tương_đối -> doc_id`` để có thể dùng ngay cho truy vấn.
    """
    client = client or _make_client()
    doc_ids = _load_doc_ids()
    markdown_files = sorted(STANDARDIZED_DIR.rglob("*.md"))

    if not markdown_files:
        warnings.warn(
            f"Không có tài liệu Markdown trong {STANDARDIZED_DIR}", RuntimeWarning
        )
        return doc_ids

    for markdown_path in markdown_files:
        cache_key = markdown_path.relative_to(STANDARDIZED_DIR).as_posix()
        if cache_key in doc_ids:
            print(f"  ↷ Cached: {cache_key} -> {doc_ids[cache_key]}")
            continue

        pdf_path = _pdf_path_for(markdown_path)
        if not pdf_path.exists() or pdf_path.stat().st_mtime_ns < markdown_path.stat().st_mtime_ns:
            _markdown_to_pdf(markdown_path, pdf_path)

        response = client.submit_document(str(pdf_path))
        doc_id = _extract_identifier(response, "doc_id", "id")
        if not doc_id:
            raise RuntimeError(
                f"PageIndex không trả về doc_id khi upload {cache_key}: {response!r}"
            )

        doc_ids[cache_key] = doc_id
        _save_doc_ids(doc_ids)  # Không mất các upload trước nếu lần sau bị lỗi.
        print(f"  ✓ Uploaded: {cache_key} -> {doc_id}")

    return doc_ids


def _poll_retrieval(client: Any, retrieval_id: str) -> dict[str, Any]:
    """Poll retrieval mỗi 2 giây, tối đa 60 giây."""
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    while True:
        response = client.get_retrieval(retrieval_id)
        if not isinstance(response, dict):
            raise RuntimeError(f"PageIndex trả retrieval không hợp lệ: {response!r}")

        status = str(response.get("status", "")).lower()
        if status == "completed" or (not status and "retrieved_nodes" in response):
            return response
        if status in {"failed", "error", "cancelled", "canceled"}:
            detail = response.get("error") or response.get("message") or status
            raise RuntimeError(f"PageIndex retrieval thất bại: {detail}")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"PageIndex retrieval {retrieval_id} quá {POLL_TIMEOUT_SECONDS:g} giây"
            )
        time.sleep(min(POLL_INTERVAL_SECONDS, remaining))


def _retrieved_nodes(response: dict[str, Any]) -> list[Any]:
    """Tìm ``retrieved_nodes`` kể cả khi SDK bọc response trong result/data."""
    nodes = response.get("retrieved_nodes")
    if isinstance(nodes, list):
        return nodes
    for wrapper in ("result", "data"):
        nested = response.get(wrapper)
        if isinstance(nested, dict):
            nodes = _retrieved_nodes(nested)
            if nodes:
                return nodes
    return []


def _iter_relevant_contents(
    value: Any, inherited_section: str | None = None
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Flatten cả schema list[dict] lẫn list[list[dict]] của PageIndex."""
    if isinstance(value, list):
        for item in value:
            yield from _iter_relevant_contents(item, inherited_section)
        return
    if not isinstance(value, dict):
        return

    section = (
        value.get("section_title")
        or value.get("title")
        or inherited_section
    )
    content = value.get("relevant_content")
    if isinstance(content, str) and content.strip():
        metadata = {
            "section": section,
            "node_id": value.get("node_id"),
            "page_index": value.get("page_index"),
        }
        yield content.strip(), {key: val for key, val in metadata.items() if val is not None}

    nested = value.get("relevant_contents")
    if nested is not None:
        yield from _iter_relevant_contents(nested, section)


def _parse_results(
    retrieval: dict[str, Any], document: str, doc_id: str
) -> list[dict]:
    """Chuẩn hóa response PageIndex về contract chung của retrieval pipeline."""
    parsed: list[dict] = []
    rank = 0
    for node in _retrieved_nodes(retrieval):
        for content, metadata in _iter_relevant_contents(node):
            rank += 1
            metadata.update({"document": document, "doc_id": doc_id})
            parsed.append(
                {
                    "content": content,
                    # Retrieval legacy không có relevance score; dùng rank score.
                    "score": round(1.0 / rank, 6),
                    "metadata": metadata,
                    "source": "pageindex",
                }
            )
    return parsed


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.

    Nếu PageIndex chưa được cấu hình, hàm trả ``[]`` để Task 9 tiếp tục dùng kết
    quả hybrid. Mỗi kết quả có ``content``, rank-based ``score``, ``metadata`` và
    ``source='pageindex'``.
    """
    query = query.strip() if isinstance(query, str) else ""
    if not query or top_k <= 0:
        return []
    if not _configured_api_key():
        warnings.warn(
            "Skip PageIndex fallback because PAGEINDEX_API_KEY is not configured.",
            RuntimeWarning,
        )
        return []
    if PageIndexClient is None:
        warnings.warn(
            "Skip PageIndex fallback because PageIndex SDK is not installed.",
            RuntimeWarning,
        )
        return []

    client = _make_client()
    doc_ids = _load_doc_ids()
    if not doc_ids:
        doc_ids = upload_documents(client)
    if not doc_ids:
        return []

    all_results: list[dict] = []
    for document, doc_id in doc_ids.items():
        try:
            response = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = _extract_identifier(response, "retrieval_id", "id")
            if not retrieval_id:
                raise RuntimeError(
                    f"PageIndex không trả về retrieval_id cho {document}: {response!r}"
                )
            retrieval = _poll_retrieval(client, retrieval_id)
            all_results.extend(_parse_results(retrieval, document, doc_id))
        except Exception as exc:
            # Một tài liệu lỗi không được ngăn truy vấn các tài liệu còn lại.
            warnings.warn(f"PageIndex query lỗi với {document}: {exc}", RuntimeWarning)

    # Điểm 1/rank được tính trong từng document, giúp nhiều document có cơ hội
    # xuất hiện. Loại content trùng và giữ bản có rank tốt nhất.
    best_by_content: dict[str, dict] = {}
    for result in all_results:
        previous = best_by_content.get(result["content"])
        if previous is None or result["score"] > previous["score"]:
            best_by_content[result["content"]] = result
    ranked = sorted(best_by_content.values(), key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


if __name__ == "__main__":
    if not _configured_api_key():
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("tuition fee payment methods", top_k=3)
        if not results:
            print("  Không có kết quả PageIndex.")
        for result in results:
            print(f"[{result['score']:.3f}] {result['content'][:100]}...")
