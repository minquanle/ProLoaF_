from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


REPO = Path(r"C:\Nam 4\ProLoaF_")
SOURCE = Path(r"C:\Nam 4\ProLoaF-paper-reproduction")
OUT = REPO / "artifacts" / "paper_reproduction" / "report" / "Bao_cao_hoan_chinh_ProLoaF_cap_nhat_calibration.docx"
ASSET_DIR = Path(tempfile.gettempdir()) / "proloaf_detailed_report_assets"

TRAIN_LOG = REPO / "artifacts" / "paper_reproduction" / "training.log"
LSTM_METRICS = REPO / "artifacts" / "paper_reproduction" / "evaluation" / "opsd_recurrent.csv"
BASELINE_METRICS = REPO / "artifacts" / "paper_reproduction" / "baseline_comparison" / "two_baseline_comparison.csv"
RUN_METADATA = REPO / "artifacts" / "paper_reproduction" / "baseline_comparison" / "run_metadata.json"
PROPHET_DIR = REPO / "artifacts" / "paper_reproduction" / "prophet_paper_protocol"
PROPHET_METRICS = PROPHET_DIR / "prophet_protocol_metrics.csv"
PROPHET_COMPARISON = PROPHET_DIR / "lstm_vs_prophet_indicative_MW.csv"
PROPHET_FIG = PROPHET_DIR / "prophet_first_aligned_24h.png"
CALIBRATION_DIR = REPO / "artifacts" / "paper_reproduction" / "interval_calibration"
CALIBRATION_TEST = CALIBRATION_DIR / "test_calibrated_summary.csv"
CALIBRATION_META = CALIBRATION_DIR / "selected_calibration.json"
CALIBRATION_FIG = CALIBRATION_DIR / "validation_calibration_tradeoff.png"
CONFIG = SOURCE / "targets" / "opsd" / "config.json"

METRIC_FIG = SOURCE / "oracles" / "eval_opsd_recurrent" / "metrics-evaluation_metrics_plot.png"
FORECAST_FIG_0 = SOURCE / "oracles" / "eval_opsd_recurrent" / "eval_hour0.png"
FORECAST_FIG_24 = SOURCE / "oracles" / "eval_opsd_recurrent" / "eval_hour24.png"
BOXPLOT_FIG = SOURCE / "oracles" / "eval_opsd_recurrent" / "boxplot.png"

NAVY = "1F4E78"
PALE_BLUE = "EAF2F8"
PALE_GRAY = "F4F6F7"
BORDER = "D9D9D9"
TEXT_GRAY = "555555"


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=110, bottom=90, end=110):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_borders(cell, color=BORDER, size="6"):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = f"w:{edge}"
        elem = borders.find(qn(tag))
        if elem is None:
            elem = OxmlElement(tag)
            borders.append(elem)
        elem.set(qn("w:val"), "single")
        elem.set(qn("w:sz"), size)
        elem.set(qn("w:color"), color)


def format_run(run, size=10.5, bold=False, color="000000", italic=False, font="Aptos"):
    run.font.name = font
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), font)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), font)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), font)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)


def set_cell_text(cell, value, *, bold=False, color="000000", size=9.2, align=None):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.05
    if align is not None:
        p.alignment = align
    run = p.add_run(str(value))
    format_run(run, size=size, bold=bold, color=color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    set_cell_margins(cell)
    set_cell_borders(cell)


def add_table(doc, headers, rows, widths=None, numeric_cols=None, font_size=9.2):
    numeric_cols = set(numeric_cols or [])
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_repeat_table_header(table.rows[0])
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        shade(cell, NAVY)
        set_cell_text(cell, header, bold=True, color="FFFFFF", size=font_size)
        if widths:
            cell.width = Inches(widths[i])
    for row_index, row in enumerate(rows):
        cells = table.add_row().cells
        for col_index, value in enumerate(row):
            if row_index % 2 == 0:
                shade(cells[col_index], PALE_BLUE)
            align = WD_ALIGN_PARAGRAPH.CENTER if col_index in numeric_cols else WD_ALIGN_PARAGRAPH.LEFT
            set_cell_text(cells[col_index], value, size=font_size, align=align)
            if widths:
                cells[col_index].width = Inches(widths[col_index])
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)
    return table


def body(doc, text, *, bold_lead=None, keep_with_next=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.13
    p.paragraph_format.keep_with_next = keep_with_next
    if bold_lead and text.startswith(bold_lead):
        a = p.add_run(bold_lead)
        format_run(a, bold=True)
        b = p.add_run(text[len(bold_lead):])
        format_run(b)
    else:
        r = p.add_run(text)
        format_run(r)
    return p


def bullet(doc, text, level=0):
    p = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.08
    format_run(p.add_run(text))
    return p


def numbered(doc, text):
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.08
    format_run(p.add_run(text))
    return p


def heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(5)
    return p


def caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)
    format_run(p.add_run(text), size=9, italic=True, color=TEXT_GRAY)


def add_figure(doc, path, caption_text, width=6.55):
    if not Path(path).exists():
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(width))
    caption(doc, caption_text)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, text, end])
    format_run(run, size=9, color=TEXT_GRAY)


def add_toc(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = 'TOC \\o "1-3" \\h \\z \\u'
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = "Mục lục sẽ được cập nhật khi mở tài liệu trong Word"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, placeholder, end])


def setup_styles(doc):
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Aptos"
    normal._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Aptos")
    normal._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Aptos")
    normal._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Aptos")
    normal.font.size = Pt(10.5)
    for style_name, size in (("Title", 23), ("Subtitle", 13), ("Heading 1", 15), ("Heading 2", 12), ("Heading 3", 10.8)):
        style = styles[style_name]
        style.font.name = "Aptos Display" if style_name in {"Title", "Subtitle", "Heading 1"} else "Aptos"
        style._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), style.font.name)
        style._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), style.font.name)
        style._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), style.font.name)
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.bold = style_name != "Subtitle"
        if style_name == "Title":
            p_pr = style._element.get_or_add_pPr()
            p_bdr = p_pr.find(qn("w:pBdr"))
            if p_bdr is not None:
                p_pr.remove(p_bdr)


def parse_training_log():
    epochs, train_loss, val_loss = [], [], []
    pattern = re.compile(r"Epoch (\d+)/50\s+train_loss\s+([+-]?[0-9.]+e[+-][0-9]+)\s+val_loss\s+([+-]?[0-9.]+e[+-][0-9]+)")
    for line in TRAIN_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            epochs.append(int(match.group(1)))
            train_loss.append(float(match.group(2)))
            val_loss.append(float(match.group(3)))
    return epochs, train_loss, val_loss


def create_charts(comparison):
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    epochs, train_loss, val_loss = parse_training_log()
    training_curve = ASSET_DIR / "training_curve.png"
    draw_line_chart(training_curve, epochs, train_loss, val_loss)

    comparison_chart = ASSET_DIR / "baseline_comparison.png"
    draw_bar_comparison(comparison_chart, comparison)
    return training_curve, comparison_chart


def chart_fonts():
    font_dir = Path(r"C:\Windows\Fonts")
    regular_path = font_dir / "arial.ttf"
    bold_path = font_dir / "arialbd.ttf"
    regular = ImageFont.truetype(str(regular_path), 25) if regular_path.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(regular_path), 21) if regular_path.exists() else ImageFont.load_default()
    bold = ImageFont.truetype(str(bold_path), 28) if bold_path.exists() else regular
    return regular, small, bold


def draw_line_chart(path, epochs, train_loss, val_loss):
    width, height = 1500, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    regular, small, bold = chart_fonts()
    left, top, right, bottom = 120, 85, width - 55, height - 100
    y_min = min(train_loss + val_loss) - 0.08
    y_max = max(train_loss + val_loss) + 0.08

    def xy(epoch, value):
        x = left + (epoch - min(epochs)) / (max(epochs) - min(epochs)) * (right - left)
        y = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
        return int(x), int(y)

    for i in range(6):
        y = top + i * (bottom - top) / 5
        value = y_max - i * (y_max - y_min) / 5
        draw.line((left, y, right, y), fill="#D9E2F3", width=2)
        draw.text((15, y - 13), f"{value:.2f}", fill="#444444", font=small)
    draw.line((left, top, left, bottom), fill="#555555", width=3)
    draw.line((left, bottom, right, bottom), fill="#555555", width=3)
    for e in (1, 10, 20, 30, 41):
        x, _ = xy(e, y_min)
        draw.text((x - 14, bottom + 18), str(e), fill="#444444", font=small)
    for values, color in ((train_loss, "#5B9BD5"), (val_loss, "#ED7D31")):
        points = [xy(e, v) for e, v in zip(epochs, values)]
        draw.line(points, fill=color, width=5, joint="curve")
    best_x, _ = xy(34, y_min)
    for y in range(top, bottom, 18):
        draw.line((best_x, y, best_x, min(y + 9, bottom)), fill="#C0392B", width=3)
    draw.text((width // 2 - 185, 22), "Training and validation loss", fill="#000000", font=bold)
    draw.text((left, height - 52), "Train NLL", fill="#5B9BD5", font=regular)
    draw.text((left + 190, height - 52), "Validation NLL", fill="#ED7D31", font=regular)
    draw.text((left + 450, height - 52), "Best epoch 34", fill="#C0392B", font=regular)
    image.save(path, dpi=(180, 180))


def draw_bar_comparison(path, comparison):
    width, height = 1550, 730
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    regular, small, bold = chart_fonts()
    labels = ["ProLoaF LSTM", "Seasonal Naive 24h", "Simple Naive"]
    keys = ["ProLoaF-LSTM", "seasonal-naive-24h", "simple-naive"]
    colors = ["#1F4E78", "#5B9BD5", "#A5A5A5"]
    metrics = [("Rmse_MW", "RMSE quy đổi MW"), ("Mae_MW", "MAE quy đổi MW")]
    panels = [(70, 80, 750, 650), (820, 80, 1500, 650)]
    values = comparison.set_index("model")
    for (metric, title), (left, top, right, bottom) in zip(metrics, panels):
        vals = [float(values.loc[key, metric]) for key in keys]
        max_val = max(vals) * 1.18
        draw.text((left + 190, 20), title, fill="#000000", font=bold)
        draw.line((left + 95, top, left + 95, bottom), fill="#555555", width=3)
        draw.line((left + 95, bottom, right, bottom), fill="#555555", width=3)
        for i in range(6):
            y = bottom - i * (bottom - top) / 5
            grid_val = i * max_val / 5
            draw.line((left + 95, y, right, y), fill="#E6E6E6", width=2)
            draw.text((left, y - 12), f"{grid_val:,.0f}", fill="#444444", font=small)
        available = right - (left + 115)
        bar_width = 130
        gap = (available - 3 * bar_width) / 4
        for i, (label, val, color) in enumerate(zip(labels, vals, colors)):
            x1 = left + 115 + gap * (i + 1) + bar_width * i
            x2 = x1 + bar_width
            y1 = bottom - val / max_val * (bottom - top)
            draw.rectangle((x1, y1, x2, bottom), fill=color)
            draw.text((x1 + 8, y1 - 32), f"{val:,.0f}", fill="#222222", font=small)
            short = ["LSTM", "Seasonal", "Simple"][i]
            draw.text((x1 + 18, bottom + 15), short, fill="#333333", font=small)
    image.save(path, dpi=(180, 180))


def add_cover(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(75)
    p.paragraph_format.space_after = Pt(18)
    r = p.add_run("BÁO CÁO TÁI LẬP THỰC NGHIỆM PROLOAF")
    format_run(r, size=23, bold=True, font="Aptos Display")
    p.style = doc.styles["Title"]

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(40)
    format_run(p.add_run("Dự báo xác suất phụ tải điện Đức trước 24 giờ"), size=14, italic=True, color=TEXT_GRAY)

    add_table(doc, ["Nội dung", "Thông tin"], [
        ["Loại báo cáo", "Báo cáo thực nghiệm phục vụ bài giữa kỳ"],
        ["Paper", "ProLoaF Probabilistic load forecasting for power systems"],
        ["Dữ liệu", "Open Power System Data OPSD time series 60 minute"],
        ["Phương pháp chính", "Encoder Decoder LSTM dự báo xác suất"],
        ["Baseline", "Simple Naive, Seasonal Naive 24 giờ và Prophet"],
        ["Ngày cập nhật", "30 tháng 9 năm 2026"],
    ], [1.8, 5.2])
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(28)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_run(p.add_run("Repository thực nghiệm  ProLoaF_"), size=10.5, color=TEXT_GRAY)
    doc.add_page_break()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    metadata = json.loads(RUN_METADATA.read_text(encoding="utf-8"))
    lstm = pd.read_csv(LSTM_METRICS, sep=";", index_col=0).iloc[0]
    comparison = pd.read_csv(BASELINE_METRICS)
    prophet_metrics = pd.read_csv(PROPHET_METRICS)
    prophet_comparison = pd.read_csv(PROPHET_COMPARISON)
    prophet_literal = prophet_metrics.loc[prophet_metrics["protocol"] == "paper_literal_80pct"].iloc[0]
    prophet_aligned = prophet_metrics.loc[prophet_metrics["protocol"] == "chronological_aligned_95pct"].iloc[0]
    calibration_test = pd.read_csv(CALIBRATION_TEST)
    calibration_meta = json.loads(CALIBRATION_META.read_text(encoding="utf-8"))
    lstm_original_interval = calibration_test.loc[calibration_test["interval"] == "original_paper_checkpoint"].iloc[0]
    lstm_calibrated_interval = calibration_test.loc[calibration_test["interval"] == "validation_calibrated_frozen_c"].iloc[0]
    training_curve, comparison_chart = create_charts(comparison)

    doc = Document()
    setup_styles(doc)
    section = doc.sections[0]
    section.top_margin = Inches(0.68)
    section.bottom_margin = Inches(0.68)
    section.left_margin = Inches(0.72)
    section.right_margin = Inches(0.72)
    add_page_number(section.footer.paragraphs[0])
    add_cover(doc)

    heading(doc, "Mục lục", 1)
    add_toc(doc)
    doc.add_page_break()

    heading(doc, "Tóm tắt kết quả", 1)
    body(doc, "Báo cáo này ghi lại toàn bộ quá trình tái lập case study dự báo phụ tải điện Đức của paper ProLoaF. Chúng ta đã dùng mã nguồn chính thức của tác giả, chuẩn bị đúng dữ liệu OPSD, huấn luyện lại mô hình Encoder Decoder LSTM, đánh giá checkpoint trên tập test và xây dựng hai baseline đơn giản. Phần Prophet trong notebook gốc cũng đã được chạy và kiểm tra lại theo timestamp.")
    body(doc, "Kết quả chính là pipeline chạy hoàn chỉnh trên máy cá nhân. Mô hình LSTM đạt RMSE 0,070556 và MAE 0,045787 trên thang MinMax. Quy đổi theo miền giá trị train, hai sai số này xấp xỉ 3 263 MW và 2 117 MW. Bản gốc có khoảng dự báo 95 phần trăm bao phủ 97,10 phần trăm dữ liệu thực tế. Cải tiến calibration hậu xử lý đưa coverage test về 95,06 phần trăm và thu hẹp khoảng 15,8 phần trăm, không làm thay đổi dự báo điểm. So với Seasonal Naive 24 giờ, LSTM giảm RMSE 52,93 phần trăm và giảm MAE 53,38 phần trăm.")
    body(doc, "Giới hạn quan trọng là paper không công bố một bảng metric tổng hợp tương ứng cho checkpoint LSTM, đồng thời repository chính thức không kèm checkpoint đã huấn luyện. Vì vậy kết quả hiện tại chứng minh rằng chúng ta tái lập được quy trình và thu được kết quả hợp lý; chưa thể tuyên bố các con số trùng khớp tuyệt đối với tác giả.")
    add_table(doc, ["Hạng mục", "Trạng thái", "Kết quả hiện tại"], [
        ["Nguồn paper", "Hoàn thành", "Đã cố định source chính thức tại commit 1055d1fd"],
        ["Khám phá dữ liệu", "Hoàn thành", "43 824 dòng theo giờ, target không thiếu"],
        ["Huấn luyện", "Hoàn thành", "Dừng ở epoch 41, khôi phục best epoch 34"],
        ["Đánh giá LSTM", "Hoàn thành", "8 595 forecast origins, horizon 24 giờ"],
        ["Hai baseline", "Hoàn thành", "Simple Naive và Seasonal Naive 24 giờ"],
        ["Prophet", "Hoàn thành", "Fit một lần, forecast liên tục và căn timestamp"],
        ["Calibration khoảng LSTM", "Hoàn thành", "Validation chọn c, test đạt coverage 95,06 phần trăm"],
        ["Demo và slide", "Chưa hoàn thành", "Đây là ưu tiên tiếp theo"],
    ], [2.0, 1.25, 3.75])

    heading(doc, "1 Yêu cầu bài giữa kỳ và phạm vi thực hiện", 1)
    heading(doc, "1 1 Yêu cầu của bài nộp", 2)
    body(doc, "Đề bài yêu cầu chọn một paper khoa học đã được giảng viên xác nhận, viết báo cáo bằng tiếng Anh hoặc tiếng Việt, đính kèm mã nguồn GitHub, trình bày trong tối đa 15 slide và trả lời câu hỏi. Nội dung bắt buộc gồm năm phần sau.")
    numbered(doc, "Giới thiệu bài toán, vai trò và ứng dụng thực tế.")
    numbered(doc, "Khám phá bộ dữ liệu.")
    numbered(doc, "Trình bày chi tiết giải pháp.")
    numbered(doc, "Xây dựng demo.")
    numbered(doc, "So sánh với công nghệ hoặc thuật toán khác giải cùng bài toán.")
    heading(doc, "1 2 Cách hiểu yêu cầu tái lập paper", 2)
    body(doc, "Trong bài này, tái lập không có nghĩa là chạy toàn bộ mọi benchmark mà tác giả từng thử. Phạm vi hợp lý là chạy lại đúng pipeline chính mà paper đề xuất cho case study OPSD, tạo checkpoint mới, đánh giá bằng các metric phù hợp, sau đó dùng một số baseline đủ rõ để chứng minh giá trị của phương pháp.")
    body(doc, "Phần cốt lõi đã làm là LSTM xác suất của ProLoaF. Hai baseline và Prophet được thêm để đáp ứng yêu cầu so sánh nhưng không thay thế mục tiêu tái lập. SARIMA và các mô hình khác chỉ nên bổ sung sau khi demo và slide hoàn thành.")

    heading(doc, "2 Paper và bài toán dự báo", 1)
    heading(doc, "2 1 Paper ProLoaF giải quyết vấn đề gì", 2)
    body(doc, "ProLoaF là một framework mã nguồn mở cho dự báo phụ tải xác suất. Framework tổ chức toàn bộ vòng đời thực nghiệm gồm nạp dữ liệu, tiền xử lý, tạo tensor, cấu hình mô hình, huấn luyện, lưu checkpoint, chạy benchmark và tính metric. Paper dùng bài toán phụ tải điện để chứng minh framework có thể tạo cả dự báo điểm lẫn độ bất định.")
    heading(doc, "2 2 Đầu vào và đầu ra dễ hiểu", 2)
    body(doc, "Tại một thời điểm dự báo, hệ thống nhìn lại 147 giờ gần nhất. Mỗi giờ lịch sử chứa phụ tải Đức, phụ tải Áo và ba biến thời tiết của Đức. Hệ thống cũng biết trước lịch của 24 giờ tương lai như giờ trong ngày, thứ trong tuần và tháng. Đầu ra là 24 dự báo liên tiếp cho ngày kế tiếp.")
    add_table(doc, ["Thành phần", "Nội dung", "Ý nghĩa"], [
        ["Lịch sử đầu vào", "147 giờ", "Khoảng 6 ngày và 3 giờ gần nhất"],
        ["Encoder features", "5 biến", "Phụ tải Đức và Áo cùng nhiệt độ và bức xạ"],
        ["Decoder features", "46 biến lịch", "Thông tin biết trước của 24 giờ tương lai"],
        ["Forecast horizon", "24 giờ", "Một đường dự báo cho ngày tiếp theo"],
        ["Đầu ra mỗi giờ", "Mean và log variance", "Giá trị kỳ vọng và độ bất định"],
    ], [1.55, 2.0, 3.45])
    heading(doc, "2 3 Ý nghĩa thực tế", 2)
    body(doc, "Dự báo phụ tải ngày hôm sau hỗ trợ điều độ nguồn điện, lập kế hoạch mua bán điện, bố trí nguồn dự phòng và giảm rủi ro thiếu hoặc thừa công suất. Khoảng dự báo còn cho biết mức độ không chắc chắn, giúp người vận hành phân biệt giữa một dự báo ổn định và một dự báo có rủi ro cao.")

    heading(doc, "3 Nguồn mã và môi trường tái lập", 1)
    heading(doc, "3 1 Cố định mã nguồn chính thức", 2)
    body(doc, "Mã nguồn thực nghiệm được lấy từ repository chính thức ElsevierSoftwareX SOFTX D 21 00190 và cố định tại commit 1055d1fd3ae00ffce0b67eed03ec84a7410f5da8. Việc cố định commit giúp bảo đảm rằng cấu hình, xử lý dữ liệu và kiến trúc không thay đổi trong quá trình làm bài.")
    heading(doc, "3 2 Môi trường chạy", 2)
    add_table(doc, ["Thành phần", "Phiên bản hoặc trạng thái"], [
        ["Conda environment", "proloaf paper repro"],
        ["Python", "3.9.25"],
        ["PyTorch", "1.13.1 CPU"],
        ["Prophet", "1.1.5 cùng cmdstanpy 1.2.5"],
        ["Thiết bị", "CPU, không dùng CUDA"],
        ["Hệ điều hành", "Windows"],
        ["Thời gian train", "1 giờ 11 phút 47 giây"],
    ], [2.4, 4.6])
    heading(doc, "3 3 Những gì đã chuẩn bị", 2)
    bullet(doc, "Tạo môi trường conda riêng để tránh xung đột với Python 3.14 của hệ thống.")
    bullet(doc, "Cài đúng nhóm thư viện mà source paper cần, gồm PyTorch, pandas, scikit learn, statsmodels và các thư viện trực quan hóa.")
    bullet(doc, "Kiểm tra interpreter thực tế trỏ tới môi trường proloaf paper repro.")
    bullet(doc, "Tách repository dùng để nộp bài khỏi bản source paper dùng để chạy thực nghiệm, đồng thời lưu artifact cần thiết vào repository nộp bài.")
    heading(doc, "3 4 Kết quả của bước setup", 2)
    body(doc, "Source có thể nạp dữ liệu, train và evaluate trên Windows. Lỗi môi trường Python ban đầu đã được xử lý bằng cách kích hoạt conda hook và chọn đúng interpreter. CUDA bằng False không phải lỗi vì quá trình hiện tại chủ động chạy trên CPU.")

    heading(doc, "4 Khám phá bộ dữ liệu OPSD", 1)
    heading(doc, "4 1 File dữ liệu được dùng", 2)
    body(doc, "File time series 60 minute singleindex là dữ liệu điện năng theo tần suất một giờ do Open Power System Data tổng hợp. Source paper không đưa toàn bộ file raw khổng lồ trực tiếp vào model; repository sử dụng file opsd csv đã được chọn cột và chuẩn bị cho case study.")
    heading(doc, "4 2 Kết quả kiểm tra dữ liệu", 2)
    add_table(doc, ["Thuộc tính", "Kết quả"], [
        ["Khoảng thời gian", "01 tháng 01 năm 2015 đến 31 tháng 12 năm 2019"],
        ["Tần suất", "1 giờ"],
        ["Tổng số dòng", "43 824"],
        ["Target", "DE load actual entsoe transparency"],
        ["Đơn vị target", "MW"],
        ["Giá trị thiếu của target", "Không có trong dữ liệu đã chuẩn bị"],
        ["Min target trên train", "31 307 MW"],
        ["Max target trên train", "77 549 MW"],
        ["Range target trên train", "46 242 MW"],
    ], [2.5, 4.5])
    heading(doc, "4 3 Chia dữ liệu theo thời gian", 2)
    body(doc, "Dữ liệu time series phải được chia theo thứ tự thời gian. Trộn ngẫu nhiên sẽ làm thông tin tương lai xuất hiện trong train set và khiến kết quả quá lạc quan. Cấu hình ProLoaF dùng 60 phần trăm đầu cho train, 20 phần trăm tiếp theo cho validation và 20 phần trăm cuối cho test.")
    add_table(doc, ["Tập", "Số dòng raw", "Vai trò"], [
        ["Train", "26 294", "Học scaler và cập nhật trọng số LSTM"],
        ["Validation", "8 765", "Theo dõi khả năng tổng quát và early stopping"],
        ["Test", "8 765", "Đánh giá cuối cùng, không dùng để học"],
    ], [1.3, 1.5, 4.2], numeric_cols=[1])
    body(doc, "Tập test raw bắt đầu lúc 19 giờ ngày 31 tháng 12 năm 2018 và kết thúc lúc 23 giờ ngày 31 tháng 12 năm 2019. Sau khi dành 147 giờ làm lịch sử và 24 giờ làm nhãn tương lai, code tạo được 8 595 forecast origins. Mỗi origin sinh 24 dự báo nên tổng số điểm đánh giá là 206 280.")
    heading(doc, "4 4 Kết luận sau khi khám phá dữ liệu", 2)
    body(doc, "Dữ liệu đã đủ sạch để đưa vào pipeline. Chu kỳ ngày đêm thể hiện rõ nên Seasonal Naive 24 giờ là baseline hợp lý. Đồng thời, nhiệt độ và đặc trưng lịch có thể giúp LSTM xử lý những ngày không lặp lại hoàn toàn theo ngày trước.")

    heading(doc, "5 Tiền xử lý và tạo mẫu", 1)
    heading(doc, "5 1 Chuẩn hóa dữ liệu", 2)
    body(doc, "Nhóm feature chính được chuẩn hóa MinMax về khoảng từ 0 đến 1. Scaler chỉ được fit trên train set rồi áp dụng cho validation và test. Quy tắc này ngăn việc dùng min hoặc max của tương lai trong quá trình huấn luyện.")
    heading(doc, "5 2 Tạo đặc trưng lịch", 2)
    body(doc, "Decoder nhận 46 đặc trưng biết trước, gồm 24 biến one hot cho giờ, 12 biến cho tháng, 7 biến cho thứ trong tuần và ba biến sin biểu diễn giờ, thứ và tháng. One hot giúp mô hình phân biệt từng trạng thái lịch; biến sin giữ được tính tuần hoàn, ví dụ 23 giờ gần 0 giờ hơn 12 giờ.")
    heading(doc, "5 3 Tạo cửa sổ dự báo", 2)
    add_table(doc, ["Bước", "Đầu vào", "Xử lý", "Đầu ra"], [
        ["Lấy lịch sử", "147 giờ trước origin", "Chọn 5 encoder features", "Tensor encoder 147 x 5"],
        ["Lấy lịch tương lai", "24 giờ sau origin", "Chọn 46 decoder features", "Tensor decoder 24 x 46"],
        ["Tạo nhãn", "Target 24 giờ sau origin", "Chuẩn hóa bằng scaler train", "Tensor target 24 x 1"],
        ["Trượt cửa sổ", "Chuỗi thời gian liên tục", "Dịch origin từng giờ", "Nhiều mẫu train validation test"],
    ], [1.15, 1.55, 2.15, 2.15], font_size=8.7)
    heading(doc, "5 4 Kết quả của bước tiền xử lý", 2)
    body(doc, "Code tạo được tensor đúng hình dạng mà encoder và decoder yêu cầu. Log không báo missing data. Dữ liệu validation và test giữ nguyên thứ tự thời gian, vì vậy metric sau này phản ánh bài toán dự báo thực tế hơn so với chia ngẫu nhiên.")

    doc.add_page_break()
    heading(doc, "6 Kiến trúc và cách học của mô hình", 1)
    heading(doc, "6 1 Kiến trúc Encoder Decoder LSTM", 2)
    add_table(doc, ["Khối", "Cấu hình", "Chức năng"], [
        ["Encoder LSTM", "Input 5, hidden 51, 1 layer", "Tóm tắt 147 giờ lịch sử thành trạng thái ẩn"],
        ["Decoder LSTM", "Input 46, hidden 51, 1 layer", "Kết hợp trạng thái lịch sử với lịch của 24 giờ tương lai"],
        ["Fully connected 1", "51 đến 51", "Biến đổi representation của decoder"],
        ["Dropout", "Core 0,3 và FC 0,4", "Giảm overfitting trong khi train"],
        ["Fully connected 2", "51 đến 2", "Sinh expected value và log variance"],
    ], [1.55, 2.1, 3.35])
    heading(doc, "6 2 Vì sao đầu ra có hai giá trị", 2)
    body(doc, "Dự báo điểm chỉ cần một giá trị kỳ vọng. Dự báo xác suất còn cần mức độ phân tán. Mô hình xuất expected value và log variance, sau đó ProLoaF xây dựng phân phối Gaussian cho từng giờ. Từ phân phối này, framework lấy median và các quantile tạo khoảng dự báo 95 phần trăm.")
    heading(doc, "6 3 Hàm loss", 2)
    body(doc, "Mô hình dùng Gaussian negative log likelihood NllGauss. Loss phạt cả dự báo trung tâm sai và ước lượng độ bất định không phù hợp. Nếu mô hình tạo khoảng quá hẹp, các quan sát ngoài khoảng làm loss tăng. Nếu mô hình luôn tạo khoảng rất rộng, phần variance trong loss cũng bị phạt.")
    heading(doc, "6 4 Thông số huấn luyện", 2)
    add_table(doc, ["Thông số", "Giá trị"], [
        ["Optimizer", config["optimizer_name"]],
        ["Learning rate", f'{config["learning_rate"]:.10f}'],
        ["Batch size", config["batch_size"]],
        ["Max epochs", config["max_epochs"]],
        ["Early stopping patience", config["early_stopping_patience"]],
        ["Early stopping margin", config["early_stopping_margin"]],
        ["Model class", config["model_class"]],
    ], [2.6, 4.4])

    heading(doc, "7 Quá trình huấn luyện lại", 1)
    heading(doc, "7 1 Lệnh đã chạy", 2)
    body(doc, "Từ thư mục source paper reproduction, quá trình được bắt đầu bằng lệnh python src train py s opsd. Script đọc config của target opsd, nạp dữ liệu, tạo DataLoader, khởi tạo LSTM, tối ưu bằng Adam, theo dõi validation loss và lưu checkpoint tốt nhất.")
    heading(doc, "7 2 Diễn biến huấn luyện", 2)
    body(doc, "Train bắt đầu lúc 14 giờ 44 phút 09 giây và kết thúc lúc 15 giờ 55 phút 56 giây ngày 24 tháng 9 năm 2026. Tổng thời gian khoảng 1 giờ 11 phút 47 giây trên CPU. Train loss giảm từ khoảng âm 0,784 ở epoch đầu xuống âm 2,22 ở epoch 41. Validation loss tốt nhất là khoảng âm 2,35 ở epoch 34.")
    add_figure(doc, training_curve, "Hình 1 Đường loss train và validation trong 41 epoch", width=6.65)
    heading(doc, "7 3 Vì sao chỉ chạy đến epoch 41", 2)
    body(doc, "Cấu hình cho phép tối đa 50 epoch nhưng early stopping theo dõi validation loss. Từ sau epoch 34, validation loss không tạo mức tốt hơn trong 7 epoch liên tiếp. Code vì vậy dừng ở epoch 41 và nạp lại trọng số của epoch 34. Đây là hành vi đúng của pipeline, không phải quá trình train bị lỗi hoặc bị dừng thủ công.")
    heading(doc, "7 4 Đầu ra của bước train", 2)
    add_table(doc, ["Đầu ra", "Ý nghĩa"], [
        ["opsd recurrent pkl", "Checkpoint chứa cấu hình và trọng số tốt nhất"],
        ["training log", "Log từng epoch, loss, thời gian và trạng thái early stopping"],
        ["TensorBoard event", "Dữ liệu dùng để xem diễn biến huấn luyện"],
        ["Best validation NLL", "Khoảng âm 2,35 tại epoch 34"],
    ], [2.25, 4.75])

    heading(doc, "8 Đánh giá mô hình LSTM", 1)
    heading(doc, "8 1 Cách đánh giá", 2)
    body(doc, "Checkpoint tốt nhất được chạy trên 8 595 forecast origins của test set bằng lệnh python src evaluate py s opsd. Mỗi origin dự báo 24 giờ. Script gom toàn bộ dự báo để tính metric tổng hợp, đồng thời tính metric theo từng horizon từ giờ 1 đến giờ 24.")
    heading(doc, "8 2 Ý nghĩa các metric", 2)
    add_table(doc, ["Metric", "Ý nghĩa", "Cách đọc"], [
        ["RMSE", "Căn sai số bình phương trung bình", "Nhỏ hơn tốt hơn, nhạy với sai số lớn"],
        ["MAE", "Sai số tuyệt đối trung bình", "Nhỏ hơn tốt hơn, dễ diễn giải"],
        ["PICP", "Tỷ lệ actual nằm trong khoảng 95 phần trăm", "Nên gần 95 phần trăm"],
        ["Sharpness", "Độ rộng trung bình của khoảng dự báo", "Nhỏ hơn tốt hơn nếu coverage vẫn đạt"],
        ["MIS", "Kết hợp độ rộng và phạt khi actual nằm ngoài khoảng", "Nhỏ hơn tốt hơn"],
        ["NllGauss", "Mức phù hợp của phân phối Gaussian", "Nhỏ hơn tốt hơn"],
        ["Residual mean", "Actual trừ prediction trung bình", "Gần 0 nghĩa là ít thiên lệch tổng thể"],
    ], [1.3, 3.0, 2.7], font_size=8.8)
    heading(doc, "8 3 Kết quả tổng hợp", 2)
    add_table(doc, ["Metric", "Giá trị scaled", "Quy đổi hoặc nhận xét"], [
        ["NllGauss", f'{lstm["NllGauss"]:.6f}', "Metric xác suất chính của quá trình train"],
        ["RMSE", f'{lstm["Rmse"]:.6f}', "Xấp xỉ 3 263 MW"],
        ["MAE", f'{lstm["Mae"]:.6f}', "Xấp xỉ 2 117 MW"],
        ["PICP", f'{lstm["Picp"]:.2f} phần trăm', "Cao hơn mục tiêu danh nghĩa 95 phần trăm"],
        ["Sharpness", f'{lstm["Sharpness"]:.6f}', "Độ rộng xấp xỉ 12 416 MW"],
        ["MIS", f'{lstm["Mis"]:.6f}', "Thấp hơn nhiều so với hai baseline"],
        ["Residual mean", f'{lstm["Residuals"]:.6f}', "Xấp xỉ âm 27 MW, gần như không lệch tổng thể"],
    ], [1.65, 1.8, 3.55])
    body(doc, "PICP 97,10 phần trăm nghĩa là khoảng dự báo đang hơi bảo thủ: coverage cao hơn mục tiêu 95 phần trăm nhưng khoảng có thể rộng hơn cần thiết. Đây không phải lỗi nghiêm trọng; nó cho thấy hướng cải thiện sau này là calibration để đưa coverage gần 95 phần trăm trong khi giảm sharpness.")
    add_figure(doc, METRIC_FIG, "Hình 2 RMSE coverage sharpness và MIS theo 24 horizon", width=6.55)
    heading(doc, "8 4 Cải tiến calibration khoảng dự báo", 2)
    body(doc, "Cải tiến này không train lại LSTM và không đổi dự báo trung tâm mu. LSTM gốc vẫn dự báo đồng thời mu và độ bất định sigma cho mỗi giờ. Ta chỉ dùng validation từ 60 đến 80 phần trăm chuỗi để tìm hệ số c, sau đó thay khoảng gốc mu cộng trừ 1,95996 sigma bằng mu cộng trừ 1,95996 c sigma. Hệ số được đóng băng trước khi đánh giá test nên test không được dùng để chọn c.")
    body(doc, "Hệ số chọn được là c bằng {:.6f}, nhỏ hơn 1. Điều đó có nghĩa khoảng gốc của LSTM hơi rộng; calibration thu hẹp khoảng nhưng vẫn giữ nguyên giá trị tải điện dự báo. Vì mu không đổi, RMSE và MAE phải giữ nguyên về mặt toán học. PICP, sharpness và MIS mới là các metric thay đổi.".format(calibration_meta["selected_c"]))
    add_table(doc, ["Phiên bản LSTM trên test", "PICP", "Sharpness", "RMSE", "MAE", "MIS 95 phần trăm"], [
        ["Gốc", f'{lstm_original_interval["PICP_percent"]:.2f} phần trăm', f'{lstm_original_interval["Sharpness"]:.6f}', f'{lstm_original_interval["RMSE"]:.6f}', f'{lstm_original_interval["MAE"]:.6f}', f'{lstm_original_interval["MIS_standard_95"]:.6f}'],
        ["Calibrated c cố định", f'{lstm_calibrated_interval["PICP_percent"]:.2f} phần trăm', f'{lstm_calibrated_interval["Sharpness"]:.6f}', f'{lstm_calibrated_interval["RMSE"]:.6f}', f'{lstm_calibrated_interval["MAE"]:.6f}', f'{lstm_calibrated_interval["MIS_standard_95"]:.6f}'],
    ], [1.6, 1.0, 1.0, 0.95, 0.95, 1.1], numeric_cols=[1, 2, 3, 4, 5], font_size=8.4)
    body(doc, "Kết quả test xác nhận cải tiến thành công: PICP giảm từ 97,10 xuống 95,06 phần trăm, chỉ cách mục tiêu 95 phần trăm 0,06 điểm phần trăm. Sharpness giảm từ 0,268506 xuống 0,226052, tương đương khoảng 15,8 phần trăm. Standard MIS 95 phần trăm cũng giảm nhẹ từ 0,354555 xuống 0,352670. Đây là cải tiến về chất lượng ước lượng bất định, không phải cải tiến độ chính xác điểm.")
    body(doc, "Lưu ý kỹ thuật: output Mis 0,091573 của evaluate py gốc không được dùng để kết luận cho khoảng 95 phần trăm, vì Mis.from quantiles của phiên bản source này vô tình xét quantile trung vị 0,5 khi chọn alpha. Báo cáo dùng MIS standard 95 phần trăm được tính trực tiếp từ hai biên khoảng và alpha bằng 0,05; đồng thời vẫn lưu legacy MIS để đối chiếu output gốc.")
    add_figure(doc, CALIBRATION_FIG, "Hình 3 Calibration trên validation: coverage, sharpness và standard MIS theo hệ số c", width=6.55)
    heading(doc, "8 5 Kết quả theo horizon", 2)
    body(doc, "RMSE giảm mạnh sau giờ đầu, đạt mức thấp nhất quanh horizon 4 rồi tăng dần đến giờ 24. Mẫu này hợp lý vì dự báo càng xa thường càng khó. PICP giảm nhẹ từ khoảng 97,47 phần trăm ở horizon đầu xuống khoảng 96,77 phần trăm ở horizon cuối nhưng vẫn cao hơn 95 phần trăm. Sharpness cũng thu hẹp nhanh ở vài giờ đầu rồi tăng nhẹ khi horizon dài hơn.")
    add_figure(doc, FORECAST_FIG_0, "Hình 4 Ví dụ dự báo tại một cửa sổ test", width=6.45)
    add_figure(doc, FORECAST_FIG_24, "Hình 5 Ví dụ dự báo tại một cửa sổ test khác", width=6.45)
    heading(doc, "8 6 Lưu ý về MASE", 2)
    body(doc, "Output gốc có MASE 1,109877. Tuy nhiên evaluate py hiện tại dùng tần suất mặc định 1, nên mẫu số tương ứng với persistence một giờ chứ không phải Seasonal Naive 24 giờ. Vì lý do đó, báo cáo không dùng MASE để kết luận LSTM tốt hay kém hơn Seasonal Naive.")

    heading(doc, "9 Xây dựng hai baseline", 1)
    heading(doc, "9 1 Lý do chọn baseline", 2)
    body(doc, "Simple Naive và Seasonal Naive 24 giờ dễ kiểm tra, chạy rất nhanh và thể hiện hai giả định phổ biến. Simple Naive giả định phụ tải tương lai giữ nguyên ở mức mới nhất. Seasonal Naive giả định ngày mai lặp lại hình dạng của hôm qua.")
    heading(doc, "9 2 Công thức dự báo điểm", 2)
    bullet(doc, "Simple Naive: tại mỗi origin, lặp lại quan sát cuối cùng cho cả 24 horizon.")
    bullet(doc, "Seasonal Naive 24 giờ: dự báo mỗi giờ bằng giá trị cùng giờ của 24 giờ trước.")
    heading(doc, "9 3 Tạo khoảng dự báo cho baseline", 2)
    body(doc, "Hai baseline không tự sinh variance. Để có thể so sánh PICP, sharpness và MIS, residual trên validation set được tính riêng cho từng horizon. Quantile 2,5 phần trăm và 97,5 phần trăm của residual validation được cộng vào dự báo điểm trên test. Cách này không dùng thông tin test để calibration và tạo được khoảng 95 phần trăm.")
    body(doc, "Đây là phần mở rộng do chúng ta xây dựng để so sánh công bằng. Không nên mô tả nó là chính xác cùng cách tạo khoảng baseline trong paper nếu paper không cung cấp chi tiết tương ứng.")
    heading(doc, "9 4 Kiểm tra tính công bằng", 2)
    add_table(doc, ["Điều kiện", "LSTM", "Hai baseline"], [
        ["Cùng target", "Có", "Có"],
        ["Cùng train validation test split", "Có", "Có"],
        ["Cùng 8 595 test origins", "Có", "Có"],
        ["Cùng horizon 24 giờ", "Có", "Có"],
        ["Cùng target scaling", "Có", "Có"],
        ["Dùng test để fit hoặc calibration", "Không", "Không"],
    ], [3.25, 1.85, 1.9], numeric_cols=[1, 2])
    heading(doc, "9 5 Đầu ra của bước baseline", 2)
    body(doc, "Script run two baselines py xuất bảng metric tổng hợp, metric theo từng horizon, quantile dùng để calibration, dự báo của 24 giờ đầu tiên và metadata chứa hash của data cùng config. Hai baseline hoàn thành trong khoảng 0,013 giây cho mỗi phương pháp vì không cần tối ưu trọng số.")

    heading(doc, "9 6 Chạy Prophet từ notebook của paper", 2)
    body(doc, "Notebook Train and Benchmark Models của source gốc tạo bảng ds và y, fit một Facebook Prophet duy nhất trên 80 phần trăm đầu dữ liệu, sau đó forecast liên tục theo giờ. Script run prophet paper protocol py tái tạo đúng ý tưởng này. Prophet không nhìn 147 giờ gần nhất tại mỗi origin như LSTM; nó học trend và seasonality trên toàn bộ chuỗi train, rồi tạo một chuỗi forecast dài cho giai đoạn test.")
    body(doc, "Khi kiểm tra kỹ notebook gốc, phát hiện y val 1D của TensorLoader bắt đầu sau 147 giờ history, trong khi forecast Prophet lại bắt đầu ngay tại test split. Nếu đem hai mảng đó so trực tiếp, forecast bị lệch 147 giờ so với actual. Vì vậy, artifact lưu cả kết quả literal theo notebook để minh chứng và kết quả chronological aligned để đánh giá có timestamp khớp nhau. Bản aligned không fit lại Prophet; chỉ ghép forecast với target đúng thời điểm và yêu cầu khoảng dự báo 95 phần trăm để tương thích với cách đọc PICP của LSTM.")
    add_table(doc, ["Kết quả Prophet", "RMSE", "MAE", "PICP", "Diễn giải"], [
        ["Literal notebook 80 phần trăm", f'{prophet_literal["Rmse_MW"]:,.0f} MW', f'{prophet_literal["Mae_MW"]:,.0f} MW', f'{prophet_literal["Picp"]:.2f} phần trăm', "Lệch 147 giờ, chỉ dùng kiểm tra source"],
        ["Aligned 95 phần trăm", f'{prophet_aligned["Rmse_MW"]:,.0f} MW', f'{prophet_aligned["Mae_MW"]:,.0f} MW', f'{prophet_aligned["Picp"]:.2f} phần trăm', "Kết quả dùng để theo dõi và đối chiếu"],
    ], [1.65, 1.0, 1.0, 1.05, 2.0], numeric_cols=[1, 2, 3], font_size=8.7)
    body(doc, f'Prophet fit một lần mất khoảng {prophet_aligned["fit_seconds"]:.1f} giây trên CPU. Bản aligned đạt RMSE {prophet_aligned["Rmse"]:.6f} trên thang MinMax của notebook, tương đương khoảng {prophet_aligned["Rmse_MW"]:,.0f} MW; MAE khoảng {prophet_aligned["Mae_MW"]:,.0f} MW. PICP {prophet_aligned["Picp"]:.2f} phần trăm gần mức danh nghĩa 95 phần trăm, nhưng khoảng dự báo rộng khoảng {prophet_aligned["Sharpness_MW"]:,.0f} MW.')
    add_figure(doc, PROPHET_FIG, "Hình 5 Prophet dự báo 24 giờ sau khi căn đúng timestamp", width=6.65)

    heading(doc, "10 So sánh kết quả hiện tại", 1)
    comparison_rows = []
    for model in ["ProLoaF-LSTM", "seasonal-naive-24h", "simple-naive"]:
        row = comparison.loc[comparison["model"] == model].iloc[0]
        label = {"ProLoaF-LSTM": "ProLoaF LSTM", "seasonal-naive-24h": "Seasonal Naive 24 giờ", "simple-naive": "Simple Naive"}[model]
        comparison_rows.append([
            label,
            f'{row["Rmse"]:.6f}',
            f'{row["Mae"]:.6f}',
            f'{row["Picp"]:.2f} phần trăm',
            f'{row["Sharpness"]:.6f}',
            f'{row["Mis"]:.6f}',
        ])
    add_table(doc, ["Mô hình", "RMSE", "MAE", "PICP", "Sharpness", "MIS"], comparison_rows,
              [1.65, 0.85, 0.85, 1.15, 1.2, 1.0], numeric_cols=[1, 2, 3, 4, 5], font_size=8.7)
    doc.add_page_break()
    add_table(doc, ["Mô hình", "RMSE xấp xỉ", "MAE xấp xỉ", "Độ rộng khoảng xấp xỉ"], [
        ["ProLoaF LSTM", "3 263 MW", "2 117 MW", "12 416 MW"],
        ["Seasonal Naive 24 giờ", "6 932 MW", "4 541 MW", "31 733 MW"],
        ["Simple Naive", "11 121 MW", "8 778 MW", "38 566 MW"],
    ], [2.05, 1.55, 1.55, 1.85], numeric_cols=[1, 2, 3])
    add_figure(doc, comparison_chart, "Hình 5 So sánh RMSE và MAE quy đổi sang MW", width=6.65)
    heading(doc, "10 1 LSTM so với Seasonal Naive", 2)
    body(doc, "Seasonal Naive là baseline mạnh hơn trong hai phương pháp vì tải điện có chu kỳ 24 giờ. Dù vậy, LSTM giảm RMSE từ 0,149910 xuống 0,070556, tương ứng 52,93 phần trăm. MAE giảm từ 0,098203 xuống 0,045787, tương ứng 53,38 phần trăm. Sharpness giảm 60,87 phần trăm và MIS giảm 88,14 phần trăm.")
    heading(doc, "10 2 LSTM so với Simple Naive", 2)
    body(doc, "So với Simple Naive, LSTM giảm RMSE 70,66 phần trăm và giảm MAE 75,88 phần trăm. Khoảng dự báo của LSTM hẹp hơn khoảng 67,80 phần trăm trong khi PICP vẫn cao hơn 95 phần trăm. Kết quả này cho thấy việc chỉ giữ nguyên giá trị cuối không đủ cho bài toán dự báo cả ngày.")
    heading(doc, "10 3 So sánh coverage", 2)
    body(doc, "Simple Naive có PICP 95,82 phần trăm, Seasonal Naive đạt 94,92 phần trăm và LSTM đạt 97,10 phần trăm. Chỉ nhìn PICP chưa đủ vì một khoảng rất rộng có thể dễ dàng bao phủ actual. LSTM có lợi thế rõ khi vừa duy trì coverage tốt vừa có sharpness và MIS thấp hơn nhiều.")
    heading(doc, "10 4 LSTM so với Prophet", 2)
    prophet_compare_rows = []
    for _, row in prophet_comparison.iterrows():
        label = "ProLoaF LSTM" if str(row["model"]).startswith("ProLoaF") else "Facebook Prophet aligned"
        prophet_compare_rows.append([
            label,
            f'{row["Rmse_MW"]:,.0f} MW',
            f'{row["Mae_MW"]:,.0f} MW',
            f'{row["Picp"]:.2f} phần trăm',
            f'{row["Sharpness_MW"]:,.0f} MW',
            f'{row["Mis_MW"]:,.0f} MW',
        ])
    add_table(doc, ["Mô hình", "RMSE", "MAE", "PICP 95 phần trăm", "Sharpness", "MIS"], prophet_compare_rows,
              [1.6, 0.9, 0.9, 1.15, 1.15, 1.0], numeric_cols=[1, 2, 3, 4, 5], font_size=8.5)
    body(doc, "Theo số liệu MW hiện tại, LSTM đạt RMSE khoảng 3 263 MW so với 3 946 MW của Prophet, giảm khoảng 17,3 phần trăm; MAE giảm từ khoảng 2 898 MW xuống 2 117 MW, tương đương khoảng 26,9 phần trăm. Đây là đối chiếu có ích để giải thích xu hướng, nhưng không được gọi là benchmark hoàn toàn công bằng: Prophet theo notebook fit trên 80 phần trăm dữ liệu và scaler được fit trên toàn bộ data, còn LSTM dùng split 60 20 20 và scaler chỉ fit trên train. Hai dòng có cùng giai đoạn target cuối và cùng horizon 24 giờ, đồng thời đều hiển thị interval 95 phần trăm.")

    heading(doc, "11 Đối chiếu với kết quả trong paper", 1)
    heading(doc, "11 1 Những điểm đã bám đúng paper", 2)
    bullet(doc, "Dùng case study phụ tải điện Đức và dữ liệu OPSD.")
    bullet(doc, "Dự báo trước 24 giờ và tạo khoảng dự báo 95 phần trăm.")
    bullet(doc, "Dùng pipeline cấu hình của repository chính thức.")
    bullet(doc, "Dùng Encoder Decoder LSTM và Gaussian NLL.")
    bullet(doc, "Tách dữ liệu theo thời gian và đánh giá rolling trên test set.")
    heading(doc, "11 2 Điểm không thể đối chiếu số học trực tiếp", 2)
    body(doc, "Paper trình bày kết quả minh họa và kết luận định tính rằng core model có hiệu năng tốt hơn các benchmark theo ngày như Prophet. Tuy nhiên paper không cung cấp bảng aggregate metric cho chính LSTM gồm RMSE, MAE, PICP, sharpness và MIS. Repository công khai cũng không kèm checkpoint tác giả đã huấn luyện.")
    body(doc, "Do đó, báo cáo không dùng cụm từ tái lập chính xác con số của paper. Phát biểu phù hợp là đã tái lập thành công pipeline thực nghiệm của paper trên mã nguồn chính thức và thu được kết quả có lợi thế rõ so với hai baseline được triển khai cùng protocol.")
    heading(doc, "11 3 Khác biệt giữa paper và phạm vi bài làm", 2)
    add_table(doc, ["Nội dung", "Paper và source gốc", "Bài làm hiện tại"], [
        ["Mô hình chính", "LSTM xác suất trong framework", "Đã train lại đúng pipeline"],
        ["Benchmark", "Nhiều baseline gồm naive, ARIMA, Prophet", "Hai naive baseline và Prophet đã chạy"],
        ["Checkpoint tác giả", "Không công khai", "Tạo checkpoint mới trên máy"],
        ["Bảng metric LSTM tác giả", "Không có bảng tương ứng", "Báo cáo metric của lần chạy hiện tại"],
        ["Khoảng baseline", "Không đủ chi tiết để khẳng định", "Calibration bằng residual validation"],
    ], [1.6, 2.65, 2.75], font_size=8.7)

    heading(doc, "12 Các file và artifact đã tạo", 1)
    add_table(doc, ["File hoặc thư mục", "Nội dung", "Dùng cho"], [
        ["paper source", "Source paper cố định tại commit chính thức", "Tái lập code gốc"],
        ["opsd recurrent pkl", "Checkpoint tốt nhất của LSTM", "Evaluate và demo"],
        ["training log", "Log 41 epoch", "Giải thích diễn biến train"],
        ["evaluation opsd recurrent csv", "Metric tổng hợp LSTM", "Bảng kết quả"],
        ["eval figures", "Metric theo horizon và forecast mẫu", "Báo cáo và slide"],
        ["run two baselines py", "Script baseline công bằng", "Chạy lại so sánh"],
        ["two baseline comparison csv", "Bảng so sánh ba mô hình", "Kết luận định lượng"],
        ["run prophet paper protocol py", "Tái tạo phần Prophet của notebook paper", "Fit và đánh giá Prophet"],
        ["prophet paper protocol", "Metric, model JSON, forecast CSV và hình", "Theo dõi Prophet"],
        ["run metadata json", "Split, timestamp, SHA256 và scaler", "Kiểm tra khả năng tái lập"],
    ], [2.15, 2.75, 2.1], font_size=8.7)
    heading(doc, "12 1 Lệnh chạy lại", 2)
    body(doc, "Các lệnh dưới đây được chạy sau khi kích hoạt môi trường conda proloaf paper repro.")
    add_table(doc, ["Mục đích", "Lệnh"], [
        ["Train lại model", "python .\\src\\train.py -s opsd"],
        ["Evaluate checkpoint", "$env:MPLBACKEND = Agg rồi python .\\src\\evaluate.py -s opsd"],
        ["Chạy hai baseline", "python .\\reproduction\\run_two_baselines.py"],
        ["Chạy Prophet", "python .\\reproduction\\run_prophet_paper_protocol.py"],
    ], [2.2, 4.8])

    heading(doc, "13 Những giới hạn còn lại", 1)
    bullet(doc, "Mô hình mới được train một lần trên CPU. Chưa chạy nhiều random seed để ước lượng độ biến động của kết quả.")
    bullet(doc, "Chưa có bảng số của tác giả để xác nhận mức sai khác giữa lần chạy này và môi trường gốc.")
    bullet(doc, "Hai baseline hiện tại đủ cho so sánh cơ bản nhưng chưa đại diện cho toàn bộ nhóm mô hình thống kê và machine learning.")
    bullet(doc, "Khoảng dự báo baseline là cách mở rộng của bài làm, không phải artifact có sẵn từ paper.")
    bullet(doc, "Prophet đã được chạy theo notebook, nhưng protocol của notebook dùng 80 20 và scaling toàn bộ data. Vì vậy, bảng LSTM Prophet là đối chiếu chỉ báo chứ chưa phải thiết kế benchmark cùng ngân sách dữ liệu.")
    bullet(doc, "Các forecast origins trượt từng giờ và chồng lặp, vì vậy 8 595 origins không tương đương 8 595 ngày độc lập.")
    bullet(doc, "PICP của LSTM cao hơn 95 phần trăm, cho thấy vẫn còn cơ hội hiệu chỉnh để thu hẹp khoảng dự báo.")

    heading(doc, "14 Hướng làm tiếp theo", 1)
    heading(doc, "14 1 Ưu tiên một hoàn thiện demo", 2)
    body(doc, "Bước tiếp theo quan trọng nhất là xây demo dùng checkpoint đã train. Demo nên cho phép chọn một forecast origin trong test set, hiển thị 147 giờ lịch sử, 24 giờ actual, median dự báo và khoảng 95 phần trăm. Bên cạnh biểu đồ, giao diện cần hiển thị RMSE, MAE và coverage của cửa sổ đang chọn.")
    add_table(doc, ["Đầu vào demo", "Xử lý", "Đầu ra"], [
        ["Ngày hoặc forecast origin", "Nạp dữ liệu và tạo cửa sổ 147 cộng 24 giờ", "Thông tin mẫu được chọn"],
        ["Checkpoint opsd recurrent pkl", "Chạy inference không cập nhật trọng số", "Median và hai biên 95 phần trăm"],
        ["Actual test", "Ghép theo timestamp", "Đường actual để đối chiếu"],
        ["Prediction và actual", "Tính metric theo cửa sổ", "RMSE MAE coverage"],
    ], [2.0, 2.8, 2.2], font_size=8.8)
    heading(doc, "14 2 Ưu tiên hai hoàn thiện báo cáo và slide", 2)
    bullet(doc, "Chọn 10 đến 12 hình và bảng quan trọng từ báo cáo, không đưa toàn bộ chi tiết code lên slide.")
    bullet(doc, "Dành khoảng 3 phút cho bài toán và dữ liệu, 5 phút cho pipeline, 4 phút cho kết quả, 2 phút demo và 1 phút kết luận.")
    bullet(doc, "Chuẩn bị câu trả lời cho lý do dùng 147 giờ, ý nghĩa PICP, early stopping ở epoch 41 và giới hạn khi paper không công bố metric đầy đủ.")
    heading(doc, "14 3 Ưu tiên ba kiểm tra khả năng chạy lại", 2)
    body(doc, "1. Viết README với đúng ba lệnh train, evaluate và baseline.")
    body(doc, "2. Ghi rõ phiên bản Python và PyTorch đã dùng.")
    body(doc, "3. Không đẩy dataset hoặc file tạm quá lớn nếu GitHub đã có nguồn tải rõ ràng.")
    body(doc, "4. Đảm bảo checkpoint, metric CSV, hình kết quả và script baseline nằm trong repository nộp bài.")
    heading(doc, "14 4 Nâng cấp nếu còn thời gian", 2)
    body(doc, "Calibration variance trên validation đã hoàn thành và đưa PICP test từ 97,10 phần trăm về 95,06 phần trăm. Nếu còn thời gian sau demo và slide, nâng cấp tiếp theo nên là dùng một calibration set tách riêng khỏi early stopping hoặc thử calibration riêng cho từng horizon. Nếu cần benchmark công bằng hơn với Prophet, hãy retrain LSTM trên 80 phần trăm đầu dữ liệu, fit scaler chỉ trên tập train và đánh giá cả hai mô hình trên cùng timestamp. SARIMA chỉ nên thêm khi còn đủ thời gian.")

    heading(doc, "15 Kết luận", 1)
    body(doc, "Đến thời điểm hiện tại, phần tái lập cốt lõi đã hoàn thành. Chúng ta đã chạy source của paper, dùng dữ liệu OPSD, train lại Encoder Decoder LSTM, tạo checkpoint, đánh giá trên test set, so sánh với hai baseline và chạy Prophet từ notebook benchmark. LSTM có sai số thấp hơn Prophet trong kết quả hiện tại, đồng thời tạo khoảng dự báo hẹp hơn; kết luận này cần đi kèm lưu ý về khác biệt split giữa hai protocol.")
    body(doc, "Ngoài tái lập, bài làm còn có một cải tiến hậu xử lý đã được kiểm tra đúng quy trình validation test: standard deviation scaling hiệu chỉnh khoảng dự báo từ coverage 97,10 về 95,06 phần trăm mà không thay đổi RMSE hay MAE. Phần còn thiếu để hoàn thành bài giữa kỳ là demo có thể trình bày trực tiếp, slide tối đa 15 trang và rà soát README GitHub.")

    heading(doc, "Tài liệu tham khảo", 1)
    body(doc, "1. ProLoaF Probabilistic load forecasting for power systems. SoftwareX 2023. DOI 10.1016/j.softx.2023.101487.")
    body(doc, "2. ElsevierSoftwareX SOFTX D 21 00190. https://github.com/ElsevierSoftwareX/SOFTX-D-21-00190")
    body(doc, "3. Open Power System Data Time series. https://data.open-power-system-data.org/time_series/2020-10-06")

    doc.core_properties.title = "Báo cáo tái lập thực nghiệm ProLoaF"
    doc.core_properties.subject = "Dự báo xác suất phụ tải điện Đức trước 24 giờ"
    doc.core_properties.author = ""
    doc.core_properties.keywords = "ProLoaF OPSD LSTM probabilistic load forecasting"
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
