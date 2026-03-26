"""Generate commentary.pdf from commentary.md content."""

from fpdf import FPDF
import re


class CommentaryPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 5, "Grokking in Neural Networks: Commentary", align="C")
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def build_pdf(md_path, out_path):
    with open(md_path) as f:
        text = f.read()

    pdf = CommentaryPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    for line in text.split("\n"):
        stripped = sanitize(line.strip())

        if stripped.startswith("# ") and not stripped.startswith("## "):
            pdf.set_font("Helvetica", "B", 18)
            pdf.ln(4)
            pdf.multi_cell(0, 9, stripped[2:])
            pdf.ln(4)

        elif stripped.startswith("## "):
            pdf.set_font("Helvetica", "B", 14)
            pdf.ln(6)
            pdf.multi_cell(0, 8, stripped[3:])
            pdf.ln(2)

        elif stripped.startswith("### "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.ln(4)
            pdf.multi_cell(0, 7, stripped[4:])
            pdf.ln(2)

        elif stripped.startswith("- **"):
            pdf.set_font("Helvetica", "", 10)
            clean = stripped[2:]
            clean = re.sub(r"\*\*(.+?)\*\*", r"\1", clean)
            clean = re.sub(r"`(.+?)`", r"\1", clean)
            pdf.cell(5)
            pdf.multi_cell(0, 6, f"- {clean}")
            pdf.ln(1)

        elif stripped.startswith("- "):
            pdf.set_font("Helvetica", "", 10)
            clean = stripped[2:]
            clean = re.sub(r"\*\*(.+?)\*\*", r"\1", clean)
            clean = re.sub(r"`(.+?)`", r"\1", clean)
            pdf.cell(5)
            pdf.multi_cell(0, 6, f"- {clean}")
            pdf.ln(1)

        elif stripped.startswith(("1.", "2.", "3.")):
            pdf.set_font("Helvetica", "", 10)
            clean = re.sub(r"\*\*(.+?)\*\*", r"\1", stripped)
            clean = re.sub(r"`(.+?)`", r"\1", clean)
            pdf.cell(5)
            pdf.multi_cell(0, 6, clean)
            pdf.ln(1)

        elif stripped == "":
            pdf.ln(3)

        else:
            pdf.set_font("Helvetica", "", 10)
            clean = re.sub(r"\*\*(.+?)\*\*", r"\1", stripped)
            clean = re.sub(r"`(.+?)`", r"\1", clean)
            pdf.multi_cell(0, 6, clean)

    pdf.output(out_path)


def sanitize(text):
    """Replace non-latin1 characters."""
    replacements = {
        '\u2014': '--',   # em dash
        '\u2013': '-',    # en dash
        '\u2018': "'",    # left single quote
        '\u2019': "'",    # right single quote
        '\u201c': '"',    # left double quote
        '\u201d': '"',    # right double quote
        '\u2022': '-',    # bullet
        '\u2026': '...',  # ellipsis
        '\u03c4': 'tau',  # tau
        '\u2265': '>=',   # >=
        '\u2264': '<=',   # <=
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode('latin-1', errors='replace').decode('latin-1')
    print(f"Saved {out_path}")


if __name__ == "__main__":
    build_pdf("commentary.md", "commentary.pdf")
