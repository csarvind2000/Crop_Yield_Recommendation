#!/usr/bin/env python3
"""
Combine all PNG figures in a directory tree into one PDF
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

def collect_pngs(root: Path):
    return sorted([p for p in root.rglob("*.png")])

def pngs_to_pdf(png_files, pdf_path):
    with PdfPages(pdf_path) as pdf:
        for img_path in png_files:
            fig, ax = plt.subplots(figsize=(8.27, 11.7))  # A4 portrait
            ax.axis("off")
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.set_title(img_path.name, fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Directory containing PNGs (will search recursively)")
    ap.add_argument("--out", required=True, help="Output PDF file")
    args = ap.parse_args()

    root = Path(args.indir)
    pngs = collect_pngs(root)
    if not pngs:
        print("❌ No PNG files found in", root)
        return
    print(f"Found {len(pngs)} PNG files. Writing to {args.out} ...")
    pngs_to_pdf(pngs, Path(args.out))
    print("✅ PDF created:", args.out)

if __name__ == "__main__":
    main()


# python pngs_to_pdf.py --indir eda_outputs --out eda_outputs/eda_report.pdf
