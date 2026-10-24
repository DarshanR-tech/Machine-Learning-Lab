"""
Script to generate the final PDF report from report_template.html
"""

import os
import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import subprocess

BASE_DIR = r"c:\Users\thami\BRO-AI\IrisFlowerClassification"
TEMPLATE_HTML = os.path.join(BASE_DIR, "pdf_report", "report_template.html")
FINAL_HTML = os.path.join(BASE_DIR, "pdf_report", "final_report.html")
OUTPUT_PDF = os.path.join(BASE_DIR, "Iris_Flower_Classification_Project_Report.pdf")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "pdf_report", "screenshots")

def uri_for(filename):
    p = os.path.join(SCREENSHOTS_DIR, filename).replace("\\", "/")
    return f"file:///{p}"

def main():
    print("Reading report_template.html...")
    with open(TEMPLATE_HTML, "r", encoding="utf-8") as f:
        html = f.read()

    replacements = {
        "{{FIG1_URI}}": uri_for("fig1_project_structure.png"),
        "{{FIG2_URI}}": uri_for("fig2_dataset_collection.png"),
        "{{FIG3_URI}}": uri_for("fig3_preprocessing.png"),
        "{{FIG4_URI}}": uri_for("fig4_eda_species_sepal.png"),
        "{{FIG5_URI}}": uri_for("fig5_eda_petal_dist.png"),
        "{{FIG6_URI}}": uri_for("fig6_eda_correlation.png"),
        "{{FIG7_URI}}": uri_for("fig7_model_training.png"),
        "{{FIG8_URI}}": uri_for("fig8_model_evaluation.png"),
        "{{FIG9_URI}}": uri_for("fig9_prediction_setosa.png"),
        "{{FIG10_URI}}": uri_for("fig10_prediction_versicolor.png"),
        "{{FIG11_URI}}": uri_for("fig11_prediction_virginica.png"),
        "{{FIG12_URI}}": uri_for("fig12_input_validation_error.png")
    }

    for placeholder, val in replacements.items():
        html = html.replace(placeholder, val)

    with open(FINAL_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✓ Prepared rendered HTML: {FINAL_HTML}")

    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    if not os.path.exists(edge_path):
        edge_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

    print("Converting to PDF via Microsoft Edge headless...")
    cmd = [
        edge_path,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={OUTPUT_PDF}",
        "--no-pdf-header-footer",
        FINAL_HTML
    ]
    subprocess.run(cmd, check=True)

    if os.path.exists(OUTPUT_PDF):
        size_kb = os.path.getsize(OUTPUT_PDF) / 1024
        print(f"\n=======================================================")
        print(f"🎉 PDF REPORT GENERATED SUCCESSFULLY!")
        print(f"Location: {OUTPUT_PDF}")
        print(f"Size:     {size_kb:.1f} KB")
        print(f"=======================================================")
    else:
        print("❌ Error: Output PDF was not created.")

if __name__ == "__main__":
    main()
