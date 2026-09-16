"""Assemble the Data Visualizations section document.

Reads every figure from reports/figures/ and pairs it with an explanation of its
purpose, the reason that chart type was chosen, the design decisions behind it,
and what it shows. Writes reports/Data_Visualizations.docx.

Run the figure generators first:

    python -m carebridge.viz.descriptives
    python -m carebridge.viz.eda
    python -m carebridge.reports.visualizations

Missing figures are skipped with a warning rather than crashing, so the document
can be built at any stage.
"""
from __future__ import annotations

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from carebridge.config import REPORTS

FIGS = REPORTS / "figures"
OUT = REPORTS / "Data_Visualizations.docx"

NAVY = RGBColor(0x1F, 0x38, 0x64)
SLATE = RGBColor(0x44, 0x54, 0x6A)
ACCENT = RGBColor(0xA6, 0x46, 0x2E)
FONT = "Calibri"

_n = {"fig": 0}
_missing: list[str] = []


def _style(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.12

    for s in doc.sections:
        s.top_margin = s.bottom_margin = Inches(0.9)
        s.left_margin = s.right_margin = Inches(1)

    for name, size, colour in [("Heading 1", 15, NAVY), ("Heading 2", 12, SLATE)]:
        st = doc.styles[name]
        st.font.name = FONT
        st.font.size = Pt(size)
        st.font.color.rgb = colour
        st.font.bold = True
        st.paragraph_format.space_before = Pt(13)
        st.paragraph_format.space_after = Pt(5)


def para(doc, text, size=10.5, after=6):
    """Paragraph supporting **bold** inline markup."""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    for i, chunk in enumerate(text.split("**")):
        if chunk:
            r = p.add_run(chunk)
            r.bold = i % 2 == 1
            r.font.size = Pt(size)
    return p


def bullet(doc, text, size=10):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    for i, chunk in enumerate(text.split("**")):
        if chunk:
            r = p.add_run(chunk)
            r.bold = i % 2 == 1
            r.font.size = Pt(size)
    return p


def _label(doc, word, text, colour):
    """A bold inline label followed by body text, e.g. 'Purpose. ...'"""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.left_indent = Inches(0.12)
    r = p.add_run(word + "  ")
    r.bold = True
    r.font.size = Pt(10)
    r.font.color.rgb = colour
    for i, chunk in enumerate(text.split("**")):
        if chunk:
            rr = p.add_run(chunk)
            rr.bold = i % 2 == 1
            rr.font.size = Pt(10)
    return p


def entry(doc, filename, title, purpose, chart_type, decisions, shows,
          width=6.1):
    """One catalogue entry: heading, figure, then the four explanation blocks."""
    path = FIGS / filename
    if not path.exists():
        _missing.append(filename)
        print(f"  [skip] {filename}")
        return False

    _n["fig"] += 1
    doc.add_heading(f"Figure {_n['fig']} — {title}", level=2)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    p.add_run().add_picture(str(path), width=Inches(width))

    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(8)
    r = cap.add_run(filename)
    r.italic = True
    r.font.size = Pt(8)
    r.font.color.rgb = SLATE

    _label(doc, "Purpose.", purpose, NAVY)
    _label(doc, "Why this chart type.", chart_type, NAVY)
    _label(doc, "Design decisions.", decisions, NAVY)
    _label(doc, "What it shows.", shows, ACCENT)
    print(f"  [fig ] {filename}")
    return True


def build() -> Document:
    doc = Document()
    _style(doc)

    # ---------------------------------------------------------- title
    t = doc.add_paragraph()
    r = t.add_run("Data Visualizations")
    r.bold = True
    r.font.size = Pt(20)
    r.font.color.rgb = NAVY
    st = doc.add_paragraph()
    rs = st.add_run("CareBridge — Diabetes Readmission Risk and Cardiometabolic Comorbidity")
    rs.font.size = Pt(11)
    rs.font.color.rgb = SLATE
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    a = p.add_run("Team Lead:  ")
    a.bold = True; a.font.size = Pt(10); a.font.color.rgb = NAVY
    b = p.add_run("Antoine Ward  ·  B.S. Data Science Capstone")
    b.font.size = Pt(10)

    # ---------------------------------------------------------- intro
    doc.add_heading("1. Purpose and Conventions", level=1)
    para(doc, "Each entry below states what question the figure serves, why that chart type "
              "was chosen over the alternatives, which design decisions make it readable or "
              "honest, and what it shows. Every figure is generated by the analysis pipeline "
              "and rebuilt on each run.")

    bullet(doc, "**Colour carries meaning.** Navy for primary series, mid-blue for secondary, "
                "slate for annotation. One warm accent is reserved for things that mean "
                "*attention* — a threshold crossed, a value excluded, an anomaly.")
    bullet(doc, "**Confidence intervals wherever a rate appears.** Group sizes span two orders "
                "of magnitude here (491 to 52,666). Bare point estimates would imply uniform "
                "precision that does not exist.")
    bullet(doc, "**Baselines and sample sizes are shown.** A rate is only interpretable "
                "against the cohort rate, and 0% on n = 1,642 means something different from "
                "0% on n = 2. Levels below 100 observations are suppressed.")
    bullet(doc, "**Discrete data is drawn as discrete.** Variables taking few integer values "
                "are plotted as exact counts, not binned into histograms that would impose "
                "arbitrary boundaries on already-separated values.")
    bullet(doc, "**No frame, gridlines behind the data.** Top and right spines removed. "
                "Neither carries information; both compete with the data for attention.")

    # ---------------------------------------------------------- DQ
    doc.add_heading("2. Data Quality Visualizations", level=1)
    para(doc, "These figures support cleaning decisions. Each one changed a choice in the "
              "pipeline.", after=4)

    entry(doc, "dq_encounters_missingness.png",
          "Missingness by Column",
          purpose="Decide which columns to use, which need missingness encoded as a category, "
                  "and which to drop. A decision figure, not a descriptive one.",
          chart_type="Ranked horizontal bars. Horizontal because the labels are column names — "
                     "words, not numbers — and vertical bars would force rotated labels.",
          decisions="The x-axis is **fixed 0 to 100** rather than scaled to the data: without "
                    "that anchor, a chart topping out at 12% looks identical to one topping "
                    "out at 97%. Bars above 50% take the accent colour because that threshold "
                    "separates *encode as category* from *drop the column*, so colour carries "
                    "a decision rather than decoration.",
          shows="Weight is ~97% missing and is dropped; imputation at that rate would be model "
                "output presented as observation. Medical specialty (49.1%) and payer code "
                "(39.5%) are retained with missingness encoded explicitly, since which fields "
                "go unrecorded may itself carry signal.",
          width=4.9)

    entry(doc, "dq_disposition_audit.png",
          "Readmission Rate by Discharge Disposition",
          purpose="Validate the leakage exclusion list empirically. The Kaggle mirror ships a "
                  "description PDF instead of the ID lookup, so code meanings had to be tested "
                  "rather than read. A code meaning *expired* predicts a rate of exactly zero.",
          chart_type="Vertical bars against a horizontal reference line. The line because each "
                     "bar is meaningful only relative to the 11.16% baseline, which a reader "
                     "cannot hold in mind while scanning five bars.",
          decisions="Colour encodes the **decision**, not the value — accent for excluded, "
                    "blue for retained. Sample size prints under each bar, because 0% on 1,642 "
                    "encounters is evidence and 0% on 2 is not. Rates show three decimals so "
                    "*exactly* zero is distinguishable from *approximately* zero; that "
                    "precision is the claim.",
          shows="Codes 11, 19, 20 return exactly 0.000% across 1,652 encounters. Codes 13 and "
                "14 return 4.762% and 6.452% — half the baseline, but unambiguously nonzero. "
                "The six codes conventionally excluded as one group do not behave as one.",
          width=5.2)

    # ---------------------------------------------------------- distributions
    doc.add_heading("3. Distribution Visualizations", level=1)
    para(doc, "These characterize variable shape, which determines whether a linear "
              "specification is appropriate and which transforms are required.", after=4)

    entry(doc, "eda_numeric_distributions.png",
          "Distribution of Numeric Features",
          purpose="Establish the shape of every numeric predictor before a model assumes "
                  "anything about it.",
          chart_type="**Small multiples** — eight panels sharing a layout. Separate figures "
                     "would let a reader compare each variable to itself but not to the "
                     "others; the grid makes relative shape immediately legible.",
          decisions="Each panel picks its own renderer: 25 or fewer distinct values are drawn "
                    "as **exact counts**, everything else as a histogram. Length of stay takes "
                    "integers 1–14, and binning it would create gaps and doubled bars that do "
                    "not exist in the data. Skewness is annotated per panel and printed in "
                    "accent beyond |2|, so the figure flags its own outliers.",
          shows="Three prior-utilization variables are severely skewed. Lab procedures and "
                "diagnoses recorded are mildly *left*-skewed and need no transformation — "
                "describing the feature set as uniformly right-skewed would be inaccurate.",
          width=5.9)

    entry(doc, "eda_los_distribution.png",
          "Length of Stay, Linear and Logarithmic",
          purpose="Motivate the distributional choice for H1e. Poisson and negative binomial "
                  "make different assumptions about spread, and this is the outcome of the "
                  "count-regression component.",
          chart_type="Two panels, same data, different scales. Left is a bar chart of exact "
                     "counts, because plotting a discrete count as continuous would undercut "
                     "the argument the figure exists to make. Right is log-scaled, because a "
                     "long thin right tail is invisible on a linear axis.",
          decisions="The left panel carries an **annotation box reporting mean, variance, and "
                    "their ratio** — Poisson requires variance to equal the mean exactly, so "
                    "the ratio is the direct test. Printing it on the chart means the figure "
                    "argues its own case rather than deferring to surrounding prose. Placed in "
                    "axes coordinates so it stays in the corner regardless of data scaling.",
          shows="A variance-to-mean ratio near 2 — double what Poisson permits. Descriptive "
                "evidence for H1e that precedes the formal test, which later confirms it.",
          width=5.7)

    # ---------------------------------------------------------- bivariate
    doc.add_heading("4. Outcome Relationship Visualizations", level=1)
    para(doc, "These establish which features carry univariate signal before modelling, and "
              "give a prior for what the multivariable model should find.", after=4)

    entry(doc, "eda_readmission_by_decile.png",
          "Readmission Rate by Feature Decile",
          purpose="Identify which numeric features separate readmitted patients, and whether "
                  "the relationship is monotone. A flat line means no univariate signal.",
          chart_type="Small multiples as **line-with-error-bar** plots rather than bars. "
                     "Deciles are ordered, so a line encodes the shape of the relationship in "
                     "a way categorical bars cannot. Slope is the finding; a line renders slope.",
          decisions="Binomial **confidence intervals** at every point — quantile boundaries "
                    "collapse on mostly-zero variables, so bins are unequal and precision "
                    "genuinely varies. The **number of bins actually formed prints in each "
                    "panel title**, an honesty measure: a reader seeing '2 bins' knows not to "
                    "read that panel as a decile curve.",
          shows="Prior inpatient visits produces the steepest relationship by a wide margin. "
                "Prior emergency collapses to a **single bin** at 92.71% zeros, making its "
                "spread exactly zero and the variable appear uninformative — which the binary "
                "encoding contradicts.",
          width=5.9)

    entry(doc, "eda_readmission_by_category.png",
          "Readmission Rate by Categorical Level",
          purpose="Identify which categorical levels carry elevated risk, and provide the "
                  "baseline picture for the later fairness audit.",
          chart_type="Horizontal bars with error bars, **sorted by rate rather than by level**, "
                     "because the ordering is itself the finding — the reader wants to know "
                     "which categories rank highest, not to read them alphabetically.",
          decisions="Levels below 100 observations are suppressed, and sample size prints in "
                    "each remaining label. Bars above baseline take the accent colour, so "
                    "elevated-risk categories are identifiable without reading values. Error "
                    "bars matter here because category sizes span two orders of magnitude.",
          shows="Supplementary diagnosis codes — aftercare and complications of prior "
                "treatment — rank highest, then injury and mental disorders. Clinically "
                "coherent: all three describe elevated care-transition risk. Unadjusted race "
                "rates span 2.25 points with substantially overlapping intervals.",
          width=5.7)

    entry(doc, "eda_correlation_matrix.png",
          "Correlation Among Numeric Predictors",
          purpose="Detect multicollinearity before fitting a regression. Three features divide "
                  "a count by length of stay, and length of stay is itself a predictor — a "
                  "structural risk that would destabilize coefficients.",
          chart_type="A **heatmap**. Thirteen predictors produce 169 pairwise correlations; a "
                     "table that size is unreadable, while colour intensity locates the strong "
                     "relationships in one pass.",
          decisions="**Spearman, not Pearson** — several features are heavily skewed counts, "
                    "where a linear coefficient is dominated by extreme values. The colormap "
                    "is **diverging** and anchored at ±1, because correlation has a meaningful "
                    "midpoint and sign that a sequential map would obscure. Diagonals omitted; "
                    "text colour flips on saturated cells so every annotation stays legible.",
          shows="No correlation strong enough to threaten estimation. Rate features and their "
                "underlying counts are mildly related but not redundant, confirmed by variance "
                "inflation factors below 5. All thirteen predictors retained.",
          width=4.9)

    # ---------------------------------------------------------- RQ2
    doc.add_heading("5. Population Survey Visualizations", level=1)
    para(doc, "These support RQ2 — predictors of diabetes, stroke, and cardiac disease at "
              "population level, and how strongly those conditions co-occur.", after=4)

    entry(doc, "eda_outcome_balance.png",
          "Class Balance Across All Four Outcomes",
          purpose="Establish prevalence, which determines the evaluation metric. "
                  "Precision-recall is required at low prevalence; ROC-based measures flatter "
                  "models on imbalanced data.",
          chart_type="Paired bar charts, one panel per source. Separate panels because the two "
                     "sources differ in scale and grain — shared axes would misrepresent both.",
          decisions="The left panel prints **both count and percentage**: the count establishes "
                    "scale, the percentage establishes prevalence, and prevalence drives the "
                    "metric choice. Only the target class takes the primary colour. Axis limits "
                    "are extended so labels do not collide with the title.",
          shows="Prevalence runs 4.1% (stroke) to 13.9% (diabetes). Stroke is rarest and will "
                "be hardest to model, with the widest intervals on any predictor ranking.",
          width=5.5)

    entry(doc, "eda_condition_cooccurrence.png",
          "Co-occurrence of the Three Survey Conditions",
          purpose="Quantify how strongly the three conditions appear together — the basis for "
                  "the comorbidity component of RQ2, and for whether one diagnosis should "
                  "trigger screening for the others.",
          chart_type="A **heatmap of conditional prevalences**. The matrix is deliberately "
                     "asymmetric — the share of diabetics who have had a stroke is not the "
                     "share of stroke patients with diabetes, since base rates differ "
                     "threefold — and a heatmap preserves what a symmetric plot would destroy.",
          decisions="Each off-diagonal cell reports the conditional percentage **and the lift "
                    "over base rate**. Lift is essential: 22% co-occurrence means nothing "
                    "without knowing whether 22% of everyone has the condition. The colormap "
                    "is **sequential**, not diverging, because values run from none to a lot "
                    "with no meaningful midpoint, and is anchored at zero so contrast is "
                    "stable across runs.",
          shows="All three pairs co-occur well above independence. Stroke and cardiac disease "
                "link at ~4.1× base rate, roughly twice either one's association with "
                "diabetes. The conditions do not cluster uniformly, suggesting any shared "
                "predictor core may prove cardiovascular rather than common to all three.",
          width=4.3)

    entry(doc, "eda_bmi_by_diabetes_status.png",
          "Body Mass Index by Diabetes Status",
          purpose="Assess whether BMI discriminates diabetes status, and by how much — the "
                  "descriptive form of the hypothesis that predictor importance differs across "
                  "the three outcomes.",
          chart_type="**Overlapping** density histograms rather than side-by-side, because the "
                     "question is about distributional overlap and adjacent panels make that "
                     "comparison much harder to make visually.",
          decisions="**Density, not counts** — mandatory rather than stylistic at an 86/14 "
                    "split, where raw counts would render the diabetes distribution as an "
                    "invisible sliver. **Shared bin edges** are equally mandatory: "
                    "independently binned histograms would not align and the comparison would "
                    "be invalid. Partial transparency keeps the overlap readable.",
          shows="Clear separation, with group means on opposite sides of the obesity "
                "threshold. Standardized mean difference 0.641, against 0.181 for cardiac "
                "disease and 0.102 for stroke — BMI discriminates diabetes several times more "
                "strongly than the cardiovascular outcomes.",
          width=4.5)

    entry(doc, "eda_income_gradient.png",
          "Diabetes Prevalence by Income Band",
          purpose="Determine whether income relates to prevalence as a continuous gradient or "
                  "a threshold. The distinction has policy consequences: a threshold justifies "
                  "cutoff-based eligibility, a gradient argues against it.",
          chart_type="A **line plot with error bars** across ordered categories. A line because "
                     "the *shape* is the finding — bars would render the same values while "
                     "discarding the continuity that distinguishes gradient from threshold.",
          decisions="Binomial intervals at every point, and they matter more than usual: band "
                    "sizes run 9,811 to 90,385, so precision visibly varies along the curve. "
                    "Every band gets an explicit tick, since these are ordered categories and "
                    "automatic selection would label only some. The anomalous band is **marked "
                    "in accent and annotated**, so the irregularity is encountered in the "
                    "figure rather than discovered in the text. The axis label states the "
                    "coding direction, because a reader assuming the opposite would read the "
                    "chart backwards.",
          shows="A 3.05-fold decline from the second-lowest to the highest band, monotonic "
                "across bands 2–8 — consistent with a graded relationship. The lowest band "
                "departs, with an interval that does not overlap the band above it. Since the "
                "outcome measures *diagnosed* diabetes, reduced healthcare contact may lower "
                "detection rather than disease.",
          width=5.1)

    # ---------------------------------------------------------- summary
    doc.add_heading("6. Summary of Design Rationale", level=1)
    para(doc, "**The encoding must match the data type.** Discrete counts are drawn as counts, "
              "ordered categories as lines, asymmetric relationships in asymmetric matrices. "
              "Each alternative would render the same numbers while discarding a property the "
              "figure exists to communicate — and for length of stay, would actively undercut "
              "the argument being made about it.")
    para(doc, "**Uncertainty is shown, not implied.** Group sizes span two orders of magnitude. "
              "Every rate carries binomial intervals, every categorical label carries its "
              "sample size, and categories too small to estimate are suppressed rather than "
              "plotted.")
    para(doc, "**Colour carries meaning or is not used.** One warm accent appears throughout, "
              "and only to mark something requiring attention: a missingness rate above the "
              "drop threshold, a code excluded for leakage, a skewness beyond the conventional "
              "bound, a band breaking a monotonic pattern. Using it decoratively anywhere "
              "would weaken it everywhere.")
    para(doc, "**The figure should argue its own case.** Where one statistic is the point, it "
              "is printed on the figure: the variance-to-mean ratio, the lift multiple, the "
              "bin count, the standardized difference. A figure depending on surrounding prose "
              "will be misread the moment it is extracted into a slide.")

    if _missing:
        doc.add_heading("Figures not available at build time", level=2)
        for m in _missing:
            bullet(doc, m)

    return doc


def main() -> None:
    print("assembling data visualizations document...")
    doc = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"\nwrote {OUT}")
    print(f"  {_n['fig']} figures catalogued"
          + (f", {len(_missing)} missing" if _missing else ""))


if __name__ == "__main__":
    main()
