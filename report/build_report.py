"""Build the exactly 25-page APA-style Project 3 report."""

from __future__ import annotations

import csv
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)
DOCX_PATH = OUTPUT / "Quantum_Radiation_Flow_APA_Report.docx"

BLACK = RGBColor(0, 0, 0)
NAVY = RGBColor(23, 34, 59)
CYAN = RGBColor(23, 140, 164)
AMBER = RGBColor(190, 112, 12)
LIGHT = "EEF2F7"
USABLE_DXA = 9120
BREAK_INDEX = 0
# Only the title, abstract, reference section, and three appendix code leaves
# require hard page boundaries. Body subsections flow continuously so APA
# double spacing does not create partially empty pages.
HARD_BREAKS = {1, 2, 20, 21, 22}


def set_font(run, name: str = "Times New Roman", size: float = 12, *, bold: bool = False, italic: bool = False, color: RGBColor = BLACK) -> None:
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = color


def page_field(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for element in (begin, instruction, separate, text, end):
        run._r.append(element)
    set_font(run, size=12)


def configure_document(document: Document) -> None:
    section = document.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(.45)
    section.footer_distance = Inches(.5)
    page_field(section.header.paragraphs[0])

    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.widow_control = True

    for style_name, size in (("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 12)):
        style = document.styles[style_name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = BLACK
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(8)
        style.paragraph_format.space_after = Pt(0)

    document.core_properties.title = "Deep Generative Modeling of Hawking Radiation"
    document.core_properties.subject = "Conditional normalizing flows, Page-curve benchmarks, and toy information recovery"
    document.core_properties.author = "Salem Morelli"
    document.core_properties.keywords = "black hole, Page curve, normalizing flow, variational inference, PyTorch"


def add_page_break(document: Document) -> None:
    global BREAK_INDEX
    BREAK_INDEX += 1
    if BREAK_INDEX not in HARD_BREAKS:
        return
    paragraph = document.add_paragraph()
    paragraph.add_run().add_break(WD_BREAK.PAGE)


def add_page_heading(document: Document, title: str, *, continuation: bool = False) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(8)
    run = paragraph.add_run(title + (" (continued)" if continuation else ""))
    set_font(run, size=14, bold=True)


def add_body(document: Document, text: str, *, indent: bool = True, size: float = 12, spacing: float = 2.0) -> None:
    for block in [part.strip() for part in text.strip().split("\n\n") if part.strip()]:
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.first_line_indent = Inches(.5) if indent else Inches(0)
        paragraph.paragraph_format.line_spacing = spacing
        paragraph.paragraph_format.space_after = Pt(0)
        run = paragraph.add_run(block)
        set_font(run, size=size)


def add_centered(document: Document, text: str, *, size: float = 12, bold: bool = False, italic: bool = False, after: float = 0) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(after)
    run = paragraph.add_run(text)
    set_font(run, size=size, bold=bold, italic=italic)


def add_equation(document: Document, equation: str, number: int) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.line_spacing = 1.15
    paragraph.paragraph_format.space_before = Pt(5)
    paragraph.paragraph_format.space_after = Pt(5)
    run = paragraph.add_run(f"{equation}    ({number})")
    set_font(run, name="Cambria Math", size=11.5)


def add_figure(document: Document, filename: str, number: int, title: str, note: str, *, width: float = 6.25, alt: str) -> None:
    label = document.add_paragraph()
    label.paragraph_format.space_before = Pt(5)
    label.paragraph_format.space_after = Pt(0)
    run = label.add_run(f"Figure {number}")
    set_font(run, size=10.5, bold=True)
    title_p = document.add_paragraph()
    title_p.paragraph_format.space_after = Pt(4)
    title_run = title_p.add_run(title)
    set_font(title_run, size=10.5, italic=True)
    image_p = document.add_paragraph()
    image_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    image_p.paragraph_format.space_after = Pt(3)
    # Preserve legibility while leaving enough vertical room for APA double-spaced
    # interpretation on the same page.
    picture = image_p.add_run().add_picture(
        str(ROOT / "figures" / filename), width=Inches(width * 0.88)
    )
    inline = picture._inline
    inline.docPr.set("descr", alt)
    note_p = document.add_paragraph()
    note_p.paragraph_format.line_spacing = 1.0
    note_p.paragraph_format.space_after = Pt(2)
    note_run = note_p.add_run("Note. " + note)
    set_font(note_run, size=9, italic=True)


def set_cell_width(cell, width_dxa: int) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width_dxa))
    tc_w.set(qn("w:type"), "dxa")


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    tc_pr.append(shading)


def set_cell_margins(cell, value: int = 95) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    margins = tc_pr.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        tc_pr.append(margins)
    for side in ("top", "left", "bottom", "right"):
        node = margins.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def add_table(document: Document, csv_name: str, number: int, title: str, widths: list[int], note: str) -> None:
    label = document.add_paragraph()
    label.paragraph_format.space_before = Pt(5)
    label.paragraph_format.space_after = Pt(0)
    run = label.add_run(f"Table {number}")
    set_font(run, size=10.5, bold=True)
    title_p = document.add_paragraph()
    title_p.paragraph_format.space_after = Pt(4)
    title_run = title_p.add_run(title)
    set_font(title_run, size=10.5, italic=True)
    with (ROOT / "tables" / csv_name).open(encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    table = document.add_table(rows=len(rows), cols=len(rows[0]))
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    table_pr = table._tbl.tblPr
    table_width = table_pr.find(qn("w:tblW"))
    if table_width is None:
        table_width = OxmlElement("w:tblW")
        table_pr.append(table_width)
    table_width.set(qn("w:w"), str(sum(widths)))
    table_width.set(qn("w:type"), "dxa")
    indent = OxmlElement("w:tblInd")
    indent.set(qn("w:w"), "120")
    indent.set(qn("w:type"), "dxa")
    table_pr.append(indent)
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        column = OxmlElement("w:gridCol")
        column.set(qn("w:w"), str(width))
        grid.append(column)
    for row_index, row in enumerate(rows):
        if row_index == 0:
            row_pr = table.rows[row_index]._tr.get_or_add_trPr()
            header = OxmlElement("w:tblHeader")
            header.set(qn("w:val"), "true")
            row_pr.append(header)
        for column_index, text in enumerate(row):
            cell = table.cell(row_index, column_index)
            set_cell_width(cell, widths[column_index])
            set_cell_margins(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if row_index == 0:
                set_cell_shading(cell, LIGHT)
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.line_spacing = 1.0
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            run = paragraph.add_run(text)
            set_font(run, size=8.2, bold=row_index == 0, color=NAVY if row_index == 0 else BLACK)
    note_p = document.add_paragraph()
    note_p.paragraph_format.line_spacing = 1.0
    note_p.paragraph_format.space_before = Pt(3)
    note_run = note_p.add_run("Note. " + note)
    set_font(note_run, size=8.8, italic=True)


def add_reference(document: Document, text: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.left_indent = Inches(.5)
    paragraph.paragraph_format.first_line_indent = Inches(-.5)
    paragraph.paragraph_format.line_spacing = 2.0
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run(text)
    set_font(run, size=10.5)


def add_code_page(document: Document, title: str, lines: list[str]) -> None:
    add_page_heading(document, title)
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.line_spacing = 1.0
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.keep_together = True
    for index, line in enumerate(lines, start=1):
        run = paragraph.add_run(line.expandtabs(4) + ("\n" if index < len(lines) else ""))
        set_font(run, name="Liberation Mono", size=5.7)


def build() -> None:
    global BREAK_INDEX
    BREAK_INDEX = 0
    document = Document()
    configure_document(document)

    # Page 1 - APA title page
    for _ in range(5):
        document.add_paragraph()
    add_centered(document, "Deep Generative Modeling of Hawking Radiation", size=16, bold=True, after=8)
    add_centered(document, "Conditional Normalizing Flows, Variational Inference, and the Statistical Geometry of Information Recovery", size=13, bold=True, after=30)
    add_centered(document, "Salem Morelli", size=12, after=4)
    add_centered(document, "PhD Research Program in Statistics", size=12, after=4)
    add_centered(document, "Project 3 Research Report", size=12, after=4)
    add_centered(document, "August 24, 2026", size=12)
    add_page_break(document)

    # Page 2 - abstract
    add_page_heading(document, "Abstract")
    add_body(document, "This report develops a differentiable statistical framework for a controlled toy version of black-hole information recovery. A finite-dimensional bipartite pure state supplies the quantum benchmark: radiation entropy is computed from its reduced density operator and compared with Page's exact Haar-average formula. A separate conditional RealNVP represents continuous detector telemetry. Its Jacobian provides exact sample-wise log densities; differential Shannon entropy remains a Monte Carlo expectation. Thus, a classical density model is not a density operator, and neural invertibility is not physical reversibility of an unknown quantum channel.", indent=False)
    add_body(document, "The experiment conditions every coupling block on time and stopped mass, trains by likelihood or reverse-KL variational free energy, and adds paired latent alignment. Alignment addresses a formal non-identifiability caused by the rotational symmetry of a spherical Gaussian base. A frozen reference scrambler and a Page-time visibility transition define an exact-density benchmark in which later radiation becomes correlated with a toy complex secret. Evaluation keeps quantum Page entropy, classical flow entropy, both KL directions, Monte Carlo error, round-trip error, and paired pure-state fidelity as distinct estimands.", indent=False)
    add_body(document, "Reference calculations locate the 10-qubit Page maximum at normalized time 0.5 with entropy 2.966 nats. They also demonstrate that telemetry entropy need not have the Page shape and that measurement entropy depends on basis. The results establish a falsifiable computational design, not evidence that physical Hawking radiation has been decoded.", indent=False)
    add_body(document, "Keywords: black-hole information paradox, Page curve, Hayden-Preskill protocol, normalizing flows, variational inference, conditional density estimation, PyTorch", indent=False, size=11)
    add_page_break(document)

    # Page 3
    add_page_heading(document, "Deep Generative Modeling of Hawking Radiation")
    add_body(document, "Hawking's semiclassical calculation assigns outgoing radiation an approximately thermal character while the black hole loses mass (Hawking, 1975). If evaporation maps a pure collapsing state to a terminal mixed state, ordinary unitary quantum mechanics fails. Page (1993) converted that conceptual tension into a quantitative trajectory: in a globally pure bipartite system, radiation entanglement rises while the radiation Hilbert space is smaller, reaches a maximum near equal dimensions, and falls as the remaining black-hole factor becomes smaller. Modern island and entanglement-wedge calculations reproduce this qualitative transition in controlled gravitational models (Almheiri et al., 2019; Penington, 2020).")
    add_body(document, "Project 3 asks a narrower computational question. Given a synthetic, time-indexed measurement channel whose outputs are strongly scrambled but statistically controlled, can a conditional invertible neural density estimator fit the telemetry and recover a paired information-bearing coordinate after a Page-time visibility transition? The answer is evaluated as a statistical experiment, not as a replacement for holography. The observable distribution, its latent representation, and the quantum state are connected by explicit maps and are never treated as interchangeable.")
    add_figure(document, "figure_1_estimand_map.png", 1, "Separation of the Quantum, Measurement, Flow, and Decoder Objects", "Arrows denote modeled maps or inferential comparisons. The diagram intentionally contains no equality between von Neumann entropy and classical differential entropy.", width=6.15, alt="Diagram separating quantum state entropy, measurement outcomes, conditional flow density, paired decoder, and statistical outputs.")
    add_page_break(document)

    # Page 4
    add_page_heading(document, "Quantum State and Entropy Framework")
    add_body(document, "Let the total Hilbert space at evaporation time t factor as H=H_B(t) tensor H_R(t), where B denotes the remaining black-hole degrees of freedom and R the collected radiation. In the unitary toy model the joint state is pure, |Psi_t>, and evolves by a unitary circuit U_t. The radiation observer has access only to the reduced density operator obtained by tracing over B.")
    add_equation(document, "|Ψ_t⟩ = U_t|Ψ_0⟩,    U_t†U_t = I", 1)
    add_equation(document, "ρ_R(t) = Tr_B(|Ψ_t⟩⟨Ψ_t|)", 2)
    add_body(document, "The radiation von Neumann entropy is basis invariant, nonnegative, and equal to the black-hole entropy for a globally pure bipartite state. It depends on the eigenvalue spectrum of rho_R rather than on a selected measurement basis.")
    add_equation(document, "S_R(t) = -Tr[ρ_R(t) log ρ_R(t)] = -Σ_j λ_j(t) log λ_j(t)", 3)
    add_body(document, "A computational density matrix must remain Hermitian, positive semidefinite, and trace one. The repository computes reduced states from pure vectors by tensor reshaping and contraction, symmetrizes numerical roundoff before eigendecomposition, clamps only negligible negative eigenvalues, and renormalizes the spectrum. These operations are safeguards, not physical dynamics. For larger systems, explicit density matrices scale exponentially and must be replaced by tensor-network, shadow-tomography, or structured neural-state estimators.")
    add_equation(document, "ρ_R = LL† / Tr(LL†)", 4)
    add_body(document, "Equation 4 gives a Cholesky-style parameterization for a future mixed-state tomography module. It is distinct from the normalizing flow, which parameterizes a probability density over classical outcomes.")
    add_page_break(document)

    # Page 5
    add_page_heading(document, "The Finite-Dimensional Page Benchmark")
    add_body(document, "For a Haar-random pure state on C^m tensor C^n with m less than or equal to n, Page's expectation for the entropy of the smaller subsystem is exact. Writing H_k for the kth harmonic number gives Equation 5. When radiation becomes the larger subsystem, purity implies that its entropy equals that of the smaller remaining black-hole factor; exchanging m and n therefore produces the descending branch.")
    add_equation(document, "E[S_m,n] = H_mn - H_n - (m - 1)/(2n)", 5)
    add_body(document, "This curve is a typical-state benchmark, not a trajectory generated by Hawking's local calculation. It assumes a global pure state and strong scrambling sufficient for typical-subsystem reasoning. Its Page time is a Hilbert-space balance point, which need not coincide exactly with one-half of mass for a realistic evaporation law. The repository uses normalized time only as a controlled index and reports remaining mass separately.")
    add_figure(document, "figure_2_page_curves.png", 2, "Exact Haar-Average Page Curves for Finite Qubit Systems", "Curves use Equation 5 in nats. The maximum occurs at equal Hilbert-space dimensions for even total qubit counts.", width=6.15, alt="Line chart showing symmetric Page curves for eight, ten, and twelve total qubits with maxima at normalized time one half.")
    add_page_break(document)

    # Page 6
    add_page_heading(document, "Hayden-Preskill Retrieval as a Statistical Design")
    add_body(document, "Hayden and Preskill (2007) considered a rapidly mixing unitary black hole entangled with previously emitted radiation. After the halfway point, newly deposited quantum information can be reconstructed from a modest additional radiation subsystem, assuming access to the early radiation and knowledge of the scrambling dynamics. The result is a decoupling statement about quantum subsystems, not a guarantee that an arbitrary classical learner can discover the decoder from unpaired measurement samples.")
    add_body(document, "The toy design operationalizes accessibility through a scalar visibility v(t). An information-bearing Gaussian coordinate s and independent nuisance epsilon are mixed before an exact conditional scrambler. Because both are standard Gaussian, the accessible latent y_t remains standard Gaussian for every t, making the reference density exactly normalized. Its correlation with s, however, rises across Page time.")
    add_equation(document, "y_t = v(t)s + √(1-v(t)^2) ε", 6)
    add_equation(document, "v(t) = sigmoid[(t-t_Page)/w]", 7)
    add_body(document, "The construction separates density difficulty from retrieval difficulty. Marginally, y_t has the same base law at every time, while its paired mutual dependence with the secret changes. A conditional frozen RealNVP then imposes nonlinear scrambling and a time-dependent Jacobian. The learned flow must undo that map, but faithful naming of its inverse coordinates requires alignment. Pre-Page failure is therefore produced by inaccessible signal rather than by a noninvertible network.")
    add_page_break(document)

    # Page 7
    add_page_heading(document, "The Information-Theoretic Bridge")
    add_body(document, "A detector is represented by a positive operator-valued measure {E_x}. The Born rule maps the radiation density operator to an outcome law. For discrete outcomes, p_t(x)=Tr[rho_R(t)E_x]; for continuous outcomes the notation denotes a density relative to a specified reference measure. The conditional flow approximates this classical law, not rho_R itself.")
    add_equation(document, "p_t(x) = Tr[ρ_R(t) E_x],    E_x ⪰ 0,    ∫E_x dx = I", 8)
    add_body(document, "If a rank-one projective measurement dephases rho in a fixed basis, the Shannon entropy of its outcome probabilities is at least the von Neumann entropy, with equality in an eigenbasis. A single basis generally loses phase information. Informationally complete tomography requires multiple settings or an informationally complete POVM, after which a constrained density model can be fitted through the Born likelihood.")
    add_table(document, "table_1_estimands.csv", 1, "Distinct Objects and Estimands", [2050, 3290, 3780], "The flow density and its entropy are classical. Quantum entropy requires a density operator or a justified estimator thereof.")
    add_page_break(document)

    # Page 8
    add_page_heading(document, "Measurement Entropy Is Not Von Neumann Entropy")
    add_body(document, "Consider a qubit density operator with eigenvalues 0.85 and 0.15. Its von Neumann entropy is 0.423 nats. Measuring in the eigenbasis produces the same binary entropy, but measuring in a complementary basis produces a uniform distribution and entropy log 2=0.693 nats. Both classical distributions can be fitted perfectly by a flow; only the first entropy equals S(rho).")
    add_figure(document, "figure_4_measurement_gap.png", 4, "Entropy of One State Under Different Measurement Bases", "The same density operator produces different outcome entropies. Equality with von Neumann entropy occurs in the eigenbasis in this example.", width=5.6, alt="Bar chart comparing von Neumann entropy, eigenbasis measurement entropy, and higher complementary-basis measurement entropy.")
    add_body(document, "Accordingly, an empirical flow curve can support claims about detector uncertainty, likelihood calibration, or accessible classical information. It cannot be relabeled as a Page curve without a theorem tying the measurement design and estimator to the spectrum of rho_R. The report reserves 'Page curve' for Equation 5 or a density-operator entropy estimate and calls the flow trajectory 'telemetry entropy.' This naming rule is enforced throughout the software outputs.")
    add_page_break(document)

    # Page 9
    add_page_heading(document, "Conditional Normalizing-Flow Geometry")
    add_body(document, "Let z in R^D follow the standard Gaussian density phi, and let c_t=(t,M_t) be the conditioning vector. A conditional bijection f_theta maps z to telemetry x. The change-of-variables theorem yields an exact normalized density whenever every layer is bijective and its Jacobian determinant is tractable.")
    add_equation(document, "x = f_θ(z;c_t),    z = f_θ^{-1}(x;c_t)", 9)
    add_equation(document, "log q_θ(x|c_t) = log φ(z) + log|det D_x f_θ^{-1}(x;c_t)|", 10)
    add_body(document, "A sequence of K transforms composes additively in log-Jacobian space. This is both computationally stable and statistically useful: likelihood, reverse-KL free energy, and differential entropy all reuse the same log-density calculation.")
    add_equation(document, "log|det Df_θ| = Σ_(k=1)^K log|det Df_k|", 11)
    add_body(document, "The phrase exact entropy requires qualification. Equation 10 gives exact log density for each floating-point sample. Differential entropy is h(q)=-E_q log q(X), so its generic estimate is a sample mean with Monte Carlo error. Exact sample-wise likelihood does not eliminate expectation error, model misspecification, or numerical approximation. The repository returns the entropy estimate and its standard error together.")
    add_page_break(document)

    # Page 10
    add_page_heading(document, "RealNVP Coupling Layers and Invertibility")
    add_body(document, "Partition u into masked coordinates u_A and transformed coordinates u_B. A conditional affine coupling block leaves u_A fixed and applies an elementwise scale and shift to u_B. Its Jacobian is triangular; therefore the determinant is the product of transformed-coordinate scales and the inverse is analytic.")
    add_equation(document, "v_A=u_A,    v_B=u_B⊙exp[s_θ(u_A,c)] + t_θ(u_A,c)", 12)
    add_equation(document, "u_B=(v_B-t_θ(v_A,c))⊙exp[-s_θ(v_A,c)]", 13)
    add_equation(document, "log|det J| = Σ_(j∈B) s_θ,j(u_A,c)", 14)
    add_body(document, "The implementation alternates masks and permutations so every coordinate can influence every other coordinate across depth. Log-scale is bounded as a*tanh(raw/a). This guarantees finite positive scale for finite network outputs and limits local condition numbers. Zero initialization of each final conditioner layer makes the initial transformation the identity, reducing early optimization shocks. These safeguards preserve differentiability but do not prove global optimization or adequate expressiveness.")
    add_body(document, "For particle-like radiation, a dense ordered vector is a convenience rather than a symmetry principle. A future architecture should use permutation-equivariant conditioners, set coupling transforms, or mode-aware attention while preserving a tractable determinant. Conservation laws can enter through constrained coordinates or volume-preserving blocks, but volume preservation also fixes entropy and may be too restrictive when detector coarse-graining changes effective volume.")
    add_page_break(document)

    # Page 11
    add_page_heading(document, "Variational and Likelihood Objectives")
    add_body(document, "When independent samples from the physical telemetry law p_t are available, maximum likelihood minimizes cross-entropy. Up to the fixed entropy of p_t, this is the forward divergence KL(p_t||q_theta), which strongly penalizes failure to cover observed modes.")
    add_equation(document, "L_MLE(θ) = -E_(t∼π) E_(x∼p_t) [log q_θ(x|c_t)]", 15)
    add_body(document, "When p_t is available through a normalized or unnormalized energy, sampling from q_theta gives the reverse-KL variational free energy. This objective can be mode seeking, so a confirmatory study should compare both directions rather than treating one scalar loss as sufficient.")
    add_equation(document, "F(θ) = E_(t∼π) E_(x∼q_θ) [log q_θ(x|c_t)-log p_t(x)]", 16)
    add_body(document, "The paired alignment term anchors inverse coordinates to the accessible latent y_t. It is not required for density validity; it is required for the intended decoder interpretation. The full objective weights time snapshots and separates all terms in reporting.")
    add_equation(document, "L_total = L_MLE + λ_align E||f_θ^{-1}(x;c_t)-y_t||²", 17)
    add_table(document, "table_2_objectives.csv", 2, "Objectives and Their Identifying Content", [2200, 3160, 3760], "No single objective identifies both the outcome density and a physically named latent coordinate without additional assumptions.")
    add_page_break(document)

    # Page 12
    add_page_heading(document, "Entropy Estimation Across the Evaporation Timeline")
    add_body(document, "At each condition c_t, the flow draws z_i, computes x_i=f_theta(z_i;c_t), and evaluates log q_theta(x_i|c_t) using the base density and accumulated Jacobian. The estimator in Equation 18 is unbiased for h(q_theta) when samples are independent and the expectation exists. Its estimated standard error is the sample standard deviation of negative log density divided by the square root of N.")
    add_equation(document, "ĥ_t = -(1/N)Σ_i log q_θ(x_i|c_t)", 18)
    add_equation(document, "MCSE(ĥ_t) = sd{-log q_θ(x_i|c_t)}/√N", 19)
    add_figure(document, "figure_3_three_curves.png", 3, "Three Non-Equivalent Time Trajectories", "The Page curve is analytic. Telemetry entropy and fidelity are deterministic browser/report surrogates derived from the stated toy model; they are not fitted physical data.", width=5.7, alt="Three stacked line plots showing Page entropy, telemetry differential entropy, and toy decoder fidelity across normalized evaporation time.")
    add_body(document, "The figure shows why shape matching is insufficient. A conditional flow can generate a peaked classical entropy trajectory by architecture or target design, but that visual resemblance would not establish unitarity. The quantum curve, classical curve, and recovery curve must be validated against their own estimands and uncertainty statements.")
    add_page_break(document)

    # Page 13
    add_page_heading(document, "Reference Experiment and Architecture")
    add_body(document, "The baseline uses eight real telemetry coordinates, interpreted as four complex amplitudes only for the paired toy-state fidelity calculation. A frozen four-block conditional scrambler generates the reference density. The learned model contains eight conditional coupling blocks, 128-unit SiLU conditioners, time-mass context embeddings, and bounded log-scales. All default calculations use float64 arithmetic and explicit generators.")
    add_table(document, "table_3_architecture.csv", 3, "Baseline Conditional-Flow Architecture", [2250, 3250, 3620], "The architecture is intentionally modest enough for CPU verification while retaining the same interfaces needed for larger experiments.")
    add_body(document, "The remaining mass follows a stopped semiclassical schedule M(t)=max(M_floor,M_0(1-t)^(1/3)). This covariate supplies a physically motivated monotone condition but is not used at M=0, where the semiclassical law is singular. The reference channel is synthetic and exactly evaluable. Consequently, forward and reverse KL can both be estimated, making approximation error auditable in a way unavailable for an implicit simulator.")
    add_body(document, "A publication run should freeze configuration, software versions, device, seed family, evaluation grid, and all early-stopping rules before looking at the final curve. Multiple independent training seeds are required because neural optimization variance is a scientific uncertainty source, not merely an engineering nuisance.")
    add_page_break(document)

    # Page 14
    add_page_heading(document, "PyTorch Implementation and Autograd")
    add_body(document, "Every learnable transformation is an ordinary torch.nn.Module. The context encoder maps time and mass to a shared embedding. Each affine coupling conditioner consumes masked coordinates and context and returns scale and shift. Forward decoding accumulates log|det dx/dz|; reverse encoding traverses layers in reverse and accumulates log|det dz/dx|. Unit tests require both state round-trip accuracy and cancellation of the two log determinants.")
    add_body(document, "The training loop samples paired target batches, evaluates log q on data, adds the alignment penalty, calls loss.backward(), clips the global gradient norm, checks finiteness, and updates with AdamW. No event is detached from the learned computation graph. The target generator is frozen and sampled under no_grad because it defines data rather than an optimized adversary.")
    add_equation(document, "∇_θ L_total = autograd[L_MLE + λ_align L_align]", 20)
    add_body(document, "Production extensions should use automatic mixed precision only after a float64 reference is validated, because log determinants and small KL differences are sensitive to precision. Batch size, layer width, and number of time conditions should be scaled separately. Gradient accumulation can reduce memory without changing the estimand, whereas truncating dimensions changes the measurement model and must be documented as coarse-graining.")
    add_body(document, "The repository exposes a typed configuration hierarchy, command-line entry point, atomic JSON/CSV writers, CI across Python 3.11 and 3.12, and tests for quantum entropy, target normalization, exact inversion, and a differentiable smoke optimization. Generated results are excluded from source control until explicitly frozen for a release.")
    add_page_break(document)

    # Page 15
    add_page_heading(document, "Information Decryption and Pure-State Fidelity")
    add_body(document, "For paired synthetic data, the learned inverse produces z_hat=f_theta^{-1}(x;c_t). The eight real coordinates are split into real and imaginary parts and normalized to a four-dimensional complex vector. Fidelity with the normalized secret is then computed by Equation 21.")
    add_equation(document, "|ψ̂⟩ = normalize(ẑ_1:4 + i ẑ_5:8)", 21)
    add_equation(document, "F_t = |⟨ψ_t|ψ̂_t⟩|²", 22)
    add_body(document, "At zero visibility, secret and accessible latent are independent, so expected fidelity is the random-overlap baseline 1/d for complex dimension d. At high visibility, a correctly aligned inverse approaches the secret up to residual nuisance and approximation error. The analytic surrogate used in browser and report figures is 1/d+(1-1/d)v(t)^2; it is a design expectation, not a trained result.")
    add_body(document, "A physical Hayden-Preskill decoder would act on quantum subsystems and may require coherent access to early radiation, a reference system, the scrambling unitary, and a quantum computation of formidable complexity. The toy fidelity instead tests whether the statistical inverse coordinate tracks a known paired vector. It is useful because it can fail for identifiable reasons, but it must be labeled as a proxy.")
    add_body(document, "The decoder is evaluated on held-out secrets and times, compared with an unconditioned flow and a no-alignment ablation, and summarized with seed-level confidence intervals. High likelihood with baseline fidelity is evidence of latent non-identifiability rather than successful decryption.")
    add_page_break(document)

    # Page 16
    add_page_heading(document, "Latent Non-Identifiability: A Formal Obstruction")
    add_body(document, "Suppose q_theta fits the telemetry distribution exactly and its base is N(0,I). For every orthogonal matrix O, the transformed latent z'=Oz has the same base density and zero log-volume change. Composing the decoder with O therefore produces another flow with identical likelihood. Unless the secret definition is also invariant to O, density observations alone cannot identify which inverse coordinate is the secret.")
    add_equation(document, "φ(Oz)=φ(z),    |det O|=1,    OᵀO=I", 23)
    add_equation(document, "q_θ(x)=q_(θ,O)(x) but decode_θ(x) ≠ decode_(θ,O)(x)", 24)
    add_body(document, "This proposition disproves the strongest form of the claim that invertibility automatically decrypts the source. An invertible flow guarantees a one-to-one relationship between its own latent and observation coordinates. It does not guarantee that the learned latent equals a privileged physical coordinate. Paired anchors, a known forward operator, symmetry-breaking priors, independent interventions, or a supervised decoder readout are required.")
    add_body(document, "The alignment penalty in Equation 17 selects one representative from the likelihood-equivalent orbit. Its weight creates a bias-variance trade-off: too little leaves orientation unstable, while too much can distort density fitting when the assumed accessible latent is misspecified. Sensitivity analysis should therefore vary both alignment strength and number of paired anchors, with held-out likelihood and fidelity reported jointly.")
    add_page_break(document)

    # Page 17
    add_page_heading(document, "Adversarial and Decryptability Analysis")
    add_body(document, "Strong scrambling produces statistical challenges beyond ordinary overfitting. Reverse KL can concentrate on a subset of modes; coupling layers can develop stiff Jacobians; conditioning can average incompatible time slices; and an incomplete measurement design can leave quantum phases unidentified even when classical likelihood is excellent. Each threat has a distinct observable failure signature.")
    add_table(document, "table_4_adversarial.csv", 4, "Adversarial Failure Modes and Mitigations", [2140, 3340, 3640], "Mitigations must be validated by ablation; none converts incomplete quantum observations into complete information.")
    add_body(document, "Permutation invariance is appropriate when detector records are unordered particle sets, while autoregressive order is appropriate when arrival time carries meaning. Conservation-aware transforms can preserve charge or energy summaries. Volume-preserving transforms are attractive for unitary analogies because their determinant is one, but they force the model's differential entropy to equal the base entropy. That restriction can be scientifically wrong when the detector map includes attenuation, noise, or coarse-graining.")
    add_body(document, "The recommended architecture is therefore hybrid: equivariant conditioners for exchangeable particles, causal temporal context for ordered emissions, bounded non-volume-preserving couplings for detector effects, and explicit conserved-feature channels. The network's symmetry should match the measurement process, not a metaphorical resemblance to quantum unitarity.")
    add_page_break(document)

    # Page 18
    add_page_heading(document, "Computational Complexity and High-Dimensional Scaling")
    add_body(document, "For L affine coupling blocks, D telemetry coordinates, and conditioner width H, a dense implementation costs approximately O[L(DH+H^2)] multiply-adds per observation and O(LH^2) parameters when H dominates D. Exact log determinants remain O(LD), which is the main advantage over unrestricted invertible maps. Memory grows with activation storage across depth during reverse-mode differentiation.")
    add_figure(document, "figure_5_scaling.png", 5, "Illustrative Scaling of Dense and Structured Conditioners", "Counts are deterministic architecture proxies for L=8 and H=128, not wall-clock benchmarks. Structured conditioners remove the repeated dense H-squared term in this illustration.", width=5.9, alt="Log-log line chart comparing increasing operation counts for dense and structured coupling conditioners as telemetry dimension grows.")
    add_body(document, "Scaling to thousands of particles requires structure rather than only hardware. Set-equivariant blocks, sparse neighborhoods, multiscale factorizations, checkpointing, and sharded batches reduce cost. A fixed-dimensional flow also requires a representation for variable particle number, such as padded sets with masks, point-process likelihoods, or separate count and mark models. Any such change alters the statistical sample space and must be reflected in the Jacobian and likelihood.")
    add_page_break(document)

    # Page 19
    add_page_heading(document, "Reference Findings, Decision Rules, and Conclusions")
    add_body(document, "The reproducibility script provides analytic and seeded design benchmarks. For N=10 qubits, Page's exact expectation peaks at normalized time 0.5 with 2.966 nats. The four-dimensional complex-state fidelity surrogate begins at the random-overlap floor 0.250, equals 0.438 at the visibility midpoint, and reaches 0.993 by t=0.9. The eight-dimensional telemetry-entropy surrogate ranges from 11.274 to 12.065 nats and does not share the Page shape. These numbers validate calculations and labels; they are not outputs of a completed physical radiation experiment.")
    add_figure(document, "figure_6_sensitivity.png", 6, "Sensitivity of Post-Page Fidelity to Visibility Width and Paired Anchors", "The deterministic surface combines the stated visibility law with an illustrative anchor-saturation function. It identifies design directions for the PyTorch experiment.", width=5.45, alt="Heatmap showing higher toy decoder fidelity with more paired alignment anchors and sharper visibility transitions.")
    add_table(document, "table_5_decisions.csv", 5, "Preregistered Claim-to-Metric Decision Rules", [2180, 2980, 3960], "A Page-curve claim requires a density-operator estimator; telemetry entropy alone is explicitly insufficient.")
    add_body(document, "Accordingly, this is a rigorous learnability framework, not a numerical solution of the information paradox. Its claims require explicit measurement and identifiability assumptions; extensions should add informationally complete data, positive density-operator estimation, seed replication, and quantum-state baselines.")
    add_page_break(document)

    references = [
        "Almheiri, A., Engelhardt, N., Marolf, D., & Maxfield, H. (2019). The entropy of bulk quantum fields and the entanglement wedge of an evaporating black hole. Journal of High Energy Physics, 2019(12), 63. https://doi.org/10.1007/JHEP12(2019)063",
        "Almheiri, A., Hartman, T., Maldacena, J., Shaghoulian, E., & Tajdini, A. (2020). Replica wormholes and the entropy of Hawking radiation. Journal of High Energy Physics, 2020(5), 13. https://doi.org/10.1007/JHEP05(2020)013",
        "Bekenstein, J. D. (1973). Black holes and entropy. Physical Review D, 7(8), 2333-2346. https://doi.org/10.1103/PhysRevD.7.2333",
        "Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.",
        "Bousso, R. (2002). The holographic principle. Reviews of Modern Physics, 74(3), 825-874. https://doi.org/10.1103/RevModPhys.74.825",
        "Carleo, G., & Troyer, M. (2017). Solving the quantum many-body problem with artificial neural networks. Science, 355(6325), 602-606. https://doi.org/10.1126/science.aag2302",
        "Cover, T. M., & Thomas, J. A. (2006). Elements of information theory (2nd ed.). Wiley.",
        "Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017). Density estimation using Real NVP. International Conference on Learning Representations. https://arxiv.org/abs/1605.08803",
        "Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019). Neural spline flows. Advances in Neural Information Processing Systems, 32.",
        "Foong, S. K., & Kanno, S. (1994). Proof of Page's conjecture on the average entropy of a subsystem. Physical Review Letters, 72(8), 1148-1151. https://doi.org/10.1103/PhysRevLett.72.1148",
        "Harlow, D. (2016). Jerusalem lectures on black holes and quantum information. Reviews of Modern Physics, 88(1), 015002. https://doi.org/10.1103/RevModPhys.88.015002",
        "Hawking, S. W. (1975). Particle creation by black holes. Communications in Mathematical Physics, 43, 199-220. https://doi.org/10.1007/BF02345020",
        "Hayden, P., & Preskill, J. (2007). Black holes as mirrors: Quantum information in random subsystems. Journal of High Energy Physics, 2007(09), 120. https://doi.org/10.1088/1126-6708/2007/09/120",
        "Hosur, P., Qi, X.-L., Roberts, D. A., & Yoshida, B. (2016). Chaos in quantum channels. Journal of High Energy Physics, 2016(2), 4. https://doi.org/10.1007/JHEP02(2016)004",
        "Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. International Conference on Learning Representations. https://arxiv.org/abs/1312.6114",
        "Kobyzev, I., Prince, S. J. D., & Brubaker, M. A. (2021). Normalizing flows: An introduction and review of current methods. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(11), 3964-3979. https://doi.org/10.1109/TPAMI.2020.2992934",
        "Lubkin, E. (1978). Entropy of an n-system from its correlation with a k-reservoir. Journal of Mathematical Physics, 19(5), 1028-1031. https://doi.org/10.1063/1.523763",
        "Nielsen, M. A., & Chuang, I. L. (2010). Quantum computation and quantum information (10th anniversary ed.). Cambridge University Press.",
        "Page, D. N. (1993). Average entropy of a subsystem. Physical Review Letters, 71(9), 1291-1294. https://doi.org/10.1103/PhysRevLett.71.1291",
        "Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., & Lakshminarayanan, B. (2021). Normalizing flows for probabilistic modeling and inference. Journal of Machine Learning Research, 22(57), 1-64.",
        "Papamakarios, G., Pavlakou, T., & Murray, I. (2017). Masked autoregressive flow for density estimation. Advances in Neural Information Processing Systems, 30, 2338-2347.",
        "Penington, G. (2020). Entanglement wedge reconstruction and the information paradox. Journal of High Energy Physics, 2020(9), 2. https://doi.org/10.1007/JHEP09(2020)002",
        "Penington, G., Shenker, S. H., Stanford, D., & Yang, Z. (2022). Replica wormholes and the black hole interior. Journal of High Energy Physics, 2022(3), 205. https://doi.org/10.1007/JHEP03(2022)205",
        "Rezende, D. J., & Mohamed, S. (2015). Variational inference with normalizing flows. Proceedings of Machine Learning Research, 37, 1530-1538.",
        "Roberts, D. A., & Yoshida, B. (2017). Chaos and complexity by design. Journal of High Energy Physics, 2017(4), 121. https://doi.org/10.1007/JHEP04(2017)121",
        "Ryu, S., & Takayanagi, T. (2006). Holographic derivation of entanglement entropy from AdS/CFT. Physical Review Letters, 96(18), 181602. https://doi.org/10.1103/PhysRevLett.96.181602",
        "Susskind, L., Thorlacius, L., & Uglum, J. (1993). The stretched horizon and black hole complementarity. Physical Review D, 48(8), 3743-3761. https://doi.org/10.1103/PhysRevD.48.3743",
        "Torlai, G., Mazzola, G., Carrasquilla, J., Troyer, M., Melko, R. G., & Carleo, G. (2018). Neural-network quantum state tomography. Nature Physics, 14, 447-450. https://doi.org/10.1038/s41567-018-0048-5",
        "Torlai, G., & Melko, R. G. (2018). Latent space purification via neural density operators. Physical Review Letters, 120(24), 240503. https://doi.org/10.1103/PhysRevLett.120.240503",
        "van Erven, T., & Harremoes, P. (2014). Renyi divergence and Kullback-Leibler divergence. IEEE Transactions on Information Theory, 60(7), 3797-3820. https://doi.org/10.1109/TIT.2014.2320500",
    ]
    # References flow continuously; the section and appendix retain hard bounds.
    add_page_heading(document, "References")
    for reference in references:
        add_reference(document, reference)
    add_page_break(document)

    # Pages 23-25 - full figure/table reproducibility code
    code_lines = (ROOT / "reproduce_report_figures.py").read_text(encoding="utf-8").splitlines()
    chunk = (len(code_lines) + 2) // 3
    chunks = [code_lines[0:chunk], code_lines[chunk:2 * chunk], code_lines[2 * chunk:]]
    for index, lines in enumerate(chunks, start=1):
        add_code_page(document, "Appendix A: Figure and Table Reproduction Code" + (" (continued)" if index > 1 else ""), lines)
        if index < 3:
            add_page_break(document)

    document.save(DOCX_PATH)
    print(DOCX_PATH)


if __name__ == "__main__":
    build()
