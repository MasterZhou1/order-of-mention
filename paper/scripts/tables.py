"""LaTeX for the paper's seven tables, built from paper/data (see paper/make_tables.py)."""
import numpy as np

from common import load

MAIN3 = (("qwen7b", "Qwen"), ("gemma4b", "Gemma"), ("llama8b", "Llama"))


def half(m):
    """Half of the 95% interval, in points."""
    lo, hi = m["ci"]
    return 100 * (hi - lo) / 2


def pm(h):
    return rf"{{\,\tiny$\pm${h:.1f}}}"


def rounded_half(m):
    """Half of the 95% interval after rounding its endpoints to 0.1 points (Tables 1 and 3)."""
    lo, hi = (round(100 * x, 1) for x in m["ci"])
    return (hi - lo) / 2


# ------------------------------------------------------------------ Table 1
def selection_evidence():
    T = load("transfer.json")["models"]

    def cell(m, bold=False, shade=True):
        v = 100 * m["mean"]
        h = rounded_half(m)
        body = rf"\textbf{{{v:.1f}}}" if bold else f"{v:.1f}"
        body += pm(h) if h >= .05 else r"\phantom{\,\tiny$\pm$0.0}"
        return (rf"\cellcolor{{tabfill!{round(.45 * v)}}}" if shade else "") + body

    def row(label, get, **kw):
        return label + " & " + " & ".join(cell(get(T[k]), **kw) for k, _ in MAIN3) + r"\\"

    rows = [
        r"\multicolumn{4}{@{}l}{\textit{A.\ Transfer to new contexts}}\\",
        row(r"Ordinal vector $v$", lambda m: m["transfer"]["ordinal_vector"], bold=True),
        row(r"\refrow{Random}", lambda m: m["transfer"]["random_0"]),
        row(r"\refrow{Ceiling}", lambda m: m["transfer"]["ceiling"], shade=False),
        r"\addlinespace[2pt]",
        r"\multicolumn{4}{@{}l}{\textit{B.\ Where the vector comes from}}\\",
        row(r"With facts ($v$)", lambda m: m["no_facts"]["with_facts"], bold=True),
        row(r"Question alone", lambda m: m["no_facts"]["question_alone"]),
        row(r"Names listed, same order", lambda m: m["no_facts"]["names_same_order"]),
        row(r"Names listed, shuffled", lambda m: m["no_facts"]["names_shuffled"]),
        row(r"\refrow{Random}", lambda m: m["no_facts"]["random_0"]),
        row(r"\refrow{Ceiling}", lambda m: m["no_facts"]["ceiling"], shade=False),
        r"\addlinespace[2pt]",
        r"\multicolumn{4}{@{}l}{\textit{C.\ Whole state from another context}}\\",
        row(r"Donor asked another fact", lambda m: m["donor_states"]["donor_asked_other_fact"], bold=True),
        row(r"Donor asked the receiver's question", lambda m: m["donor_states"]["donor_asked_receiver_question"]),
    ]
    return r"""\ifdefined\isarxiv
\begin{table}[t]
\else
\begin{wraptable}{r}{0.59\linewidth}
\vspace{-10pt}
\fi
\centering\footnotesize
\definecolor{tabfill}{HTML}{548F9C}
\newcommand{\refrow}[1]{\textcolor{black!55}{#1}}
\setlength{\tabcolsep}{2pt}
\ifdefined\isarxiv\begin{minipage}[c]{0.40\linewidth}\fi
\caption{\textbf{An ordinal vector from other contexts selects the designated fact in a new context.} Designated answers (\%), question unchanged; $\pm$ is half the 95\% word-set bootstrap interval; shading is proportional to the rate. (A) 64 new word sets, 12,288 cases per model; Random: at most $0.01\%$. (B) 64 further word sets, 1,024 cases per row, all directions scaled to one length; the question and name rows are computed without the facts (\cref{app:transfer_controls}). Ceiling: the receiver's own state for the designated question; it differs between panels because the word sets differ. (C) The receiver's state replaced by a donor's; 16 donors $\times$ 64 receivers; designated: the receiver's fact at the order of mention of the other question.}
\ifdefined\isarxiv\label{tab:contextual_selection}\label{tab:wording_controls}\end{minipage}\hfill\begin{minipage}[c]{0.56\linewidth}\centering\else

\vspace{3pt}
\fi
\begin{tabular}{@{}lrrr}
\toprule
& Qwen & Gemma & Llama\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\ifdefined\isarxiv\end{minipage}\fi

\ifdefined\isarxiv
\end{table}
\else
\label{tab:contextual_selection}\label{tab:wording_controls}
\vspace{-8pt}
\end{wraptable}
\fi
"""


# ------------------------------------------------------------------ Table 2
def rearranged():
    T = load("transfer.json")["models"]
    rows = []
    for k, name in MAIN3:
        for change, label in (("person", "Person"), ("relation", "Relation")):
            r = T[k]["rearranged"][change]
            cells = []
            for arm in ("original_pair", "new_order_of_mention", "recalibrated", "ceiling"):
                v = f"{100 * r[arm]['mean']:.1f}"
                cells.append((rf"\textbf{{{v}}}" if arm == "new_order_of_mention" else v) + pm(half(r[arm])))
            cells += [f"{100 * r[arm]['mean']:.1f}" for arm in ("random", "no_facts")]
            rows.append(f"{name if change == 'person' else ''} & {label} & " + " & ".join(cells) + r"\\")
        rows.append(r"\addlinespace[2pt]")
    return r"""\begin{table}[!ht]
\centering\small
\caption{\textbf{When facts move, the ordinal vector must follow their orders of mention.} Designated answers (\%) under two held-out arrangements, 64 test word sets and 2,048 cases per entry; $\pm$ is half the 95\% word-set bootstrap interval; bold: the new-order-of-mention edit; ceiling: the receiver's own state for the designated question; no facts: the direction computed from the question alone.}
\label{tab:rearranged}
\vspace{4pt}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llrrrrrr}
\toprule
Model & Change & Original pair & \shortstack{New order\\of mention} & Recalibrated & Ceiling & Random & No facts\\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


# ------------------------------------------------------------------ Table 3
def composition():
    C = load("composition.json")["models"]

    def row(label, f):
        return label + " & " + " & ".join(f(C[k]) for k, _ in MAIN3) + r"\\"

    rows = [
        row(r"Joint change $v_J$", lambda m: rf"\textbf{{{100 * m['edits']['joint']['mean']:.1f}}}"),
        row(r"Same-start sum $v_E+v_R$", lambda m: f"{100 * m['edits']['same_start_sum']['mean']:.1f}"),
        row(r"Sum rescaled to $\|v_J\|$", lambda m: f"{100 * m['edits']['sum_rescaled']['mean']:.1f}"),
        row(r"Ceiling (own state for the joint question)", lambda m: f"{100 * m['edits']['ceiling']['mean']:.1f}"),
        r"\midrule",
        row(r"Joint minus sum", lambda m: f"{100 * m['paired']['joint_minus_sum']['mean']:.1f}"
            + pm(rounded_half(m["paired"]["joint_minus_sum"]))),
        row(r"Joint minus rescaled sum", lambda m: f"{100 * m['paired']['joint_minus_rescaled_sum']['mean']:.1f}"
            + pm(rounded_half(m["paired"]["joint_minus_rescaled_sum"]))),
    ]
    return r"""\begin{table}[!ht]
\centering\small
\caption{\textbf{The joint change selects the joint target more often than the sum of its parts.} Designated answers (\%), 64 test word sets, 4,096 cases per edit; $\pm$: half the 95\% paired word-set bootstrap interval, in points; bold: the joint change.}
\label{tab:composition_selection}
\vspace{4pt}
\begin{tabular}{lrrr}
\toprule
Edit & Qwen & Gemma & Llama\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


# ------------------------------------------------------------------ Table 4
def development():
    DEV = load("olmo2_checkpoints.json")["checkpoints"]
    order = [("step1000", "5B"), ("step4000", "17B"), ("step10000", "42B"), ("step25000", "105B"),
             ("step50000", "210B"), ("step99000", "416B"), ("step200000", "839B"), ("step400000", "1.7T"),
             ("step928646", "3.9T"), ("main", "final")]

    def pct(m, arm):
        a = m["at_block_16"]["by_content"].get(arm)
        return "--" if not a else f"{100 * a['designated']:.1f}"

    def num(v, fmt="{:.2f}"):
        return "--" if v is None else fmt.format(v).replace("-", "$-$")

    rows = []
    for key, name in order:
        m = DEV[key]
        inv = m["at_block_16"]["invariance"]
        rows.append(" & ".join([name, str(m["main_block"]), pct(m, "unedited"), rf"\textbf{{{pct(m, 'ordinal_vector')}}}",
                                pct(m, "random"), pct(m, "ceiling"), num(inv["new_names_cosine"]),
                                num(inv["reversed_order_cosine"]),
                                f"{num(m['peak_fact_address_r2'])} ({m['peak_fact_address_block']})"]) + r"\\")
    return r"""\begin{table}[!ht]
\centering\small
\caption{\textbf{OLMo-2-7B checkpoints.} Rule-chosen block; at block 16: unedited accuracy and designated answers (\%) for the ordinal vector (bold), a random direction and the ceiling, scored by the single stated content an answer names, 64 test word sets and 12,288 cases per edit; new-name and reversed-order cosines of \cref{prop:invariance}; and the peak leave-one-word-set-out fact-address $R^2$ over blocks (its block). \emph{final}: the released model after stage-2 mid-training.}
\label{tab:development}
\vspace{4pt}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
& & \multicolumn{6}{c}{At block 16} & \\
\cmidrule(lr){3-8}
& & \multicolumn{4}{c}{Scored by stated content (\%)} & \multicolumn{2}{c}{Cosine} & \\
\cmidrule(lr){3-6}\cmidrule(lr){7-8}
Tokens & \shortstack{Rule\\block} & Unedited & $v$ & Random & Ceiling & New names & Reversed & Peak $R^2$ (block)\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


# ------------------------------------------------------------------ Tables 5-7 helpers
def pct(m, bold=False, interval=False):
    """m: a rate, or {mean|designated, ci}."""
    if m is None:
        return "--"
    if isinstance(m, dict):
        v = 100 * (m["mean"] if "mean" in m else m["designated"])
    else:
        v = 100 * m
    body = f"{v:.1f}"
    if bold:
        body = rf"\textbf{{{body}}}"
    if not (interval and isinstance(m, dict)):
        return body
    h = half(m)
    return body if h < 0.05 else rf"{body}{{\,\tiny$\pm${h:.1f}}}"


def num(x, nd=2):
    return "--" if x is None else f"{x:.{nd}f}".replace("-", "$-$", 1)


def r2_at(curves, block):
    row = next((c for c in curves if c["block"] == block), None)
    return (row["fact_address_r2"], row["person_r2"]) if row else (None, None)


def window(curve, floor=0.10):
    """First to last block at which the ordinal vector selects at least 10% (every second block)."""
    hits = sorted(int(b) for b, v in curve.items()
                  if v is not None and (v["mean"] if isinstance(v, dict) else v) >= floor)
    return f"{hits[0]}--{hits[-1]}" if hits else "--"


NAMES = {"qwen1_5b": "Qwen2.5-1.5B", "qwen3b": "Qwen2.5-3B", "qwen14b": "Qwen2.5-14B", "qwen32b": "Qwen2.5-32B",
         "qwen3_8b": "Qwen3-8B"}


# ------------------------------------------------------------------ Table 5
def generality_scale():
    S = load("scale_sweep.json")
    rows = []
    for key in ["qwen1_5b", "qwen3b", "qwen7b", "qwen14b", "qwen32b", "qwen3_8b"]:
        if key == "qwen7b":
            r = S["qwen7b_reference"]
            pos, per = r2_at(r["geometry"], r["main_block"])
            a = r["transfer"]
            rows.append(rf'Qwen2.5-7B$^\dagger$ & {r["main_block"]}/{r["blocks"]} & {pct(a["unedited"])} & '
                        rf'{pct(a["ordinal_vector"], True)} & {pct(a["random_0"])} & '
                        rf'{pct(a["ceiling"])} & {num(r["invariance"]["new_names_cosine"])} & '
                        rf'{num(r["invariance"]["reversed_order_cosine"])} & {num(pos)} & '
                        rf'{num(per)} & {window(r["depth"]["ordinal_vector"])}\\')
            continue
        m = S["models"][key]
        pos, per = r2_at(m["geometry"], m["main_block"])
        t = m["transfer"]
        rows.append(rf'{NAMES[key]} & {m["main_block"]}/{m["blocks"]} & {pct(t["unedited_accuracy"])} & '
                    rf'{pct(t["ordinal_vector"]["designated"], True)} & {pct(t["random_0"]["designated"])} & '
                    rf'{pct(t["ceiling"]["designated"])} & '
                    rf'{num(m["invariance"]["new_names_cosine"])} & {num(m["invariance"]["reversed_order_cosine"])} & '
                    rf'{num(pos)} & {num(per)} & {window(m["depth"]["ordinal_vector"]) if "depth" in m else "--"}\\')
        if key == "qwen32b":
            rows.append(r"\addlinespace[2pt]")
    return r'''\begin{table}[h]
\centering\footnotesize
\setlength{\tabcolsep}{2.6pt}
\caption{\textbf{Ordinal addressing holds from 1.5B to 32B parameters.} Every model runs the transfer test of \cref{tab:contextual_selection}A (64 new word sets, 12,288 cases), the cosines of \cref{fig:relational_hypothesis}d, the fact-address and person templates of \cref{app:query_geometry}, and the depth curves of \cref{fig:selection_stages} at every second block. Main block: block index / number of blocks, chosen on block-selection word sets by the rule of \cref{app:selection_protocol}. Unedited: correct answers without an edit. Window: first to last block at which the ordinal vector selects at least $10\%$, as in \cref{fig:selection_stages} (8 donors $\times$ 32 receivers). $^\dagger$The main-text model; its block was chosen before this sweep (\cref{app:selection_protocol}), and its window comes from the every-block curves of \cref{fig:selection_stages}.}
\label{tab:scale}
\vspace{6pt}
\begin{tabular}{@{}lrrrrrrrrrr@{}}
\toprule
& & \multicolumn{4}{c}{Designated answers (\%)} & \multicolumn{2}{c}{Change cosine} & \multicolumn{2}{c}{$R^2$} & \\
\cmidrule(lr){3-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}
Model & \shortstack{Main\\block} & Unedited & $v$ & Random & Ceiling & New names & Reversed & \shortstack{Fact\\address} & Person & Window\\
\midrule
''' + "\n".join(rows) + r'''
\bottomrule
\end{tabular}

\end{table}
'''


# ------------------------------------------------------------------ Table 6
def generality_base():
    S = load("scale_sweep.json")
    label = {"qwen1_5b": "Qwen2.5-1.5B", "qwen3b": "Qwen2.5-3B", "qwen3_8b": "Qwen3-8B", "gemma4b_it": "Gemma-3-4B",
             "llama3b_it": "Llama-3.2-3B"}
    role_name = {"base": "base, plain", "instruct_plain": "instruct, plain", "instruct_chat": "instruct, chat"}
    rows = []
    for pair, keys in S["base_vs_instruct"].items():
        for i, role in enumerate(("base", "instruct_plain", "instruct_chat")):
            lead = label[pair] if i == 0 else ""
            m = S["models"].get(keys[role])
            if m is None and pair == "gemma4b_it" and role == "instruct_chat":
                # the main-text Gemma-3-4B-IT run: Table 1A transfer, Figure 2d cosines, App. C.1 R^2 at block 22
                g = S["gemma4b_it_chat_reference"]
                t = g["transfer"]
                inv = load("invariance.json")["models"]["gemma4b"]
                pos, _ = r2_at(load("query_geometry.json")["models"]["gemma4b"]["curves"], g["main_block"])
                rows.append(rf"{lead} & {role_name[role]}\rlap{{$^\ddagger$}} & {g['main_block']} & {pct(t['unedited'])} & "
                            rf"{pct(t['ordinal_vector'], True)} & {pct(t['random_0'])} & {pct(t['ceiling'])} & "
                            rf"{num(float(np.mean(inv['new_names_cosines'])))} & {num(inv['reversed_order_cosine'])} & "
                            rf"{num(pos)}\\")
                continue
            pos, _ = r2_at(m["geometry"], m["main_block"])
            t, inv = m["transfer"], m["invariance"]
            rows.append(rf"{lead} & {role_name[role]} & {m['main_block']} & {pct(t['unedited_accuracy'])} & "
                        rf"{pct(t['ordinal_vector']['designated'], True)} & {pct(t['random_0']['designated'])} & "
                        rf"{pct(t['ceiling']['designated'])} & "
                        rf"{num(inv['new_names_cosine'])} & {num(inv['reversed_order_cosine'])} & {num(pos)}\\")
            if "transfer_by_content" in m:
                b = m["transfer_by_content"]
                rows.append(rf" & \quad by content\rlap{{$^\ast$}} & & {pct(b['unedited'])} & {pct(b['ordinal_vector'], True)} & "
                            rf"{pct(b['random_0'])} & {pct(b['ceiling'])} & & & \\")
        rows.append(r"\addlinespace[2pt]")
    return r'''\begin{table}[h]
\centering\footnotesize
\setlength{\tabcolsep}{2.3pt}
\caption{\textbf{Base models address facts the same way.} Plain: the instruction, facts, and question followed by \texttt{Answer:}, with no chat template, identical for base and instruct weights; chat: the instruct model's native template. Each row chooses its own main block by the rule of \cref{app:selection_protocol} and runs the tests of \cref{tab:scale}. $^\ast$Gemma-3-4B base answers in sentences, so its answers are rescored by the single stated content they name. $^\ddagger$Main-text Gemma run (\cref{tab:contextual_selection}A, \cref{fig:relational_hypothesis}d, \cref{app:query_geometry}).}
\label{tab:base_instruct}
\vspace{6pt}
\begin{tabular}{@{}llrrrrrrrr@{}}
\toprule
& & & \multicolumn{4}{c}{Designated answers (\%)} & \multicolumn{2}{c}{Change cosine} & \\
\cmidrule(lr){4-7}\cmidrule(lr){8-9}
Model & Weights, prompt & Block & Unedited & $v$ & Random & Ceiling & New names & Reversed & \shortstack{Fact-\\address $R^2$}\\
\midrule
''' + "\n".join(rows[:-1]) + r'''
\bottomrule
\end{tabular}

\end{table}
'''


# ------------------------------------------------------------------ Table 7
def generality_prose():
    N = load("prose.json")["models"]
    rows = []
    for key, name in MAIN3:
        for fmt in ("prose", "list"):
            o = N[key]["receivers"][fmt]
            rows.append(rf'{name if fmt == "prose" else ""} & {fmt} & {pct(o["unedited"])} & '
                        rf'{pct(o["lists_main"], True, True)} & {pct(o["lists_new_word_sets"], False, True)} & '
                        rf'{pct(o["prose"], False, True)} & {pct(o["random_0"])} & {pct(o["ceiling"], False, True)}\\')
        rows.append(r"\addlinespace[2pt]")
    geo = []
    for key, name in MAIN3:
        g = N[key]["geometry"]
        geo.append(rf'{name} & {num(g["prose_fact_address_r2"])} & {num(g["prose_person_r2"])} & '
                   rf'{num(g["list_fact_address_r2"])} & {num(g["template_cosine_prose_list"])} & '
                   rf'{num(g["vector_cosine_prose_list"])} & {100 * g["prose_share_in_list_span"]:.0f} & '
                   rf'{100 * g["random_share"]:.1f}\\')
    return r'''\begin{table}[!ht]
\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{\textbf{Ordinal addressing carries over to prose.} Top: designated answers (\%) on 64 new word sets, 12,288 cases per entry, each edit added once at the main block. Receivers state the four facts either as a paragraph (\emph{prose}) or as the list of the main text, with the same people, contents, arrangements, and questions. The ordinal vectors are calibrated on the main-text lists (the $v$ of \cref{tab:contextual_selection}A), on lists of 16 new word sets, or on paragraphs of the same 16 word sets. Bottom: at the main block, leave-one-word-set-out $R^2$ of fact-address and person templates on paragraphs and on the matched lists (the list $R^2$ uses this experiment's word sets, not those of \cref{app:query_geometry}); mean cosine between matching prose and list fact-address templates and between matching prose and list ordinal vectors; fraction of the squared norm of query-dependent prose states inside the span of the main-text ordinal vectors, against random subspaces of the same dimension.}
\label{tab:prose}
\vspace{6pt}
\begin{tabular}{@{}llrrrrrr@{}}
\toprule
& & & \multicolumn{3}{c}{Ordinal vector calibrated on} & & \\
\cmidrule(lr){4-6}
Model & \shortstack{Receiver\\format} & Unedited & Lists (main) & Lists (new word sets) & Prose & Random & Ceiling\\
\midrule
''' + "\n".join(rows[:-1]) + r'''
\bottomrule
\end{tabular}

\vspace{6pt}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
& \multicolumn{2}{c}{Prose $R^2$} & List $R^2$ & \multicolumn{2}{c}{Prose vs.\ list} & \multicolumn{2}{c}{Squared norm in list span (\%)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
Model & \shortstack{Fact\\address} & Person & \shortstack{Fact\\address} & Template cos. & Vector cos. & Prose states & Random\\
\midrule
''' + "\n".join(geo) + r'''
\bottomrule
\end{tabular}

\end{table}
'''


def build_all():
    return {
        "selection-evidence-table.tex": selection_evidence(),
        "rearranged-table.tex": rearranged(),
        "composition-table.tex": composition(),
        "development-table.tex": development(),
        "generality-scale-table.tex": generality_scale(),
        "generality-base-table.tex": generality_base(),
        "generality-prose-table.tex": generality_prose(),
    }
