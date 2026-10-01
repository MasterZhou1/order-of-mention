"""Word sets, contexts and questions for every experiment.

A word set is a set of names and contents. A context states one fact per name in fixed sentence frames; its
arrangement orders the facts and its content assignment pairs names with contents. Both rotate cyclically:
person k holds content (k + assignment) mod n, and order of mention p holds person (p + arrangement) mod n,
so every person and every content occupies every order of mention. Each context is asked every question.
"""
import random
from dataclasses import dataclass, field

from . import vocab

INSTRUCTION = "Read the sentences. Answer the question with only the {kind}{case}, with no explanation."

# relation: (answer kind, lowercase answers, fact frame, question frame); {a} is the indefinite article.
RELATIONS = {
    "eats": ("food", True, "{N} eats the {X}.", "What does {N} eat?"),
    "carries": ("food", True, "{N} carries the {X}.", "What does {N} carry?"),
    "owns": ("animal", True, "{N} owns {a} {X}.", "Which animal does {N} own?"),
    "visited": ("city", False, "{N} visited {X}.", "Which city did {N} visit?"),
    "works as": ("job", True, "{N} works as {a} {X}.", "What does {N} work as?"),
    "plays the": ("instrument", True, "{N} plays the {X}.", "Which instrument does {N} play?"),
    "plays": ("sport", True, "{N} plays {X}.", "Which sport does {N} play?"),
}
SIX_RELATIONS = ("eats", "owns", "visited", "works as", "plays the", "plays")
CONTENT_POOLS = {"eats": vocab.FOODS, "carries": vocab.FOODS, "owns": vocab.PETS, "visited": vocab.CITIES,
                 "works as": vocab.JOBS, "plays the": vocab.INSTRUMENTS, "plays": vocab.SPORTS}


def instruction(relation="eats"):
    kind, lower = RELATIONS[relation][:2]
    return INSTRUCTION.format(kind=kind, case=", in lowercase" if lower else "")


def article(word):
    return "an" if word[:1].lower() in "aeiou" else "a"


@dataclass
class Query:
    text: str
    answer: str
    position: int                       # order of mention p_C(q) of the queried fact (0-based)
    labels: dict = field(default_factory=dict)


@dataclass
class Context:
    instruction: str
    facts: str                          # empty for inputs without facts
    queries: list
    word_set: str
    candidates: list                    # every content the context states (for scoring)
    meta: dict = field(default_factory=dict)

    def body(self, q):
        parts = [self.instruction] + ([self.facts] if self.facts else []) + [self.queries[q].text]
        return "\n\n".join(parts)


@dataclass
class WordSet:
    id: str
    split: str
    names: list
    contents: list


# ---------------------------------------------------------------- word sets
def draw_word_sets(counts, n_people, seed, contents=None, n_contents=None, full_names=False,
                   multiword=False, exclude=()):
    """Word sets for one experiment; no complete name or content string repeats across its word sets.

    counts: {split: number of word sets}, e.g. {"calibration": 16, "test": 64}; calibration comes first. Names
    alternate the feminine and masculine rosters (rotated by one in odd word sets); contents within a set have
    distinct first words.
    `exclude` removes names and contents (e.g. those of another experiment's word sets).
    Returns (word sets, ordered content reserve for first-token qualification).
    """
    rng = random.Random(seed)
    rosters = [[n for n in r if n not in exclude] for r in (vocab.FEMININE, vocab.MASCULINE)]
    pools = [list(r) for r in rosters]
    for names in pools:
        rng.shuffle(names)
    calibration_names = set()

    def take(c, current):
        if not pools[c]:
            # Eight-person lists need more names than the rosters hold: test sets then reuse names.
            pools[c] = [n for n in rosters[c] if n not in calibration_names and n not in current]
            rng.shuffle(pools[c])
        return pools[c].pop()

    surnames = list(vocab.SURNAMES)
    rng.shuffle(surnames)
    pool = [c for c in (contents or vocab.FOODS) if c not in exclude and (" " in c or not multiword)]
    items, calibration_contents = list(pool), set()
    rng.shuffle(items)
    n_contents = n_contents or n_people
    sets, pointer, index = [], 0, 0
    for split, count in counts.items():
        for i in range(count):
            names = []
            for k in range(n_people):
                first = take((k + index) % 2, [n.split()[0] for n in names])
                if full_names:
                    used = {name.split()[-1] for name in names}
                    while surnames[pointer % len(surnames)] in used:
                        pointer += 1
                    first = f"{first} {surnames[pointer % len(surnames)]}"
                    pointer += 1
                names.append(first)
            chosen, deferred = [], []
            while len(chosen) < n_contents:
                if not items:
                    # Small pools (sports, instruments): later sets reuse contents, never calibration ones.
                    items = [c for c in pool if c not in calibration_contents and c not in chosen + deferred]
                    rng.shuffle(items)
                item = items.pop(0)
                clash = item.split()[0].lower() in {c.split()[0].lower() for c in chosen}
                (deferred if clash else chosen).append(item)
            items[:0] = deferred
            if split == "calibration":
                calibration_contents.update(chosen)
                calibration_names.update(n.split()[0] for n in names)
            sets.append(WordSet(f"{split[0]}{i:02d}", split, names, chosen))
            index += 1
    return sets, items


def qualify(word_sets, reserve, first_token, field="contents"):
    """Make the answers of every word set have pairwise distinct first tokens under the model's tokenizer.

    A colliding answer is replaced by the first reserve item whose first token and first word are new to the set.
    """
    for ws in word_sets:
        items = getattr(ws, field)
        for slot in range(len(items)):
            if first_token(items[slot]) not in {first_token(c) for c in items[:slot]}:
                continue
            rest = [c for j, c in enumerate(items) if j != slot]
            tokens, words = {first_token(c) for c in rest}, {c.split()[0].lower() for c in rest}
            choice = next(c for c in reserve if first_token(c) not in tokens and c.split()[0].lower() not in words)
            reserve.remove(choice)
            items[slot] = choice
    return word_sets


# ---------------------------------------------------------------- single-relation lists (Secs. 3-6)
def fact_list(ws, assignment, arrangement, n=None, relation="eats", order=None):
    """n facts under one relation; `order` (slots by order of mention) overrides the cyclic arrangement."""
    n = n or len(ws.names)
    names, contents = ws.names[:n], ws.contents[:n]
    order = list(order) if order is not None else [(p + arrangement) % n for p in range(n)]
    fact, question = RELATIONS[relation][2:]
    held = [contents[(k + assignment) % n] for k in range(n)]
    sentences = [fact.format(N=names[k], X=held[k], a=article(held[k])) for k in order]
    queries = [Query(question.format(N=names[k]), held[k], order.index(k), dict(person=k, content=(k + assignment) % n))
               for k in range(n)]
    return Context(instruction(relation), "\n".join(sentences), queries, ws.id, held,
                   dict(assignment=assignment, arrangement=arrangement, order=order, relation=relation))


def prefix(ctx, k):
    """The same context cut after its first k facts, asked about each of its k facts (App. E.3)."""
    lines = ctx.facts.split("\n")[:k]
    queries = [q for q in ctx.queries if q.position < k]
    return Context(ctx.instruction, "\n".join(lines), queries, ctx.word_set, [q.answer for q in queries],
                   dict(ctx.meta, prefix=k))


# ---------------------------------------------------------------- two people x two relations (App. D.2, E.4)
PERSON_RELATION_LAYOUTS = {"person_major": (0, 1, 2, 3), "person_major_reverse": (3, 2, 1, 0),
                           "relation_major": (0, 2, 1, 3), "relation_major_reverse": (3, 1, 2, 0)}


def person_relation(ws, assignment, layout):
    """Cell q = (person q // 2, relation q % 2 in {eats, carries}); layout lists cells by order of mention."""
    cells = [(q // 2, ("eats", "carries")[q % 2]) for q in range(4)]
    held = [ws.contents[(q + assignment) % 4] for q in range(4)]
    sentences = [RELATIONS[cells[q][1]][2].format(N=ws.names[cells[q][0]], X=held[q]) for q in layout]
    queries = [Query(RELATIONS[r][3].format(N=ws.names[e]), held[q], list(layout).index(q),
                     dict(cell=q, person=e, relation=r)) for q, (e, r) in enumerate(cells)]
    return Context(instruction(), "\n".join(sentences), queries, ws.id, held,
                   dict(assignment=assignment, layout=tuple(layout)))


# ---------------------------------------------------------------- two facts, active and passive (App. B)
def two_fact(i, bank, form, assignment, arrangement):
    """Word set i: person k eats foods[k XOR assignment] of content set `bank` (two contents per set)."""
    names, foods = vocab.TWO_FACT_NAMES[i], vocab.TWO_FACT_FOODS[i][2 * bank:2 * bank + 2]
    held = [foods[k ^ assignment] for k in range(2)]
    frame = "{N} eats the {X}." if form == "active" else "The {X} is eaten by {N}."
    order = [arrangement, 1 - arrangement]
    queries = [Query(f"What does {names[k]} eat?", held[k], order.index(k), dict(person=k)) for k in range(2)]
    return Context(instruction(), "\n".join(frame.format(N=names[k], X=held[k]) for k in order), queries,
                   f"w{i:02d}", held, dict(form=form, assignment=assignment, arrangement=arrangement))


# ---------------------------------------------------------------- coordinated subjects (App. C.2)
def coordinated(ws, c, b):
    """Five people, four facts; the fact at order of mention c has two subjects.

    Name order m holds person (m + c + b) mod 5; fact k states content (k + b) mod 4; fact(m) = m if m <= c else m - 1.
    """
    people = [ws.names[(m + c + b) % 5] for m in range(5)]
    foods = [ws.contents[(k + b) % 4] for k in range(4)]
    fact_of = [m if m <= c else m - 1 for m in range(5)]
    sentences = []
    for k in range(4):
        subjects = [people[m] for m in range(5) if fact_of[m] == k]
        sentences.append(f"{' and '.join(subjects)} {'eat' if len(subjects) > 1 else 'eats'} the {foods[k]}.")
    queries = [Query(f"What does {people[m]} eat?", foods[fact_of[m]], fact_of[m], dict(mention=m, fact=fact_of[m]))
               for m in range(5)]
    return Context(instruction(), "\n".join(sentences), queries, ws.id, foods, dict(coordinated=c, assignment=b))


# ---------------------------------------------------------------- two-place relations (App. C.3)
TWO_PLACE = {"loves": ("love", "loved"), "trusts": ("trust", "trusted"), "helps": ("help", "helped"),
             "visits": ("visit", "visited"), "pays": ("pay", "paid")}
PERSON_INSTRUCTION = "Read the sentences. Answer the question with only the person's name, with no explanation."


def two_place(ws, verb, order, binding, voice="active"):
    """Three facts among six people, pairs (0,1), (2,3), (4,5); binding 1 makes the second of a pair the agent.

    Questions (always in the active voice) ask for the patient ("Whom does A love?", slot "o") or for the agent
    ("Who loves B?", slot "s").
    """
    bare, participle = TWO_PLACE[verb]
    pairs = [(ws.names[2 * i + binding], ws.names[2 * i + 1 - binding]) for i in range(3)]
    order_ = [(p + order) % 3 for p in range(3)]
    frame = "{A} {v} {P}." if voice == "active" else "{P} is {pp} by {A}."
    sentences = [frame.format(A=pairs[i][0], P=pairs[i][1], v=verb, pp=participle) for i in order_]
    queries = []
    for i, (agent, patient) in enumerate(pairs):
        p = order_.index(i)
        queries.append(Query(f"Whom does {agent} {bare}?", patient, p, dict(fact=p, slot="o", person=ws.names.index(patient))))
        queries.append(Query(f"Who {verb} {patient}?", agent, p, dict(fact=p, slot="s", person=ws.names.index(agent))))
    return Context(PERSON_INSTRUCTION, "\n".join(sentences), queries, ws.id, list(ws.names),
                   dict(verb=verb, voice=voice, order=order, binding=binding))


# ---------------------------------------------------------------- facts stated in prose (App. G.3)
def prose(ws, assignment, arrangement, seed):
    """One paragraph: four facts in four different verb frames, 2-3 unrelated sentences between facts, and an
    opening and a closing sentence each with probability 1/2. Sentences are joined by single spaces."""
    ctx = fact_list(ws, assignment, arrangement)
    rng = random.Random(f"{seed}|{ws.id}|b{assignment}|o{arrangement}")
    frames = rng.sample(range(len(vocab.PROSE_FRAMES)), 4)
    count = rng.choice((2, 3))
    gaps = sorted(rng.sample(range(3), count))
    fillers = rng.sample(range(len(vocab.PROSE_FILLERS)), count)
    opening = rng.randrange(len(vocab.PROSE_OPENINGS)) if rng.random() < .5 else None
    closing = rng.randrange(len(vocab.PROSE_CLOSINGS)) if rng.random() < .5 else None
    held = {q.position: (ws.names[q.labels["person"]], q.answer) for q in ctx.queries}
    parts = [vocab.PROSE_OPENINGS[opening]] if opening is not None else []
    for p in range(4):
        parts.append(vocab.PROSE_FRAMES[frames[p]].format(N=held[p][0], F=held[p][1]))
        if p in gaps:
            parts.append(vocab.PROSE_FILLERS[fillers[gaps.index(p)]])
    if closing is not None:
        parts.append(vocab.PROSE_CLOSINGS[closing])
    ctx.facts = " ".join(parts)
    ctx.meta.update(format="prose", frames=frames, gaps=gaps)
    return ctx


# ---------------------------------------------------------------- order of mention vs token distance (App. D.4)
PAD = "After a long and quiet walk along the old river path on a cool and windy autumn afternoon,"
FILLERS = ["It was a sunny day.", "The streets were calm.", "A light breeze moved through the trees.",
           "Birds sang in the distance."]
CAL_PADS = ["Before the long meeting in the crowded hall downtown had even started that grey morning,",
            "Having finished a slow and careful tour of the empty museum late on Sunday evening,"]
CAL_FILLERS = ["The sky was grey.", "It rained a little in the morning.", "The road was quiet.",
               "A clock ticked on the wall.", "The wind was cold."]
# name: (lengthened orders of mention, filler lines after facts 1-3, filler lines before the question)
LENGTH_CONDITIONS = {"uniform": ((), (0, 0, 0), 0), "pad1": ((0,), (0, 0, 0), 0), "pad2": ((1,), (0, 0, 0), 0),
                     "pad3": ((2,), (0, 0, 0), 0), "pad4": ((3,), (0, 0, 0), 0), "pad2+4": ((1, 3), (0, 0, 0), 0),
                     "filler_lines": ((), (1, 0, 3), 0), "filler_block": ((), (0, 0, 0), 4)}


def lengthened(ws, assignment, arrangement, condition, rng=None):
    """A four-fact list whose facts differ in token length; condition "mixed" draws a random profile with rng."""
    ctx = fact_list(ws, assignment, arrangement)
    facts = ctx.facts.split("\n")
    if condition == "mixed":
        pads = [rng.choice(CAL_PADS) if rng.random() < .4 else None for _ in range(4)]
        after = [rng.choices((0, 1, 2), (.6, .25, .15))[0] for _ in range(3)]
        block = rng.choices((0, 1, 2, 3), (.55, .15, .15, .15))[0]
        filler = lambda: rng.choice(CAL_FILLERS)  # noqa: E731
    else:
        padded, after, block = LENGTH_CONDITIONS[condition]
        pads = [PAD if p in padded else None for p in range(4)]
        stream = iter(FILLERS * 4)
        filler = lambda: next(stream)  # noqa: E731
    lines = []
    for p, sentence in enumerate(facts):
        lines.append(f"{pads[p]} {sentence}" if pads[p] else sentence)
        lines += [filler() for _ in range(after[p] if p < 3 else 0)]
    lines += [filler() for _ in range(block)]
    ctx.facts = "\n".join(lines)
    ctx.meta.update(condition=condition)
    return ctx


# ---------------------------------------------------------------- inputs without facts (Table 1B, App. D.2)
def no_facts(ctx, names=None):
    """The context's questions without its facts: the question alone, or after "Names, in order: ..."."""
    facts = "Names, in order: " + ", ".join(names) + "." if names else ""
    return Context(ctx.instruction, facts, ctx.queries, ctx.word_set, [], dict(ctx.meta, no_facts=True))
