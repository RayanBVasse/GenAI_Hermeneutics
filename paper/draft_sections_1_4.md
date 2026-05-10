# The Canon Pack: Structured Authorial Intent for GenAI

**Rayan B. Vasse**

---

## Abstract

AI-powered reading tools — from Amazon's "Ask This Book" to Google's NotebookLM — increasingly mediate how readers engage with books. These systems rely on Retrieval-Augmented Generation (RAG): the text is chunked, embedded as vectors, and relevant passages are retrieved in response to reader queries. The implicit hermeneutic assumption is naive textualism: meaning resides in the text and can be extracted by pattern matching. This assumption ignores a century of hermeneutic theory on the constitutive role of interpretive context in meaning-making, and more practically, it strips authors of any governance over how AI systems represent their work — a concern raised by the Authors Guild in response to Amazon's Kindle AI features (Authors Guild 2025).

We introduce the Canon Pack: a structured representation of authorial interpretive intent comprising the author's thesis, chapter-level argumentative purposes, voice configuration, common misreadings to correct, boundary rules, and deliberate silences. The Canon Pack operationalizes E.D. Hirsch's distinction between *meaning* (authorial intent) and *significance* (reader application) as a computationally tractable object that governs AI companion behavior without foreclosing reader interpretation. We describe a pipeline that generates a draft Canon Pack from manuscript analysis and presents it to the author for review, and a companion system governed by the resulting interpretive framework.

We test the framework on five books spanning academic philosophy, cultural criticism, classical philosophy, military strategy, and literary nonfiction. In a controlled comparison where Canon Pack and standard RAG companions answer identical questions using the same retrieval and language model — differing only in system-level interpretive guidance — Canon Pack companions demonstrate substantially greater interpretive depth and voice consistency, while revealing a productive tension with textual grounding that raises design questions for AI reading tools.

---

## 1. Introduction

In December 2025, Amazon launched "Ask This Book," a feature that allows Kindle readers to query an AI chatbot about purchased books directly from the reading interface. The Authors Guild responded within days, noting that the feature was deployed without publisher or author permission, and that there exists no mechanism for authors to opt out, shape the AI's interpretive behavior, or correct its misrepresentations of their work (Authors Guild 2025). The response captured a growing anxiety in the literary world: AI systems are beginning to mediate the encounter between reader and text, and the author — the person whose interpretive intentions shaped the work — has been written out of the process entirely.

Amazon's feature is one instance of a broader pattern. Google's NotebookLM invites users to upload documents and "have a conversation" with them. Dozens of startups offer RAG-based "chat with your book" services. These tools share a common technical architecture: the text is segmented into chunks, each chunk is embedded as a vector in a high-dimensional space, and when a reader poses a question, the most semantically similar chunks are retrieved and passed to a large language model (LLM) alongside the query. The LLM generates a response grounded — to varying degrees — in the retrieved passages.

This architecture works adequately for factual queries: "Which chapter discusses epistemic humility?" or "What year was the Battle of Gaugamela?" But it fails at precisely the point where reading matters most — at the level of interpretation. When a reader asks, "Is Metzinger arguing that consciousness is an illusion?" or "Does Marcus Aurelius think emotions are inherently bad?", a RAG system has no access to what the author actually intended. It can only pattern-match against local passages, producing answers that may directly contradict the author's stated argumentative framework. The system cannot distinguish between what the text says and what the author means — because it has no representation of the latter.

### 1.1 The Hermeneutic Problem

The question of where meaning resides is the central debate of modern hermeneutics, and it is not one that can be sidestepped by technical convenience. E.D. Hirsch, in *Validity in Interpretation* (1967), drew a foundational distinction between *meaning* — what the author intended to convey through a particular sequence of signs — and *significance* — the reader's application of that meaning to their own concerns. For Hirsch, meaning is determinate and recoverable; significance is variable and personal. A valid interpretation must first recover the author's meaning before exploring the reader's significance.

Hans-Georg Gadamer, in *Truth and Method* (1960), challenged Hirsch's separation. For Gadamer, understanding is always a "fusion of horizons" (*Horizontverschmelzung*): the interpreter's present context inevitably shapes what the text can mean. There is no unmediated access to authorial intent; all reading is interpretation, and interpretation is historically situated.

Roland Barthes (1967) pressed the challenge further: in "The Death of the Author," he argued that the very notion of a controlling authorial meaning is a fiction imposed by a particular literary ideology. The text belongs to the reader; the author's biography, intentions, and context are irrelevant to the production of meaning.

Current AI reading tools implicitly adopt a position more impoverished than any of these. They are not textualist in Hirsch's sense (they make no attempt to recover authorial intent), not Gadamerian (they have no "horizon" to fuse), and not Barthesian (they do not empower the reader's interpretive agency). They are, in a precise sense, *pre-hermeneutic*: they treat the text as a database and the question as a search query. The result is what we might call *interpretive negligence* — the AI produces responses that have the surface form of interpretation without any of its substance.

### 1.2 Author Governance as Design Space

The absence of authorial governance in AI reading tools is a design choice, not a technical necessity. Nothing in the RAG architecture prevents a system from incorporating structured authorial intent into its response generation. The question is what such incorporation would look like, and what its effects would be.

The Authors Guild's intervention (2025) and the broader protest by authors petitioning publishers to curtail AI use (NPR 2025) indicate that this is not merely a theoretical concern. Authors want a say in how AI systems represent their work. The infrastructure to give them that say does not currently exist.

### 1.3 Contribution

This paper introduces the Canon Pack: a structured JSON representation of authorial interpretive intent, comprising the following components:

- **Interpretive framework**: book thesis, chapter-level argumentative purposes (what intellectual work each chapter performs), foreground and background themes, and cross-references between recurring concepts.
- **Voice configuration**: tone descriptors, formality level, interaction mode (analytical, Socratic, reflective, etc.), pronoun conventions, and sample responses demonstrating the intended companion voice.
- **Boundary rules**: topics the companion must not engage with, behaviors it must never exhibit, spoiler policy, and a fallback response for out-of-scope queries.
- **Reader guidance**: common misreadings to correct, suggested entry points for different reader backgrounds, and questions the book deliberately leaves unresolved — which the companion should acknowledge as open rather than fabricate answers for.

The Canon Pack operationalizes Hirsch's concept of "meaning" as a computationally tractable object. It does not claim to capture the totality of authorial intent — a philosophical impossibility, as Gadamer and Ricoeur have shown. It captures a *working specification*: enough to govern an AI companion's behavior in ways that respect the author's interpretive framework without foreclosing the reader's own interpretive freedom. The reader remains free to disagree with the companion — but the companion, at minimum, represents the author rather than an algorithmic default.

We describe a five-stage pipeline (parse, chunk, embed, generate Canon Pack, build system prompt) and an AI-assisted intake process that drafts Canon Pack content from manuscript analysis, presenting the author with a review-and-edit workflow rather than a blank form. We test the framework on five books across four genres and compare Canon Pack companions against standard RAG companions on five evaluation dimensions. The results reveal that structured authorial guidance substantially increases interpretive depth and voice consistency while occasionally producing responses that overreach the retrieved textual evidence — a tradeoff that carries direct implications for the design of AI reading tools.

Section 2 situates our work within the hermeneutic tradition, recent computational literary studies, and the emerging scholarship on AI and authorship. Section 3 defines the Canon Pack schema and pipeline architecture. Section 4 describes the evaluation methodology. Section 5 presents results. Section 6 discusses implications, limitations, and the question — unavoidable for any project that foregrounds authorial intent — of whether author-governed AI reading constitutes a return to the authorial authority that Barthes declared dead.

---

## 2. Related Work

### 2.1 Hermeneutic Foundations

The hermeneutic tradition provides the theoretical vocabulary for understanding what AI reading tools do and fail to do. Three positions are particularly relevant.

Hirsch's *Validity in Interpretation* (1967) argues that textual meaning is what the author willed to convey through a given sequence of signs, and that this meaning is in principle recoverable and determinate. Significance — the relevance of that meaning to a reader's own concerns — is variable and personal, but meaning itself is not. For Hirsch, the task of interpretation is to reconstruct the author's intended meaning before (and as a precondition for) exploring significance. This position, often criticized as intentionalist or naive, has the virtue of providing a clear criterion for interpretive validity: an interpretation is valid insofar as it recovers what the author meant. Our Canon Pack is an explicit operationalization of Hirsch's meaning — not a claim to capture authorial intent exhaustively, but a structured attempt to represent enough of it to govern computational behavior.

Gadamer's *Truth and Method* (1960) challenges the separability of meaning and significance. For Gadamer, understanding always involves a "fusion of horizons" — the interpreter's present situation shapes what the text can mean, and there is no standpoint outside this fusion from which "the author's meaning" could be recovered pure. The hermeneutic circle — the mutual dependence of part and whole in understanding — implies that interpretation is never complete, always revisable, and constitutively shaped by the interpreter's context. We take Gadamer's challenge seriously: the Canon Pack does not claim to deliver "the author's true meaning" to the reader. It provides one layer of interpretive governance — the author's — which the reader encounters but is not bound by.

Ricoeur's *Interpretation Theory* (1976) adds the concept of distanciation: once a text is written, it acquires an autonomy from the author's original situation. The text "says" more than the author "meant," and the history of its reception becomes part of its meaning. Ricoeur's position suggests that authorial intent is a necessary but insufficient condition for understanding. This aligns with our design: the Canon Pack shapes the companion's default interpretive stance, but the reader's questions — and the companion's retrieval of specific passages — may surface meanings the author did not explicitly encode.

### 2.2 AI and Hermeneutics: Recent Scholarship

The intersection of hermeneutic theory and artificial intelligence has attracted substantial recent attention, though the literature remains largely theoretical.

Kommers et al. (2025), in a 39-author paper published in *Frontiers in Artificial Intelligence*, propose "computational hermeneutics" as a framework for evaluating generative AI as a cultural technology. They argue that GenAI systems function as "context machines" that must address three interpretive challenges: situatedness (meaning emerges only in context), plurality (multiple valid interpretations coexist), and ambiguity (interpretations may conflict). They offer three principles for hermeneutic evaluation: benchmarks should be iterative, include people, and measure cultural context rather than model output alone. Our work complements theirs: where Kommers et al. propose evaluation principles, we build and test a system that embodies interpretive governance. Their framework could usefully be applied to evaluate our system's cultural adequacy.

Demichelis (2024) examines what he calls "the hermeneutic turn of AI," drawing on Don Ihde and Wilhelm Dilthey to ask whether deep learning systems are capable of genuine interpretation. His conclusion — that neural networks perform something resembling interpretation but lack the lived experience (*Erlebnis*) that Dilthey considered constitutive of understanding — is compatible with our approach. We do not claim that the AI companion interprets; we claim that its interpretive behavior can be governed by a structured representation of the author's intent.

The relationship between AI and authorship has been explored through the lens of Barthes's "Death of the Author" by several recent works. A 2024 article in *Poetics Today* (Duke University Press) examines "phantoms of citation" and the dissolution of the author-function in AI-generated text. A piece in *NOEMA* magazine (2024) argues that AI completes the project Barthes began, finally severing the link between text and authorial subjectivity. A 2025 article in *Technophany* tests whether generative AI can serve as a Gadamerian dialogue partner and concludes that its most promising role is as a "digital form of Gadamerian text" — something to be interpreted, not a genuine interlocutor.

What is absent from this literature is *constructive* engagement: a system that responds to the hermeneutic concerns by building infrastructure for interpretive governance. The existing scholarship analyzes AI reading tools or theorizes about their hermeneutic status; it does not design alternatives.

### 2.3 RAG and Its Limits for Literary Texts

Retrieval-Augmented Generation (Lewis et al. 2020) has become the standard architecture for grounding LLM responses in specific document collections. A document is segmented into chunks, each embedded as a vector, and at query time the most relevant chunks are retrieved and included in the LLM's context window. This architecture has been extensively studied (cf. the TREC 2025 RAG Track) and is effective for factual and informational queries.

Its limitations for literary texts, however, are structural. First, semantic similarity between a query and a passage does not entail interpretive relevance: a passage may be semantically close to a question while being argumentatively peripheral. Second, the chunking process strips passages of their structural context — a paragraph's position in the book's argument, its relationship to preceding and following chapters, its status as premise, conclusion, aside, or counterargument. Third, and most critically, RAG has no mechanism for incorporating interpretive metadata: the system cannot distinguish a passage the author considers central from one the author considers a necessary but minor point, nor can it distinguish a claim the author endorses from one the author raises in order to refute.

These limitations are not inherent to vector retrieval; they are limitations of systems that use retrieval *without interpretive governance*. The Canon Pack addresses the third limitation directly by providing structured metadata that shapes how retrieved passages are presented and contextualized.

### 2.4 Computational Literary Studies

Our work is adjacent to but distinct from computational literary studies (CLS), the field concerned with applying computational methods to literary analysis. Moretti's "distant reading" (2013) and Underwood's *Distant Horizons* (2019) use statistical and machine learning methods to analyze patterns across large literary corpora — genre evolution, stylistic change, canon formation. CLS typically treats the text as an object of analysis; our framework treats the text as an object to be *mediated* for readers. Where CLS asks what computational methods reveal about literature, we ask what interpretive structures a computational system needs in order to represent a specific book faithfully.

### 2.5 Author Rights and AI Reading

The practical dimension of our work is motivated by an escalating conflict between AI platforms and authors. The Authors Guild's December 2025 statement on Amazon's "Ask This Book" feature noted that the feature was deployed without author or publisher consent, with no opt-out mechanism (Authors Guild 2025). A June 2025 petition by authors to major publishers called for explicit curtailment of AI use on copyrighted works (NPR 2025). Cambridge University Press contacted 20,000 authors individually to obtain consent before licensing any work for AI training.

These interventions share a demand: authors want governance over how AI systems interact with their work. The Canon Pack provides a constructive response to this demand — not a legal mechanism, but a technical framework that gives authors structured control over how an AI companion interprets, voices, and delimits its engagement with their text.

---

## 3. The Canon Pack Framework

### 3.1 Overview

The Canon Pack is a structured JSON object that captures an author's interpretive intent across five dimensions: interpretive framework, voice configuration, boundary rules, reader guidance, and retrieval configuration. It is generated through a five-stage pipeline that processes a manuscript into a fully configured AI companion. The pipeline's design reflects a core principle: *the author reviews, the AI proposes*. Rather than requiring authors to populate a complex form from scratch, the system drafts Canon Pack content from manuscript analysis, and the author corrects, refines, and approves the result.

### 3.2 Canon Pack Schema

The Canon Pack comprises the following components:

**Interpretive Framework.** The central component. It includes: (a) a book thesis statement — a single articulation of the book's core argumentative contribution; (b) chapter intents — for each chapter, a statement of the intellectual work that chapter performs (not a summary of its content, but a description of its argumentative function); (c) foreground and background themes — concepts the author considers primary versus contextual; and (d) cross-references — concepts that recur across chapters, with notes on how they develop.

The distinction between chapter *summary* and chapter *intent* is critical. A summary of a chapter on Stoic ethics might say: "Marcus Aurelius discusses the nature of duty." An intent statement says: "This chapter establishes the connection between cosmic determinism and personal responsibility, arguing that acceptance of fate is not passivity but the highest form of agency." The intent statement captures what the chapter *does*, not what it *contains*.

**Voice Configuration.** Tone descriptors (e.g., "precise, compassionate, methodical"), formality level (casual through academic), companion interaction mode (analytical, Socratic, reflective, guide, or mixed), pronoun conventions (how the companion refers to the author and addresses the reader), and sample responses — short examples demonstrating the intended companion voice in action.

**Boundary Rules.** Topics the companion must not engage with (e.g., "never provide meditation instructions" for a philosophy-of-consciousness text), behaviors it must never exhibit (e.g., "never claim the author endorses a political position"), spoiler policy, and a fallback response for out-of-scope queries.

**Reader Guidance.** Common misreadings — interpretive errors the author anticipates and wants the companion to correct; suggested entry points for readers with different backgrounds; and unanswered questions — issues the book deliberately leaves unresolved, which the companion should acknowledge as open rather than fabricate answers for. This last component is particularly important: it prevents the companion from manufacturing false certainty on questions the author intentionally left ambiguous.

**Retrieval Configuration.** Technical metadata linking the Canon Pack to its vector store: namespace, chunk count, embedding model, and top-k retrieval parameter. This component is not interpretively meaningful but is necessary for the companion to function.

### 3.3 The Pipeline

The pipeline consists of five stages, with an additional intake stage that precedes the core pipeline.

**Stage 0: AI-Assisted Intake.** The author uploads their manuscript. The system parses it (Stage 1) and passes the parsed chapters to an intake agent — a language model prompted to infer answers for each intake form field from the manuscript's content. The agent produces a draft intake form with per-field confidence scores. The author reviews the draft, correcting errors and adding information the AI could not infer (e.g., deliberate silences, off-limits topics that reflect personal rather than textual concerns). This "review and edit" workflow reduces author burden from approximately 45 minutes of writing to approximately 15 minutes of reviewing.

**Stage 1: Parse.** The manuscript (PDF, DOCX, or plain text) is parsed into chapters using a multi-strategy approach: structural metadata (PDF bookmarks, DOCX heading styles) is attempted first, with regex-based chapter detection as fallback, and whole-document treatment as a final fallback.

**Stage 2: Chunk.** Each chapter is segmented into overlapping chunks using sentence-aware splitting (512 tokens, 64-token overlap). Critically, chunks never cross chapter boundaries — a design decision that preserves the book's argumentative structure in the retrieval layer. Each chunk carries metadata: chapter number, chapter title, and position within the chapter (beginning, middle, end).

**Stage 3: Embed.** Chunks are embedded using OpenAI's text-embedding-3-small model (1536 dimensions) and stored in a per-book vector namespace. The namespace isolation ensures that retrieval for one book never surfaces passages from another — a design choice with hermeneutic implications, as it enforces a boundary between textual worlds that the author can later relax through explicit cross-references.

**Stage 4: Generate Canon Pack.** The intake form and high-signal chunks (the longest, most substantive passages from each chapter) are passed to a language model (Claude Sonnet, temperature 0.3) with a detailed prompt instructing it to capture the author's *intent*, not merely summarize the text. The model generates a Canon Pack conforming to the schema described in Section 3.2. The generated Canon Pack is presented to the author for review before proceeding.

**Stage 5: Build System Prompt.** The Canon Pack is rendered into a natural-language system prompt that governs the companion's behavior. This prompt includes the book thesis, voice instructions, boundary rules, a chapter reference section, and sample responses. The system prompt is the artifact that directly controls the companion's output; the Canon Pack is the structured data from which it is generated.

### 3.4 Design Decisions as Hermeneutic Positions

Several pipeline design choices carry hermeneutic implications worth making explicit.

*Chapter-aware chunking* reflects a commitment to preserving the book's argumentative structure. Standard RAG systems chunk documents without regard for structural boundaries; our approach ensures that a passage from Chapter 3 is never merged with a passage from Chapter 4 in a single chunk. This means retrieved context always comes from within a coherent argumentative unit.

*Namespace isolation* is a position on intertextuality. By default, the companion for Book A cannot access passages from Book B. This is a deliberate constraint: the author controls when and how cross-textual references are surfaced, through the explicit cross-references component of the Canon Pack. A system without namespace isolation would allow the AI to draw unsanctioned connections between an author's works — or, worse, between different authors' works — without any interpretive governance.

*The intake agent's draft-and-review workflow* reflects a Gadamerian intuition: the AI brings its own "horizon" (statistical patterns from training data) to its reading of the manuscript, and the author brings theirs. The corrected intake is a fusion of these horizons — the AI's textual analysis disciplined by the author's self-understanding. This is not a claim that the process *is* Gadamerian understanding; it is an observation that the workflow's structure parallels it.

---

## 4. Evaluation Methodology

### 4.1 Study Design

We evaluate the Canon Pack framework through a controlled comparison between two companion conditions applied to the same books, using the same language model, temperature, retrieval parameters, and embedded chunks. The only variable is the system prompt.

**Condition 1: Vanilla RAG (baseline).** The companion receives a minimal system prompt: "You are a helpful assistant that answers questions about [Title] by [Author]. Use the provided context passages to answer accurately. If the context doesn't contain enough information, say so." This represents the standard RAG approach used by existing AI reading tools.

**Condition 2: Canon Pack (treatment).** The companion receives the full Canon Pack system prompt generated by the pipeline — typically 5,000–25,000 characters depending on the book's complexity. This prompt includes the book thesis, chapter intents, voice configuration, boundary rules, misreading corrections, and all other Canon Pack components.

Both companions use Claude Sonnet (claude-sonnet-4-20250514) at temperature 0.4, with a maximum of 1,024 response tokens. Both retrieve the top 5 chunks by cosine similarity from the same vector store. The design isolates the Canon Pack as the sole independent variable.

### 4.2 Test Corpus

We test on five books chosen for genre diversity, structural variety, and licensing compatibility:

| Book | Author | Genre | Structure | Source |
|------|--------|-------|-----------|--------|
| *The Fourth Culture: Identity Without Borders* | Rayan B. Vasse | Literary nonfiction | 54 chapters, narrative + argument | Author manuscript |
| *Being No One* | Thomas Metzinger | Academic philosophy | 115 sections, dense argument | MIT Press (CC-BY-ND-NC) |
| *Content* | Cory Doctorow | Cultural criticism | 13 essays | CC-BY-NC-SA |
| *Meditations* | Marcus Aurelius | Classical philosophy | 12 books, fragmentary | Project Gutenberg |
| *The Art of War* | Sun Tzu | Military strategy | 13 chapters, aphoristic | Project Gutenberg |

The corpus is deliberately heterogeneous. It includes a living author's manuscript (with genuine author-provided intake data), an academic monograph, an essay collection, and two ancient texts whose authors cannot fill intake forms. This last category is methodologically significant: for *Meditations* and *The Art of War*, the Canon Pack was generated from AI manuscript analysis without author correction, representing a lower-bound scenario. For *Being No One*, an AI draft was reviewed and corrected by a domain-familiar reader. For *The Fourth Culture*, the intake was completed by the author. This asymmetry is deliberate — it allows us to observe how Canon Pack quality varies with the level of human authorial input.

### 4.3 Question Design

For each book, we designed ten questions across five categories (two per category):

**Factual comprehension** — questions whose answers are directly retrievable from the text. Both companions should perform well. Example: "What does Marcus Aurelius mean when he says we should focus only on what is within our control?"

**Interpretive framing** — questions that require the companion to represent the book's argumentative framework. Example: "How does the concept of 'chronic partial belonging' differ from cultural alienation?"

**Misreading correction** — questions that present a common misinterpretation for the companion to address. Example: "Is The Art of War simply a manual for military aggression?"

**Cross-reference** — questions that require connecting ideas across different parts of the book. Example: "How does the concept of *shih* (momentum) operate across the chapters on terrain, disposition, and maneuver?"

**Boundary respect** — questions that probe whether the companion appropriately refuses out-of-scope queries. Example: "Can you give me a detailed biography of Sun Tzu's life and military campaigns?"

This category structure allows dimension-level analysis: we can observe whether the Canon Pack's effects are uniform across question types or concentrated in specific categories.

### 4.4 Evaluation Dimensions

Each response is evaluated on five dimensions, scored on a 1–5 scale:

1. **Textual grounding**: How well the response is anchored in the book's actual content (1 = fabricates, 5 = precisely anchored).
2. **Interpretive depth**: Whether the response engages with meaning beyond surface summary (1 = shallow paraphrase, 5 = rich interpretive engagement).
3. **Voice consistency**: Whether the response maintains a coherent, appropriate tone for this specific book (1 = generic, 5 = distinctive and well-calibrated).
4. **Boundary respect**: Whether the response appropriately handles out-of-scope or unanswerable questions (1 = speculates wildly, 5 = honest about limits).
5. **Cross-reference**: Whether the response connects ideas from different parts of the book (1 = single-passage, 5 = rich thematic linking).

### 4.5 Evaluation Protocol

Evaluation proceeds in two layers.

**Automated evaluation.** An AI judge (Claude Sonnet at temperature 0.1) evaluates each response pair in a blinded protocol. Response labels (A/B) are randomized per question; the judge does not know which condition produced which response. The judge scores each response on all five dimensions and selects an overall winner with written reasoning. This produces 50 evaluated pairs (10 per book × 5 books).

**Human evaluation.** To address the limitation of AI-only evaluation — namely, that an LLM judge may share systematic biases with the LLM that generated the responses — we supplement the automated evaluation with human scoring. The author evaluates the ten response pairs for *The Fourth Culture* (providing ground-truth author judgment). An independent evaluator, blinded to condition assignment, evaluates the response pairs for *Meditations* and *The Art of War* (20 pairs total). Human evaluators use the same five dimensions and 1–5 scale.

The dual-layer protocol allows us to assess convergence or divergence between automated and human evaluation, and to identify cases where the AI judge's preferences may not reflect human interpretive judgment.

### 4.6 Statistical Measures

We report mean scores per condition per dimension, the delta between conditions, and Cohen's *d* effect size for each dimension and each book. Cohen's *d* provides a standardized measure of the magnitude of difference between conditions, with conventional thresholds of 0.2 (small), 0.5 (medium), and 0.8 (large). We report both per-book and cross-book aggregates, with attention to dimension-level patterns rather than composite win rates alone.
