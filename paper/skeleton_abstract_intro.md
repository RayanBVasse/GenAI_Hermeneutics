# Paper Skeleton: Abstract & Introduction

**Working Title:** "The Canon Pack: Structuring Authorial Intent for AI Literary Companions"

**Alt:** "Beyond Retrieval: Operationalizing Authorial Intent in AI-Mediated Reading"

NOTE: Avoid "Computational Hermeneutics" as a title — Kommers et al. (2025/2026) already claimed that exact phrase for a 39-author Frontiers in AI paper. We cite it; we don't compete with it for naming rights.

---

## ABSTRACT (~250 words) — SKELETON

[PROBLEM] AI-powered reading tools — from Amazon's "Ask This Book" to Google's NotebookLM — increasingly mediate how readers engage with books. These systems treat texts as information retrieval problems: they chunk the text, embed it, and answer questions by finding relevant passages. The implicit hermeneutic model is naive textualism: meaning resides in the text and can be extracted by pattern matching.

[GAP] This approach ignores a century of hermeneutic theory (Gadamer, Ricoeur, Hirsch) on the role of interpretive context in meaning-making. More practically, it strips authors of any governance over how AI systems represent their work — a concern now actively raised by the Authors Guild in response to Amazon's Kindle AI features.

[CONTRIBUTION] We introduce the Canon Pack: a structured representation of authorial interpretive intent — including the author's thesis, chapter-level argumentative purposes, voice configuration, common misreadings to correct, boundary rules, and deliberate silences. We describe a pipeline that generates a draft Canon Pack from manuscript analysis and presents it to the author for review, and a companion system governed by the resulting interpretive framework.

[METHOD] We test the framework on five books spanning four genres (academic philosophy, cultural criticism, classical philosophy, military strategy, literary nonfiction). In a controlled comparison, Canon Pack companions and standard RAG companions answer the same questions using identical retrieval and models, differing only in their system-level interpretive guidance.

[FINDING] Canon Pack companions show [substantially / measurably] greater interpretive depth and voice consistency, while revealing a tension with textual grounding that raises productive questions about the design of AI reading tools.

---

## INTRODUCTION (~1,500 words) — SKELETON

### Opening move: The proliferation problem

[2-3 paragraphs]

AI reading tools are proliferating. Amazon's "Ask This Book" (Dec 2025) lets Kindle readers query an AI chatbot about purchased books — without author consent or opt-out (Authors Guild, 2025). Google's NotebookLM treats uploaded documents as conversation partners. Dozens of startups offer "chat with your PDF" services.

These tools share a common architecture: Retrieval-Augmented Generation (RAG). The text is chunked, embedded as vectors, and when a reader asks a question, relevant passages are retrieved and fed to a large language model alongside the query. The implicit theory of reading: the meaning is in the text, and the AI's job is to find it.

This works for factual questions. "What chapter discusses X?" "Who is the main character?" But it fails precisely where reading matters most — at the level of interpretation. When a reader asks "Is this book arguing that consciousness is an illusion?" or "Does the author think borders still matter?", a RAG system has no access to what the author actually intended. It can only pattern-match against passages, producing answers that may directly contradict the author's stated interpretive framework.

### The hermeneutic problem

[2-3 paragraphs]

This is not a new problem in literary theory. The question of where meaning resides — in the text, in the author's intention, or in the reader's encounter — is the central debate of modern hermeneutics.

E.D. Hirsch (1967) distinguished between *meaning* (what the author intended) and *significance* (the reader's application of that meaning to their own context). For Hirsch, meaning is determinate and recoverable; significance is variable and personal. A valid interpretation must recover the author's meaning before exploring the reader's significance.

Gadamer (1960) disagreed: understanding is always a "fusion of horizons" between the interpreter's context and the text's. There is no meaning prior to interpretation.

Barthes (1967) went further: the author is dead; the text belongs to the reader.

Current AI reading tools implicitly adopt an extreme version of none of these positions. They are pre-hermeneutic — they don't have a theory of interpretation at all. They treat the text as a database and the question as a query. The result is what we might call *interpretive negligence*: the AI produces answers that have the form of interpretation without any of its substance.

### The author's-rights dimension

[1-2 paragraphs]

This is not merely a theoretical concern. The Authors Guild's December 2025 statement on "Ask This Book" explicitly raised the question of whether AI-mediated reading without author consent infringes authors' rights. Authors have no mechanism to specify how an AI should interpret their work, what it should refuse to discuss, or what common misreadings it should correct.

The absence of author governance is a design choice, not a technical necessity.

### Our contribution

[2-3 paragraphs]

We introduce the Canon Pack: a structured JSON object that captures an author's interpretive intent across multiple dimensions — book thesis, chapter-level argumentative purposes, foreground and background themes, voice configuration, common misreadings to correct, off-limits topics, spoiler policy, and deliberate silences (questions the book intentionally leaves open).

The Canon Pack operationalizes Hirsch's concept of "meaning" as a computationally tractable object. It does not claim to capture the totality of authorial intent — a philosophical impossibility, as Gadamer and Ricoeur have shown. Rather, it captures a *working specification* of interpretive guidance: enough to govern an AI companion's behavior in ways that respect the author's framework without foreclosing the reader's own interpretive freedom.

We describe a five-stage pipeline (parse, chunk, embed, generate Canon Pack, build system prompt) that transforms a manuscript and an author-reviewed intake form into a fully governed AI companion. We test this on five books and compare Canon Pack companions against standard RAG companions on five evaluation dimensions. The results reveal a productive tension: structured authorial guidance substantially increases interpretive depth and voice consistency while occasionally producing responses that overreach the retrieved textual evidence — a tradeoff with direct implications for the design of AI reading tools.

### Paper structure

[1 paragraph]

Section 2 situates our work in the hermeneutic tradition and recent AI/literary scholarship. Section 3 defines the Canon Pack framework and pipeline. Section 4 describes our evaluation methodology. Section 5 presents results across five books and five dimensions. Section 6 discusses the implications, limitations, and the Barthesian question of whether author-governed AI reading constitutes a return to authorial authority.

---

## KEY LITERATURE TO ENGAGE (for Related Work section)

### Must cite and differentiate from:

1. Kommers et al. (2025/2026). "Computational Hermeneutics: Evaluating Generative AI as a Cultural Technology." Frontiers in AI. — Their framework: GenAI as "context machines" facing situatedness, plurality, ambiguity. Our differentiation: they propose evaluation principles; we build and test an actual system.

2. Demichelis (2024). "The Hermeneutic Turn of AI: Are Machines Capable of Interpreting?" arXiv 2411.12517. — Philosophical, draws on Ihde/Dilthey. Our differentiation: we don't ask whether machines interpret; we ask how to govern their interpretation.

3. Van de Ven & Chateau (2024). "Digital Culture and the Hermeneutic Tradition." Routledge. — Applies Ricoeur/Gadamer to digital platforms. Our differentiation: they analyze existing platforms; we design a new one.

### Must cite for theoretical grounding:

4. Hirsch (1967). Validity in Interpretation. — Core theoretical anchor: meaning vs. significance distinction.
5. Gadamer (1960/2004). Truth and Method. — Fusion of horizons, the hermeneutic circle.
6. Barthes (1967). "The Death of the Author." — The counter-position we must address.
7. Ricoeur (1976). Interpretation Theory. — Distanciation, the text as autonomous from author.

### Must cite for AI/authorship context:

8. "Phantoms of Citation: AI and the Death of the Author-Function." Poetics Today 45(2), 2024. Duke UP. — Barthes/Foucault + LLMs.
9. Technophany (2025). "Is Generative AI Ready to Join the Conversation That We Are?" — Gadamer vs. ChatGPT.
10. ACM Hypertext (2024). "Emotional Hermeneutics." — Dilthey + AI limits.
11. NOEMA (2024). "AI Signals The Death of the Author."

### Must cite for real-world motivation:

12. Authors Guild (2025). Statement on Amazon Kindle "Ask This Book." — No author opt-out.
13. NPR (2025). "Authors Petition Publishers to Curtail Their Use of AI."

### Should cite for RAG/technical context:

14. Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." — Original RAG paper.
15. TREC 2025 RAG Track. — Current state of RAG evaluation.

### Should cite for computational literary studies context:

16. Moretti (2013). Distant Reading. — Foundational CLS reference.
17. Underwood (2019). Distant Horizons. — Computational approaches to literary history.

---

## WHAT'S GENUINELY NEW (our positioning)

Nobody has:
- Defined a structured representation of authorial interpretive intent for governing LLM behavior
- Built a system that generates such a representation from manuscript analysis + author review
- Tested what changes when you inject authorial intent into an AI reading companion
- Reported the tension between interpretive depth and textual grounding that emerges

Everyone else is either:
- Theorizing about AI and hermeneutics without building anything (Kommers, Demichelis, Barthes/AI papers)
- Building AI reading tools without any hermeneutic framework (Amazon, NotebookLM, RAG chatbots)
- Protesting the absence of author control without proposing a technical solution (Authors Guild)

We sit at the intersection: hermeneutic theory + working system + empirical results + author governance.
