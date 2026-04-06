/* ── Config ── */
// UPDATE THIS after Render deploy:
const API_BASE = "https://genai-hermeneutics.onrender.com";

const SESSION_LIMIT = 3;
const SESSION_KEY = "gh_ask_count";

/* ── State ── */
let booksIndex = [];
let examplesCache = {};

/* ── DOM refs ── */
const bookSelect = document.getElementById("book-select");
const bookInfo = document.getElementById("book-info");
const examplesList = document.getElementById("examples-list");
const askBookSelect = document.getElementById("ask-book-select");
const askForm = document.getElementById("ask-form");
const askQuestion = document.getElementById("ask-question");
const askBtn = document.getElementById("ask-btn");
const askCounter = document.getElementById("ask-counter");
const askSpinner = document.getElementById("ask-spinner");
const askResult = document.getElementById("ask-result");
const askError = document.getElementById("ask-error");

/* ── Tab switching ── */
document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
        document.querySelectorAll(".mode-panel").forEach((p) => p.classList.remove("active"));
        tab.classList.add("active");
        document.getElementById(tab.dataset.mode + "-mode").classList.add("active");
    });
});

/* ── Init ── */
async function init() {
    try {
        const res = await fetch(API_BASE + "/books");
        const data = await res.json();
        booksIndex = data.books;
    } catch {
        booksIndex = [];
    }

    populateSelects();
    updateAskCounter();

    if (booksIndex.length > 0) {
        bookSelect.value = booksIndex[0].slug;
        askBookSelect.value = booksIndex[0].slug;
        loadExamples(booksIndex[0].slug);
    }
}

function populateSelects() {
    const opts = booksIndex
        .map((b) => `<option value="${b.slug}">${b.title} — ${b.author}</option>`)
        .join("");
    bookSelect.innerHTML = opts || '<option value="">No books available</option>';
    askBookSelect.innerHTML = opts || '<option value="">No books available</option>';
}

/* ── Browse mode ── */
bookSelect.addEventListener("change", () => loadExamples(bookSelect.value));

async function loadExamples(slug) {
    if (!slug) return;

    examplesList.innerHTML = '<p class="spinner"><span class="spinner-dot"></span>Loading...</p>';
    bookInfo.classList.add("hidden");

    try {
        let data = examplesCache[slug];
        if (!data) {
            const res = await fetch(API_BASE + "/examples/" + slug);
            if (!res.ok) throw new Error("Failed to load");
            data = await res.json();
            examplesCache[slug] = data;
        }

        renderBookInfo(data);
        renderExamples(data.examples);
    } catch (e) {
        examplesList.innerHTML = '<p class="error-msg">Could not load examples. Is the API running?</p>';
    }
}

function renderBookInfo(data) {
    const s = data.summary || {};
    const wins = s.wins || {};
    const comp = s.composite || {};

    bookInfo.innerHTML = `
        <h2>${esc(data.title)}</h2>
        <p class="author">${esc(data.author)} (${esc(data.year)})</p>
        <p class="description">${esc(data.description)}</p>
        <div class="stats">
            <span>Questions: <strong>${data.num_questions}</strong></span>
            <span>Canon Pack wins: <strong>${wins.canon_pack || 0}</strong></span>
            <span>Vanilla RAG wins: <strong>${wins.vanilla_rag || 0}</strong></span>
            ${comp.cohens_d != null ? `<span>Effect size (d): <strong>${comp.cohens_d.toFixed(2)}</strong></span>` : ""}
        </div>
    `;
    bookInfo.classList.remove("hidden");
}

function renderExamples(examples) {
    examplesList.innerHTML = examples.map((ex) => renderComparison(ex)).join("");
}

function renderComparison(ex) {
    const isCanonWinner = ex.winner === "canon_pack";
    const scores = ex.scores || {};

    return `
    <div class="comparison">
        <div class="comparison-header">
            <div class="question">${esc(ex.question)}</div>
            <span class="category">${esc(ex.category)}</span>
        </div>
        <div class="responses">
            <div class="response-col vanilla">
                <div class="response-label">
                    Vanilla RAG
                    ${!isCanonWinner && ex.winner === "vanilla_rag" ? '<span class="winner-badge" style="background:var(--vanilla-border)">Winner</span>' : ""}
                </div>
                <div class="response-text">${formatResponse(ex.vanilla_rag_response)}</div>
            </div>
            <div class="response-col canon">
                <div class="response-label">
                    Canon Pack
                    ${isCanonWinner ? '<span class="winner-badge">Winner</span>' : ""}
                </div>
                <div class="response-text">${formatResponse(ex.canon_pack_response)}</div>
            </div>
        </div>
        ${renderScoresBar(scores)}
        ${ex.reasoning ? `
        <details class="reasoning-toggle">
            <summary>Judge reasoning</summary>
            <p class="reasoning-text">${esc(ex.reasoning)}</p>
        </details>` : ""}
    </div>`;
}

function renderScoresBar(scores) {
    if (!scores.vanilla_rag && !scores.canon_pack) return "";

    const dims = ["textual_grounding", "interpretive_depth", "voice_consistency", "boundary_respect", "cross_reference"];
    const items = dims
        .filter((d) => scores.vanilla_rag && scores.vanilla_rag[d] != null)
        .map((d) => {
            const v = scores.vanilla_rag[d];
            const c = scores.canon_pack[d];
            const label = d.replace(/_/g, " ");
            return `<span class="score-item">
                <span class="dim-name">${label}:</span>
                <span class="score-pair">${v} vs ${c}</span>
            </span>`;
        })
        .join("");

    return `<div class="scores-bar">${items}</div>`;
}

/* ── Ask mode ── */
function getSessionCount() {
    return parseInt(sessionStorage.getItem(SESSION_KEY) || "0", 10);
}

function incrementSessionCount() {
    const c = getSessionCount() + 1;
    sessionStorage.setItem(SESSION_KEY, String(c));
    return c;
}

function updateAskCounter() {
    const used = getSessionCount();
    const left = Math.max(0, SESSION_LIMIT - used);
    askCounter.textContent = `${left} question${left !== 1 ? "s" : ""} remaining this session`;
    if (left <= 0) {
        askBtn.disabled = true;
        askQuestion.disabled = true;
    }
}

askForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    if (getSessionCount() >= SESSION_LIMIT) {
        showError("Session limit reached. Refresh or browse pre-computed examples.");
        return;
    }

    const slug = askBookSelect.value;
    const question = askQuestion.value.trim();
    if (!slug || question.length < 10) return;

    askBtn.disabled = true;
    askSpinner.classList.remove("hidden");
    askResult.classList.add("hidden");
    askError.classList.add("hidden");

    try {
        const res = await fetch(API_BASE + "/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ book_slug: slug, question }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Error ${res.status}`);
        }

        const data = await res.json();
        incrementSessionCount();
        updateAskCounter();

        renderAskResult(data);
    } catch (err) {
        showError(err.message);
    } finally {
        askSpinner.classList.add("hidden");
        askBtn.disabled = getSessionCount() >= SESSION_LIMIT;
    }
});

function renderAskResult(data) {
    askResult.innerHTML = `
    <div class="comparison-header">
        <div class="question">${esc(data.question)}</div>
        <span class="category">live</span>
    </div>
    <div class="responses">
        <div class="response-col vanilla">
            <div class="response-label">Vanilla RAG</div>
            <div class="response-text">${formatResponse(data.vanilla_rag_response)}</div>
        </div>
        <div class="response-col canon">
            <div class="response-label">Canon Pack</div>
            <div class="response-text">${formatResponse(data.canon_pack_response)}</div>
        </div>
    </div>`;
    askResult.classList.remove("hidden");
}

function showError(msg) {
    askError.textContent = msg;
    askError.classList.remove("hidden");
}

/* ── Helpers ── */
function esc(s) {
    if (!s) return "";
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
}

function formatResponse(text) {
    if (!text) return "<em>No response</em>";
    // Convert markdown-like bold and paragraphs
    return text
        .split(/\n{2,}/)
        .map((p) => `<p>${esc(p.trim()).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")}</p>`)
        .join("");
}

/* ── Boot ── */
init();
