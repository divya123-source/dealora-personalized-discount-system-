const API_BASE = window.DEALORA_API_BASE || "";
const API_URL = `${API_BASE}/api/model/evaluation`;

function el(tag, className, text) {
	const node = document.createElement(tag);
	if (className) node.className = className;
	if (text !== undefined && text !== null) node.textContent = text;
	return node;
}

const num = (value, digits = 4) =>
	typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "n/a";
const pct = (value, digits = 2) =>
	typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "n/a";
const whole = (value) =>
	typeof value === "number" && Number.isFinite(value) ? value.toLocaleString("en-IN") : "n/a";

function loadStyles() {
	if (document.querySelector("link[data-model-evaluation-css]")) return;
	const link = document.createElement("link");
	link.rel = "stylesheet";
	link.href = "css/model_evaluation.css";
	link.setAttribute("data-model-evaluation-css", "");
	document.head.appendChild(link);
}

function buildShell() {
	const host = document.getElementById("admin-content");
	if (!host) return null;
	let section = document.getElementById("model-evaluation");
	if (section) return section.querySelector("#me-body");

	section = el("section", "admin-section me");
	section.id = "model-evaluation";

	const heading = el("div", "section-heading section-heading--small");
	const titleWrap = el("div");
	titleWrap.appendChild(el("p", "eyebrow", "The recommendation model"));
	const h2 = el("h2", null, "Model ");
	h2.appendChild(el("em", null, "evaluation."));
	titleWrap.appendChild(h2);
	heading.appendChild(titleWrap);
	section.appendChild(heading);

	const body = el("div", "me-body");
	body.id = "me-body";
	body.appendChild(el("div", "loading-state", "Loading model evaluation..."));
	section.appendChild(body);

	host.appendChild(section);
	return body;
}

function card(label, value, hint) {
	const c = el("div", "me-card");
	c.appendChild(el("span", "me-card__label", label));
	c.appendChild(el("strong", "me-card__value", value));
	if (hint) c.appendChild(el("span", "me-card__hint", hint));
	return c;
}

function titled(title, subtitle) {
	const box = el("div", "me-block");
	box.appendChild(el("h3", "subsection-title me-block__title", title));
	if (subtitle) box.appendChild(el("p", "me-block__sub", subtitle));
	return box;
}

function list(items) {
	const ul = el("ul", "me-list");
	(items || []).forEach((text) => ul.appendChild(el("li", null, text)));
	return ul;
}

function statusBanner(data) {
	const met = data.target && data.target.target_met === true;
	const metric = (data.target && data.target.primary_reported_metric) || "ndcg_at_10";
	const measured = data.target && data.target.selected_model_ndcg_at_10;
	const box = el("div", `me-status ${met ? "me-status--ok" : "me-status--warn"}`);
	box.appendChild(el("strong", null,
		`${(data.target && data.target.requested) || "Target"}: ${met ? "met" : "not met"}`));
	box.appendChild(el("p", null,
		`Measured ${metric.replace("_at_", "@").toUpperCase()} on the untouched final test set: ` +
		`${num(measured)} (${pct(measured)}). These results are reported exactly as measured.`));
	return box;
}

function idNotice(model) {
	const parts = [];
	if (model.id_namespace) parts.push(`Data used: ${model.id_namespace}.`);
	parts.push(model.used_in_live_store
		? "This model is used in the live store."
		: "This model is not used in the live store; this page reports evaluation results only.");
	return el("div", "me-note", parts.join(" "));
}

function summaryCards(data) {
	const grid = el("div", "me-cards");
	const model = data.model || {};
	const hp = model.hyperparameters || {};
	const final = (data.final_test && data.final_test.selected_model) || {};
	const folds = (data.candidates && data.candidates[0] && data.candidates[0].folds) || 0;
	grid.appendChild(card("Selected model", model.name || "n/a",
		`Version ${model.version || "n/a"} | neighbors ${hp.neighbors ?? "n/a"} | window ${hp.recent_window ?? "n/a"}`));
	grid.appendChild(card("Interactions used", whole(data.dataset && data.dataset.valid_rows),
		`${whole(data.dataset && data.dataset.rejected_rows)} rejected rows`));
	grid.appendChild(card("Final-test users", whole(final.evaluation_users), "Users evaluated on the test period"));
	grid.appendChild(card("Validation folds", String(folds), "Walk-forward, training period only"));
	return grid;
}

const FINAL_METRICS = [
	["ndcg_at_10", "NDCG@10"],
	["hit_rate_at_10", "HitRate@10"],
	["map_at_10", "MAP@10"],
	["recall_at_10", "Recall@10"],
	["precision_at_10", "Precision@10"],
];

function comparisonChart(data) {
	const base = (data.final_test && data.final_test.popularity_baseline) || {};
	const sel = (data.final_test && data.final_test.selected_model) || {};
	const box = titled("Final test: baseline vs selected model",
		"Each metric has its own scale, so the longer bar is set to full width within that row. Real values are printed beside the bars.");

	const legend = el("div", "me-legend");
	[["me-dot me-dot--base", "Popularity baseline"], ["me-dot me-dot--sel", `Selected: ${(data.model && data.model.name) || "model"}`]]
		.forEach(([cls, text]) => {
			const item = el("span", "me-legend__item");
			item.appendChild(el("i", cls));
			item.appendChild(document.createTextNode(text));
			legend.appendChild(item);
		});
	box.appendChild(legend);

	FINAL_METRICS.forEach(([key, label]) => {
		const a = base[key];
		const b = sel[key];
		const max = Math.max(a || 0, b || 0) || 1;
		const row = el("div", "me-row");
		row.appendChild(el("div", "me-row__label", label));
		const bars = el("div", "me-row__bars");
		[[a, "me-bar me-bar--base"], [b, "me-bar me-bar--sel"]].forEach(([value, cls]) => {
			const line = el("div", "me-line");
			const track = el("div", "me-track");
			const fill = el("div", cls);
			fill.style.width = `${Math.max(((value || 0) / max) * 100, 1)}%`;
			track.appendChild(fill);
			line.appendChild(track);
			line.appendChild(el("span", "me-line__value", num(value)));
			bars.appendChild(line);
		});
		row.appendChild(bars);
		box.appendChild(row);
	});
	return box;
}

function targetChart(data) {
	const sel = (data.final_test && data.final_test.selected_model) || {};
	const box = titled("Measured score vs the 85-90% target",
		"Same 0-100% scale for every bar. The shaded band marks the requested 85-90% range.");
	[["ndcg_at_10", "NDCG@10"], ["hit_rate_at_10", "HitRate@10"]].forEach(([key, label]) => {
		const value = sel[key] || 0;
		const row = el("div", "me-target");
		row.appendChild(el("div", "me-row__label", label));
		const track = el("div", "me-track me-track--target");
		const band = el("div", "me-band");
		band.style.left = "85%";
		band.style.width = "5%";
		track.appendChild(band);
		const fill = el("div", "me-bar me-bar--sel");
		fill.style.width = `${Math.max(value * 100, 0.5)}%`;
		track.appendChild(fill);
		row.appendChild(track);
		row.appendChild(el("span", "me-line__value", pct(value)));
		box.appendChild(row);
	});
	const scale = el("div", "me-scale");
	["0%", "25%", "50%", "75%", "100%"].forEach((t) => scale.appendChild(el("span", null, t)));
	box.appendChild(scale);
	return box;
}

function cvTable(data) {
	const box = titled("Cross-validation results (mean and standard deviation across folds)",
		`Models were compared on ${(data.protocol && data.protocol.selection_metric) || "the selection metric"}.`);
	const wrap = el("div", "table-scroll");
	const table = el("table", "data-table me-table");
	const head = el("tr");
	["Model", "Neighbors", "Window", "NDCG@10", "MAP@10", "Recall@10", "Precision@10"]
		.forEach((h) => head.appendChild(el("th", null, h)));
	const thead = el("thead");
	thead.appendChild(head);
	table.appendChild(thead);

	const selectedName = data.selection && data.selection.selected_model && data.selection.selected_model.name;
	const tbody = el("tbody");
	(data.candidates || []).forEach((c) => {
		const tr = el("tr", c.name === selectedName ? "me-selected" : "");
		const nameCell = el("td", null, c.name || "n/a");
		if (c.name === selectedName) nameCell.appendChild(el("span", "me-tag", "Selected"));
		tr.appendChild(nameCell);
		tr.appendChild(el("td", null, String(c.neighbors ?? "n/a")));
		tr.appendChild(el("td", null, String(c.recent_window ?? "n/a")));
		["ndcg_at_10", "map_at_10", "recall_at_10", "precision_at_10"].forEach((k) => {
			const stats = (c.cv && c.cv[k]) || {};
			tr.appendChild(el("td", null, `${num(stats.mean)} \u00b1 ${num(stats.std)}`));
		});
		tbody.appendChild(tr);
	});
	table.appendChild(tbody);
	wrap.appendChild(table);
	box.appendChild(wrap);
	return box;
}

function notesBlock(data) {
	const box = titled("How this was evaluated");
	const p = data.protocol || {};
	box.appendChild(list([
		p.split && `Split: ${p.split}.`,
		p.cross_validation && `Cross-validation: ${p.cross_validation}.`,
		data.selection && data.selection.reason && data.selection.reason.replace(/(\d+\.\d{4})\d+/g, "$1"),
	].filter(Boolean)));

	const target = data.target || {};
	if (target.note) box.appendChild(el("p", "me-block__sub me-note-text", target.note));
	if ((target.limitations || []).length) {
		box.appendChild(el("h4", "me-small-title", "Known limitations"));
		box.appendChild(list(target.limitations));
	}
	if ((target.legitimate_improvements || []).length) {
		box.appendChild(el("h4", "me-small-title", "Legitimate ways to improve"));
		box.appendChild(list(target.legitimate_improvements));
	}
	return box;
}

function render(body, data) {
	body.replaceChildren(
		statusBanner(data),
		idNotice(data.model || {}),
		summaryCards(data),
		comparisonChart(data),
		targetChart(data),
		cvTable(data),
		notesBlock(data),
		el("p", "me-source",
			`Source: data/processed/trial/retailrocket_experiment.json and model_card.json | generated ${data.generated_at_utc || "n/a"}`),
	);
}

async function init() {
	const body = buildShell();
	if (!body) return;
	loadStyles();
	try {
		const response = await fetch(API_URL, { headers: { Accept: "application/json" } });
		const data = await response.json().catch(() => ({}));
		if (!response.ok) {
			const detail = data.detail ? ` (${data.detail})` : "";
			body.replaceChildren(el("div", "me-error",
				`Model evaluation results are unavailable: ${data.error || response.status}${detail}.`));
			return;
		}
		render(body, data);
	} catch (error) {
		body.replaceChildren(el("div", "me-error", "Model evaluation results could not be loaded. Check that the server is running."));
	}
}

if (document.readyState === "loading") {
	document.addEventListener("DOMContentLoaded", init);
} else {
	init();
}