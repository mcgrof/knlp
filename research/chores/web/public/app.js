const byId = (id) => document.getElementById(id);

const dateFormatter = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "short",
});

function formatTime(value) {
  return dateFormatter.format(new Date(value));
}

function badge(value) {
  const element = document.createElement("span");
  element.classList.add("badge", value);
  element.textContent = value.replaceAll("-", " ");
  return element;
}

function renderProgress(value) {
  const row = document.createElement("div");
  row.className = "progress-row";

  const progress = document.createElement("progress");
  progress.max = 100;
  progress.value = value;
  progress.setAttribute("aria-label", `${value}% complete`);

  const label = document.createElement("span");
  label.textContent = `${value}%`;
  row.append(progress, label);
  return row;
}

function renderTrackingDetails(record) {
  const details = [];
  if (record.performed_by) {
    details.push(`Performed by ${record.performed_by}`);
  }
  if (record.reviewed_by) {
    details.push(`Reviewed by ${record.reviewed_by}`);
  }
  if (record.next_review_at) {
    details.push(`Next review ${formatTime(record.next_review_at)}`);
  }
  if (!details.length) {
    return null;
  }

  const element = document.createElement("p");
  element.className = "tracking-details";
  element.textContent = details.join(" · ");
  return element;
}

function renderWorkstream(workstream) {
  const article = document.createElement("article");
  article.className = "workstream";

  const heading = document.createElement("div");
  heading.className = "workstream-heading";
  const title = document.createElement("h3");
  title.textContent = workstream.label;
  heading.append(title, badge(workstream.state));

  const summary = document.createElement("p");
  summary.textContent = workstream.summary;

  const badges = document.createElement("div");
  badges.className = "badges";
  badges.append(badge(workstream.coverage));

  article.append(heading, summary, badges);
  if (Number.isInteger(workstream.progress_percent)) {
    article.append(renderProgress(workstream.progress_percent));
  }
  const tracking = renderTrackingDetails(workstream);
  if (tracking) {
    article.append(tracking);
  }
  return article;
}

function publicSource(value) {
  if (!value) {
    return null;
  }
  try {
    const url = new URL(value);
    return ["http:", "https:"].includes(url.protocol) ? url.href : null;
  } catch {
    return null;
  }
}

function renderEvent(event) {
  const article = document.createElement("article");
  article.className = "event";

  const metadata = document.createElement("div");
  metadata.className = "event-meta";
  const metadataItems = [
    event.workstream,
    event.kind,
    formatTime(event.occurred_at),
  ];
  if (Number.isInteger(event.progress_percent)) {
    metadataItems.push(`${event.progress_percent}%`);
  }
  metadata.textContent = metadataItems.join(" · ");

  const title = document.createElement("h3");
  title.textContent = event.title;

  const summary = document.createElement("p");
  summary.textContent = event.summary;

  const evidence = document.createElement("p");
  evidence.className = "evidence";
  evidence.textContent = `Evidence: ${event.evidence}`;

  const source = publicSource(event.source_url);
  if (source) {
    evidence.append(" ");
    const link = document.createElement("a");
    link.href = source;
    link.textContent = "source";
    link.rel = "noreferrer";
    evidence.append(link);
  }

  article.append(metadata, title, summary, evidence);
  const tracking = renderTrackingDetails(event);
  if (tracking) {
    article.append(tracking);
  }
  return article;
}

function render(status) {
  const connection = byId("connection");
  connection.textContent = "status available";
  connection.classList.remove("error");
  connection.classList.add("online");

  byId("claim").textContent = status.claim_scope;
  byId("project-summary").textContent = status.summary;
  byId("project-state").textContent = status.state.replaceAll("-", " ");
  byId("updated").textContent = formatTime(status.updated_at);
  byId("digest").textContent = `${status.trace.sha256.slice(0, 12)}…`;
  byId("refresh-note").textContent = `Updated ${formatTime(status.updated_at)}`;

  const traceUrl = new URL(status.trace.url, window.location.origin);
  byId("perfetto").href = `https://ui.perfetto.dev/#!/?url=${encodeURIComponent(traceUrl.href)}`;

  byId("workstreams").replaceChildren(
    ...status.workstreams.map(renderWorkstream),
  );
  byId("events").replaceChildren(...status.events.map(renderEvent));
}

function retryDelay(status) {
  const seconds = Number(status?.refresh_seconds);
  if (!Number.isFinite(seconds)) {
    return 30_000;
  }
  return Math.max(10, Math.min(300, seconds)) * 1000;
}

async function refresh() {
  let status;
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    status = await response.json();
    render(status);
  } catch (error) {
    const connection = byId("connection");
    connection.textContent = "status unavailable";
    connection.classList.remove("online");
    connection.classList.add("error");
    byId("refresh-note").textContent = String(error);
  }
  window.setTimeout(refresh, retryDelay(status));
}

refresh();
