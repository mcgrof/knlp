const SECURITY_HEADERS = {
  "Content-Security-Policy": [
    "default-src 'self'",
    "connect-src 'self'",
    "img-src 'self' data:",
    "style-src 'self'",
    "script-src 'self'",
    "base-uri 'none'",
    "form-action 'none'",
    "frame-ancestors 'none'",
  ].join("; "),
  "Referrer-Policy": "no-referrer",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
  "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
};

function withHeaders(request, response, extra = {}) {
  const headers = new Headers(response.headers);
  for (const [name, value] of Object.entries(SECURITY_HEADERS)) {
    headers.set(name, value);
  }
  for (const [name, value] of Object.entries(extra)) {
    headers.set(name, value);
  }
  return new Response(request.method === "HEAD" ? null : response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

async function asset(env, request, pathname) {
  const url = new URL(request.url);
  url.pathname = pathname;
  url.search = "";
  return env.ASSETS.fetch(new Request(url, request));
}

export default {
  async fetch(request, env) {
    if (request.method !== "GET" && request.method !== "HEAD") {
      return withHeaders(
        request,
        new Response("method not allowed\n", { status: 405 }),
        { Allow: "GET, HEAD" },
      );
    }

    const url = new URL(request.url);
    if (url.pathname === "/api/status") {
      const response = await asset(env, request, "/status.json");
      return withHeaders(request, response, {
        "Cache-Control": "no-store",
        "Content-Type": "application/json; charset=utf-8",
      });
    }

    if (url.pathname === "/traces/latest.pftrace") {
      const response = await asset(env, request, url.pathname);
      return withHeaders(request, response, {
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "public, max-age=60",
        "Content-Disposition": "inline; filename=knlp-chores-public.pftrace",
        "Content-Type": "application/octet-stream",
      });
    }

    const response = await env.ASSETS.fetch(request);
    const cacheControl = ["/", "/index.html"].includes(url.pathname)
      ? "no-cache"
      : "public, max-age=300";
    return withHeaders(request, response, { "Cache-Control": cacheControl });
  },
};
