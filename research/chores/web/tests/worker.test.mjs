import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import worker from "../src/worker.js";

function environment() {
  const requests = [];
  return {
    requests,
    env: {
      ASSETS: {
        async fetch(request) {
          requests.push({
            method: request.method,
            pathname: new URL(request.url).pathname,
          });
          return new Response("asset", {
            headers: { "Content-Type": "text/plain" },
          });
        },
      },
    },
  };
}

test("landing page states the agent security purpose", async () => {
  const index = await readFile(
    new URL("../public/index.html", import.meta.url),
    "utf8",
  );

  assert.match(index, /Proactive security hygiene/);
  assert.match(index, /must immediately report suspected unauthorized/);
  assert.match(index, /does not monitor agents or deliver incident reports/);
});

test("status route maps to the generated document", async () => {
  const { env, requests } = environment();
  const response = await worker.fetch(
    new Request("https://chores.knlp.io/api/status"),
    env,
  );

  assert.equal(response.status, 200);
  assert.deepEqual(requests, [{ method: "GET", pathname: "/status.json" }]);
  assert.equal(response.headers.get("Cache-Control"), "no-store");
  assert.equal(
    response.headers.get("Content-Type"),
    "application/json; charset=utf-8",
  );
  assert.equal(response.headers.get("X-Content-Type-Options"), "nosniff");
});

test("trace route is readable by Perfetto", async () => {
  const { env, requests } = environment();
  const response = await worker.fetch(
    new Request("https://chores.knlp.io/traces/latest.pftrace"),
    env,
  );

  assert.equal(response.status, 200);
  assert.deepEqual(requests, [
    { method: "GET", pathname: "/traces/latest.pftrace" },
  ]);
  assert.equal(response.headers.get("Access-Control-Allow-Origin"), "*");
  assert.equal(response.headers.get("Content-Type"), "application/octet-stream");
  assert.match(response.headers.get("Content-Disposition"), /\.pftrace$/);
});

test("ordinary assets retain the security policy", async () => {
  const { env, requests } = environment();
  const response = await worker.fetch(
    new Request("https://chores.knlp.io/"),
    env,
  );

  assert.equal(response.status, 200);
  assert.deepEqual(requests, [{ method: "GET", pathname: "/" }]);
  assert.equal(response.headers.get("Cache-Control"), "no-cache");
  assert.match(response.headers.get("Content-Security-Policy"), /frame-ancestors/);
  assert.doesNotMatch(
    response.headers.get("Content-Security-Policy"),
    /unsafe-inline/,
  );
});

test("HEAD returns headers without an asset body", async () => {
  const { env, requests } = environment();
  const response = await worker.fetch(
    new Request("https://chores.knlp.io/api/status", { method: "HEAD" }),
    env,
  );

  assert.deepEqual(requests, [{ method: "HEAD", pathname: "/status.json" }]);
  assert.equal(response.headers.get("Cache-Control"), "no-store");
  assert.equal(await response.text(), "");
});

test("mutating methods are rejected before asset lookup", async () => {
  const { env, requests } = environment();
  const response = await worker.fetch(
    new Request("https://chores.knlp.io/api/status", { method: "POST" }),
    env,
  );

  assert.equal(response.status, 405);
  assert.deepEqual(requests, []);
  assert.equal(response.headers.get("Allow"), "GET, HEAD");
});
