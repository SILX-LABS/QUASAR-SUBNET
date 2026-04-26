import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || process.env.API_URL || "http://localhost:8000";

async function proxy(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params;
  const upstream = new URL(`/api/${path.join("/")}`, API_BASE);
  upstream.search = request.nextUrl.search;

  const headers = new Headers();
  const accept = request.headers.get("accept");
  const contentType = request.headers.get("content-type");
  if (accept) headers.set("accept", accept);
  if (contentType) headers.set("content-type", contentType);

  // IMPORTANT: forward the client's AbortSignal so that when the browser
  // disconnects, we tear down the upstream fetch too. Without this, long-lived
  // streams (e.g. /api/eval-stream SSE) leak tasks on the Python API until it
  // hits --limit-concurrency and starts returning 503 to everyone.
  let response: Response;
  try {
    response = await fetch(upstream, {
      method: request.method,
      headers,
      body: request.method === "GET" || request.method === "HEAD" ? undefined : await request.text(),
      cache: "no-store",
      signal: request.signal,
    });
  } catch {
    return NextResponse.json({ error: "upstream unavailable" }, { status: 502 });
  }

  const outHeaders = new Headers();
  for (const key of ["content-type", "cache-control"]) {
    const value = response.headers.get(key);
    if (value) outHeaders.set(key, value);
  }
  return new NextResponse(response.body, { status: response.status, headers: outHeaders });
}

export async function GET(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxy(request, context);
}

export async function POST(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxy(request, context);
}
