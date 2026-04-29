const API_BASE = "http://localhost:8000";

export interface RecommendResponse {
  response: string;
  error: string | null;
}

export async function postRecommend(userInput: string): Promise<RecommendResponse> {
  const res = await fetch(`${API_BASE}/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_input: userInput }),
  });

  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  }

  return res.json() as Promise<RecommendResponse>;
}
