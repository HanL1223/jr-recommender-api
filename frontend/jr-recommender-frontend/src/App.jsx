import React, { useState } from "react";

const styles = `
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=DM+Sans:wght@400;500&display=swap');

  * { margin: 0; padding: 0; box-sizing: border-box; }

  :root {
    --espresso: #2c1810;
    --latte: #f5ebe0;
    --cream: #fdfbf7;
    --caramel: #c4a35a;
    --mocha: #5c4033;
  }

  html, body, #root { min-height: 100vh; width: 100%; }

  body {
    font-family: 'DM Sans', sans-serif;
    background: var(--latte);
    color: var(--espresso);
  }

  .page {
    min-height: 100vh;
    width: 100%;
    background: 
      radial-gradient(ellipse at 20% 0%, rgba(196, 163, 90, 0.15) 0%, transparent 50%),
      radial-gradient(ellipse at 80% 100%, rgba(92, 64, 51, 0.1) 0%, transparent 50%),
      var(--latte);
  }

  .topbar {
    width: 100%;
    padding: 1.5rem 2rem;
    display: flex;
    align-items: center;
    justify-content: center;
    border-bottom: 1px solid rgba(44, 24, 16, 0.08);
    background: var(--cream);
  }

  .brand {
    font-family: 'Fraunces', serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--espresso);
  }

  .brand::before { content: '☕'; margin-right: 0.5rem; }

  .container {
    width: 100%;
    max-width: 900px;
    margin: 0 auto;
    padding: 3rem 1.5rem;
  }

  .card {
    background: var(--cream);
    border-radius: 1.25rem;
    padding: 2rem;
    box-shadow: 0 1px 2px rgba(44, 24, 16, 0.04), 0 4px 16px rgba(44, 24, 16, 0.06);
    margin-bottom: 2rem;
  }

  .card h2 {
    font-family: 'Fraunces', serif;
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 1.5rem;
    color: var(--mocha);
  }

  .search-box { display: flex; flex-direction: column; gap: 1rem; }
  .input-group { display: flex; gap: 1rem; flex-wrap: wrap; }
  .input-wrapper { flex: 1; min-width: 200px; }

  .input-wrapper label {
    display: block;
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--mocha);
    margin-bottom: 0.5rem;
    opacity: 0.7;
  }

  .input {
    width: 100%;
    padding: 0.875rem 1rem;
    font-size: 1rem;
    font-family: 'DM Sans', sans-serif;
    border: 1.5px solid rgba(44, 24, 16, 0.12);
    border-radius: 0.75rem;
    background: white;
    color: var(--espresso);
    transition: border-color 0.2s, box-shadow 0.2s;
  }

  .input:focus {
    outline: none;
    border-color: var(--caramel);
    box-shadow: 0 0 0 3px rgba(196, 163, 90, 0.15);
  }

  .btn-group { display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center; }

  .btn {
    padding: 1rem 2rem;
    font-size: 0.9375rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    border: none;
    border-radius: 0.75rem;
    background: var(--espresso);
    color: var(--cream);
    cursor: pointer;
    transition: transform 0.15s, box-shadow 0.15s, background 0.2s;
  }

  .btn:hover {
    background: var(--mocha);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(44, 24, 16, 0.2);
  }

  .btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }

  .btn-secondary {
    background: transparent;
    color: var(--mocha);
    border: 1.5px solid rgba(44, 24, 16, 0.2);
  }

  .btn-secondary:hover {
    background: rgba(44, 24, 16, 0.05);
    border-color: rgba(44, 24, 16, 0.3);
  }

  .divider { color: var(--mocha); opacity: 0.4; font-size: 0.875rem; }

  .error-banner {
    padding: 1rem;
    background: #fee;
    border: 1px solid #fcc;
    border-radius: 0.5rem;
    color: #c00;
    font-size: 0.875rem;
  }

  .cold-start-banner {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 1rem 1.25rem;
    background: linear-gradient(135deg, rgba(196, 163, 90, 0.15), rgba(196, 163, 90, 0.05));
    border: 1px solid rgba(196, 163, 90, 0.3);
    border-radius: 0.75rem;
    margin-bottom: 1.5rem;
    font-size: 0.9rem;
    color: var(--mocha);
  }

  .cold-start-banner span { font-size: 1.25rem; }

  .model-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 500;
    text-transform: uppercase;
    padding: 0.25rem 0.5rem;
    background: rgba(44, 24, 16, 0.08);
    color: var(--mocha);
    border-radius: 0.25rem;
    margin-left: 0.75rem;
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 1.25rem;
  }

  .product-card {
    background: white;
    border-radius: 1rem;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
    animation: fadeUp 0.4s ease-out backwards;
  }

  .product-card:nth-child(1) { animation-delay: 0.05s; }
  .product-card:nth-child(2) { animation-delay: 0.1s; }
  .product-card:nth-child(3) { animation-delay: 0.15s; }
  .product-card:nth-child(4) { animation-delay: 0.2s; }
  .product-card:nth-child(5) { animation-delay: 0.25s; }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .product-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(44, 24, 16, 0.12);
  }

  .product-img {
    height: 120px;
    background: linear-gradient(135deg, var(--caramel) 0%, var(--mocha) 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.5rem;
  }

  .product-info { padding: 1.25rem; }

  .product-name {
    font-family: 'Fraunces', serif;
    font-size: 1.0625rem;
    font-weight: 600;
    color: var(--espresso);
    margin-bottom: 0.5rem;
  }

  .score {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 0.25rem 0.625rem;
    background: rgba(196, 163, 90, 0.15);
    color: var(--mocha);
    border-radius: 1rem;
    margin-bottom: 0.75rem;
  }

  .reason {
    font-size: 0.875rem;
    color: var(--mocha);
    opacity: 0.8;
    line-height: 1.5;
  }

  .addon-section {
    margin-top: 2rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(44, 24, 16, 0.08);
  }

  .addon-section h3 {
    font-family: 'Fraunces', serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--mocha);
    margin-bottom: 1rem;
  }

  .addon-grid { display: flex; gap: 1rem; flex-wrap: wrap; }

  .addon-card {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    background: rgba(196, 163, 90, 0.1);
    border-radius: 0.75rem;
    font-size: 0.9rem;
  }

  .addon-card .emoji { font-size: 1.25rem; }

  .json-box {
    margin-top: 1.5rem;
    border-top: 1px solid rgba(44, 24, 16, 0.08);
    padding-top: 1.25rem;
  }

  .json-box summary {
    font-size: 0.8125rem;
    font-weight: 500;
    color: var(--mocha);
    cursor: pointer;
    opacity: 0.7;
  }

  .json-box pre {
    margin-top: 1rem;
    padding: 1rem;
    background: var(--espresso);
    color: var(--latte);
    border-radius: 0.75rem;
    font-size: 0.75rem;
    overflow-x: auto;
  }
`;

export default function App() {
  const [customerId, setCustomerId] = useState("");
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function fetchRecommendations(isNewCustomer = false) {
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const API = import.meta.env.VITE_API_BASE;

      const url = isNewCustomer
        ? `${API}/popular?top_k=${topK}`
        : `${API}/recommend/${customerId}?top_k=${topK}`;

      const res = await fetch(url);
      
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }
      
      const data = await res.json();
      setResult({ ...data, isNewCustomer });
    } catch (err) {
      setError(err.message);
    }

    setLoading(false);
  }

  const getEmoji = (name) => {
    const lower = (name || "").toLowerCase();
    if (lower.includes("coffee") || lower.includes("espresso") || lower.includes("latte")) return "☕";
    if (lower.includes("tea")) return "🍵";
    if (lower.includes("cake") || lower.includes("pastry")) return "🍰";
    if (lower.includes("croissant")) return "🥐";
    if (lower.includes("toast") || lower.includes("bread")) return "🍞";
    if (lower.includes("sandwich") || lower.includes("toastie")) return "🥪";
    if (lower.includes("juice") || lower.includes("smoothie")) return "🥤";
    if (lower.includes("chicken") || lower.includes("bao")) return "🍗";
    if (lower.includes("fry") || lower.includes("wedge") || lower.includes("chip")) return "🍟";
    if (lower.includes("egg")) return "🍳";
    if (lower.includes("salad")) return "🥗";
    return "✨";
  };

  const isColdStart = result?.model_used === "ColdStart" || result?.customer_id === null;

  return (
    <>
      <style>{styles}</style>
      <div className="page">
        <header className="topbar">
          <div className="brand">JR Café</div>
        </header>

        <main className="container">
          <section className="card search-box">
            <h2>Order Recommender</h2>

            <div className="input-group">
              <div className="input-wrapper">
                <label>Membership Number</label>
                <input
                  className="input"
                  placeholder="e.g. 123"
                  value={customerId}
                  onChange={(e) => setCustomerId(e.target.value)}
                />
              </div>

              <div className="input-wrapper" style={{ maxWidth: "140px" }}>
                <label>No. of items</label>
                <input
                  className="input"
                  type="number"
                  min="1"
                  max="20"
                  value={topK}
                  onChange={(e) => setTopK(e.target.value)}
                />
              </div>
            </div>

            <div className="btn-group">
              <button
                className="btn"
                onClick={() => fetchRecommendations(false)}
                disabled={loading || !customerId}
              >
                {loading ? "Brewing…" : "Get Recommendations"}
              </button>

              <span className="divider">or</span>

              <button
                className="btn btn-secondary"
                onClick={() => fetchRecommendations(true)}
                disabled={loading}
              >
                👋 New here? See popular picks 123
              </button>
            </div>

            {error && (
              <div className="error-banner">
                Error: {error}
              </div>
            )}
          </section>

          {result && (
            <section className="card">
              <h2>
                {result.isNewCustomer ? "Our Most Popular Items" : "Recommended For You"}
                <span className="model-badge">{result.model_used}</span>
              </h2>

              {result.isNewCustomer && (
                <div className="cold-start-banner">
                  <span>👋</span>
                  <div>
                    <strong>Welcome!</strong> Here are our customer favorites to get you started.
                  </div>
                </div>
              )}

              {isColdStart && !result.isNewCustomer && (
                <div className="cold-start-banner">
                  <span>🔎</span>
                  <div>
                    <strong>Customer not found.</strong> Showing popular items instead.
                  </div>
                </div>
              )}

              {result.primary_items?.length > 0 ? (
                <div className="grid">
                  {result.primary_items.map((item, idx) => (
                    <div key={idx} className="product-card">
                      <div className="product-img">{getEmoji(item.product)}</div>
                      <div className="product-info">
                        <div className="product-name">{item.product}</div>
                        <div className="score">Score: {item.score.toFixed(2)}</div>
                        <div className="reason">{item.reason}</div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ textAlign: "center", padding: "2rem", opacity: 0.6 }}>
                  No recommendations found
                </div>
              )}

              {result.addon_items?.length > 0 && (
                <div className="addon-section">
                  <h3>You might also like</h3>
                  <div className="addon-grid">
                    {result.addon_items.map((item, idx) => (
                      <div key={idx} className="addon-card">
                        <span className="emoji">{getEmoji(item.product)}</span>
                        <div>
                          <strong>{item.product}</strong>
                          <div style={{ fontSize: "0.75rem", opacity: 0.7 }}>{item.reason}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <details className="json-box">
                <summary>Show Raw API Output</summary>
                <pre>{JSON.stringify(result, null, 2)}</pre>
              </details>
            </section>
          )}
        </main>
      </div>
    </>
  );
}