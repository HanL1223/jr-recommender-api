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
    --sage: #7d9a78;
    --discovery: #6b8e7d;
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
    max-width: 1000px;
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

  /* Split sections */
  .recommendations-split {
    display: grid;
    grid-template-columns: 1fr 1.5fr;
    gap: 2rem;
  }

  @media (max-width: 768px) {
    .recommendations-split {
      grid-template-columns: 1fr;
    }
  }

  .section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
  }

  .section-header h3 {
    font-family: 'Fraunces', serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--espresso);
  }

  .section-header .icon {
    font-size: 1.25rem;
  }

  .section-badge {
    font-size: 0.65rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.2rem 0.5rem;
    border-radius: 0.25rem;
  }

  .badge-favorites {
    background: rgba(196, 163, 90, 0.2);
    color: var(--mocha);
  }

  .badge-discovery {
    background: rgba(107, 142, 125, 0.2);
    color: var(--discovery);
  }

  .favorites-section {
    padding: 1.5rem;
    background: linear-gradient(135deg, rgba(196, 163, 90, 0.08), rgba(196, 163, 90, 0.02));
    border-radius: 1rem;
    border: 1px solid rgba(196, 163, 90, 0.15);
  }

  .discovery-section {
    padding: 1.5rem;
    background: linear-gradient(135deg, rgba(107, 142, 125, 0.08), rgba(107, 142, 125, 0.02));
    border-radius: 1rem;
    border: 1px solid rgba(107, 142, 125, 0.15);
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1rem;
  }

  .grid-small {
    grid-template-columns: 1fr;
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

  .product-card-compact {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem;
    background: white;
    border-radius: 0.75rem;
  }

  .product-card-compact .emoji {
    font-size: 2rem;
    flex-shrink: 0;
  }

  .product-card-compact .product-info {
    padding: 0;
  }

  .product-img {
    height: 100px;
    background: linear-gradient(135deg, var(--caramel) 0%, var(--mocha) 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.5rem;
  }

  .product-img.discovery-bg {
    background: linear-gradient(135deg, var(--sage) 0%, var(--discovery) 100%);
  }

  .product-info { padding: 1rem; }

  .product-name {
    font-family: 'Fraunces', serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--espresso);
    margin-bottom: 0.375rem;
  }

  .product-meta {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
    flex-wrap: wrap;
  }

  .score {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 500;
    padding: 0.2rem 0.5rem;
    background: rgba(196, 163, 90, 0.15);
    color: var(--mocha);
    border-radius: 1rem;
  }

  .category-tag {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 500;
    padding: 0.15rem 0.4rem;
    background: rgba(44, 24, 16, 0.06);
    color: var(--mocha);
    border-radius: 0.25rem;
  }

  .new-tag {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 0.15rem 0.4rem;
    background: var(--discovery);
    color: white;
    border-radius: 0.25rem;
  }

  .reason {
    font-size: 0.8rem;
    color: var(--mocha);
    opacity: 0.8;
    line-height: 1.4;
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

  .excluded-info {
    margin-top: 1rem;
    padding: 0.75rem 1rem;
    background: rgba(44, 24, 16, 0.03);
    border-radius: 0.5rem;
    font-size: 0.75rem;
    color: var(--mocha);
    opacity: 0.7;
  }

  .excluded-info strong {
    font-weight: 500;
  }

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

  .empty-state {
    text-align: center;
    padding: 1.5rem;
    opacity: 0.6;
    font-size: 0.875rem;
  }
`;

export default function App() {
  const [customerId, setCustomerId] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const API = import.meta.env.VITE_API_BASE || "http://localhost:8000";

  async function fetchRecommendations(isNewCustomer = false) {
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      let url;
      if (isNewCustomer) {
        // Cold start - use popular endpoint
        url = `${API}/popular?top_k=5`;
      } else {
        // Use the split endpoint with 2 favorites + 3 discovery
        url = `${API}/recommend/${customerId}/split?n_favorites=2&n_discovery=3&n_addons=2`;
      }

      const res = await fetch(url);
      
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }
      
      const data = await res.json();
      setResult({ ...data, isNewCustomer, isSplit: !isNewCustomer });
    } catch (err) {
      setError(err.message);
    }

    setLoading(false);
  }

  const getEmoji = (name) => {
    const lower = (name || "").toLowerCase();
    if (lower.includes("latte")) return "☕";
    if (lower.includes("cappuccino")) return "☕";
    if (lower.includes("flat white")) return "☕";
    if (lower.includes("mocha")) return "☕";
    if (lower.includes("coffee") || lower.includes("espresso")) return "☕";
    if (lower.includes("americano")) return "☕";
    if (lower.includes("tea") || lower.includes("chai")) return "🍵";
    if (lower.includes("hot chocolate")) return "🍫";
    if (lower.includes("cake") || lower.includes("brownie")) return "🍰";
    if (lower.includes("muffin")) return "🧁";
    if (lower.includes("croissant")) return "🥐";
    if (lower.includes("toast") || lower.includes("bread") || lower.includes("banana bread")) return "🍞";
    if (lower.includes("sandwich") || lower.includes("toastie") || lower.includes("panini")) return "🥪";
    if (lower.includes("wrap")) return "🌯";
    if (lower.includes("juice") || lower.includes("smoothie")) return "🥤";
    if (lower.includes("milkshake") || lower.includes("frappe")) return "🥛";
    if (lower.includes("chicken") || lower.includes("bao")) return "🍗";
    if (lower.includes("fry") || lower.includes("wedge") || lower.includes("chip")) return "🍟";
    if (lower.includes("egg") || lower.includes("omelette")) return "🍳";
    if (lower.includes("salad") || lower.includes("bowl")) return "🥗";
    if (lower.includes("cookie") || lower.includes("biscuit")) return "🍪";
    if (lower.includes("babychino") || lower.includes("babycino")) return "🍼";
    return "✨";
  };

  const renderSplitResults = () => {
    const { favorites = [], discovery = [], addon_items = [], excluded_categories = [] } = result;

    return (
      <>
        <div className="recommendations-split">
          {/* Favorites Section */}
          <div className="favorites-section">
            <div className="section-header">
              <span className="icon">❤️</span>
              <h3>Your Usuals</h3>
              <span className="section-badge badge-favorites">{favorites.length} items</span>
            </div>
            
            {favorites.length > 0 ? (
              <div className="grid grid-small">
                {favorites.map((item, idx) => (
                  <div key={idx} className="product-card-compact">
                    <span className="emoji">{getEmoji(item.product)}</span>
                    <div className="product-info">
                      <div className="product-name">{item.product}</div>
                      <div className="product-meta">
                        <span className="score">{(item.score * 100).toFixed(0)}%</span>
                        {item.category && <span className="category-tag">{item.category}</span>}
                      </div>
                      <div className="reason">{item.reason}</div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-state">No previous purchases found</div>
            )}
          </div>

          {/* Discovery Section */}
          <div className="discovery-section">
            <div className="section-header">
              <span className="icon">✨</span>
              <h3>Try Something New</h3>
              <span className="section-badge badge-discovery">{discovery.length} items</span>
            </div>
            
            {discovery.length > 0 ? (
              <div className="grid">
                {discovery.map((item, idx) => (
                  <div key={idx} className="product-card">
                    <div className="product-img discovery-bg">{getEmoji(item.product)}</div>
                    <div className="product-info">
                      <div className="product-name">{item.product}</div>
                      <div className="product-meta">
                        <span className="score">{(item.score * 100).toFixed(0)}%</span>
                        <span className="new-tag">NEW</span>
                        {item.category && <span className="category-tag">{item.category}</span>}
                      </div>
                      <div className="reason">{item.reason}</div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-state">No new recommendations available</div>
            )}

            {excluded_categories.length > 0 && (
              <div className="excluded-info">
                <strong>Note:</strong> We filtered out {excluded_categories.join(", ")} items since you already have those in your usuals.
              </div>
            )}
          </div>
        </div>

        {/* Add-ons */}
        {addon_items.length > 0 && (
          <div className="addon-section">
            <h3>🥐 Complete Your Order</h3>
            <div className="addon-grid">
              {addon_items.map((item, idx) => (
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
      </>
    );
  };

  const renderPopularResults = () => {
    const { primary_items = [], addon_items = [] } = result;

    return (
      <>
        <div className="cold-start-banner">
          <span>👋</span>
          <div>
            <strong>Welcome!</strong> Here are our customer favorites to get you started.
          </div>
        </div>

        {primary_items.length > 0 ? (
          <div className="grid">
            {primary_items.map((item, idx) => (
              <div key={idx} className="product-card">
                <div className="product-img">{getEmoji(item.product)}</div>
                <div className="product-info">
                  <div className="product-name">{item.product}</div>
                  <div className="product-meta">
                    <span className="score">{(item.score * 100).toFixed(0)}%</span>
                    {item.category && <span className="category-tag">{item.category}</span>}
                  </div>
                  <div className="reason">{item.reason}</div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-state">No recommendations found</div>
        )}

        {addon_items.length > 0 && (
          <div className="addon-section">
            <h3>You might also like</h3>
            <div className="addon-grid">
              {addon_items.map((item, idx) => (
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
      </>
    );
  };

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
                  onKeyDown={(e) => e.key === "Enter" && customerId && fetchRecommendations(false)}
                />
              </div>
            </div>

            <div className="btn-group">
              <button
                className="btn"
                onClick={() => fetchRecommendations(false)}
                disabled={loading || !customerId}
              >
                {loading ? "Brewing…" : "Get My Recommendations"}
              </button>

              <span className="divider">or</span>

              <button
                className="btn btn-secondary"
                onClick={() => fetchRecommendations(true)}
                disabled={loading}
              >
                👋 New here? See popular picks
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
                {result.isNewCustomer ? "Our Most Popular Items" : `Recommendations for #${result.customer_id}`}
                <span className="model-badge">{result.model_used}</span>
              </h2>

              {result.isSplit ? renderSplitResults() : renderPopularResults()}

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