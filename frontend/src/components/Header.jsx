//frontend/src/components/Header.jsx
export default function Header() {
  return (
    <header className="site-header">
      <div className="header-top">
        <div className="header-brand">
          <div className="header-logo">
            <svg width="44" height="44" viewBox="0 0 44 44" fill="none">
              <rect width="44" height="44" rx="6" fill="white"/>
              <path d="M22 7L36 34H8L22 7Z" fill="#003087"/>
              <circle cx="22" cy="28" r="4" fill="#CC3333"/>
            </svg>
          </div>
          <div className="header-title-group">
            <h1 className="site-name">FD-NL2SQL</h1>
            <p className="site-tagline">Enhanced Oncology Research</p>
          </div>
        </div>
      </div>

      <div className="header-banner">
        <div className="banner-inner">
          <div className="banner-stat">
            <span className="stat-number">159</span>
            <span className="stat-label">Clinical Trials</span>
          </div>
          <div className="banner-divider" />
          <div className="banner-stat">
            <span className="stat-number">13</span>
            <span className="stat-label">ICI Agents</span>
          </div>
          <div className="banner-divider" />
          <div className="banner-stat">
            <span className="stat-number">19</span>
            <span className="stat-label">Cancer Types</span>
          </div>
          <div className="banner-divider" />
          <div className="banner-stat">
            <span className="stat-number">Living</span>
            <span className="stat-label">Review</span>
          </div>
          <div className="banner-message">
            <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
              <path d="M8 1a7 7 0 110 14A7 7 0 018 1zm0 1.5a5.5 5.5 0 100 11 5.5 5.5 0 000-11zM8 11a1 1 0 110 2 1 1 0 010-2zm.75-5.5v4h-1.5V5.5h1.5z"/>
            </svg>
            test-project · continuously updated
          </div>
        </div>
      </div>
    </header>
  )
}