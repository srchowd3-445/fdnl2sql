//frontend/src/components/HeroBlock.jsx
export default function HeroBlock() {
  return (
    <section className="hero-block">
      <div className="hero-block-inner">
        <div className="hero-block-badge">
          {/* TODO: Replace badge label */}
          Living Systematic Review
        </div>
        {/* TODO: Replace with your actual heading */}
        <h2 className="hero-block-heading">Test Heading: About This Database</h2>
        <div className="hero-block-body">
          <p>
            FD-NL2SQL is a natural-language-to-SQL assistant for oncology clinical-trial databases that
            leverages the power of large language models to make complex trial querying fast, flexible,
            and accessible.
          </p>
          <p>
            It enables ad-hoc, multi-constraint questions across biomarkers, endpoints, interventions,
            and time, translating everyday questions into executable database queries without requiring SQL
            or schema expertise. Designed for iterative use, it supports easy refinement and learns from
            feedback so it improves over time.
          </p>
        </div>
        <div className="hero-block-pills">
          {/* TODO: Replace with your actual keywords/scope tags */}
          <span className="hero-pill">Phase 2 &amp; 3 RCTs</span>
          <span className="hero-pill">PD-1 / PD-L1 / CTLA-4</span>
          <span className="hero-pill">19 Cancer Types</span>
          <span className="hero-pill">Peer-reviewed Sources</span>
        </div>
      </div>
    </section>
  )
}
