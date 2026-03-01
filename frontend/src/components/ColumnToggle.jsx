//frontend/src/components/ColumnToggle.jsx
import { useState } from 'react'

const COLUMN_GROUPS = {
  'TRIAL CHARACTERISTICS': [
    'NCT', 'PubMed ID', 'Trial name', 'Author', 'Year',
    'Originial publication or Follow-up', 'Trial phase', 'Number of arms',
    'Monotherapy/combination', 'Type of combination', 'Control regimen',
    'Type of control', 'Treatment regimen'
  ],
  'POPULATION CHARACTERISTICS': [
    'Total sample size', 'Lines of treatment',
    'Clincal setting in relation to surgery',
    'Is PD-L1 positivity inclusion criteria',
    'Is any other biomarker used for inclusion'
  ],
  'ICI DETAILS': [
    'Name of ICI', 'Class of ICI', 'Type of therapy', 'Control arm', 'Clinical setting'
  ],
  'RESULTS': [
    'Primary endpoint', 'Priamry Multiple, composite, or co-primary endpoints?',
    'Secondary endpoint', 'Type of follow-up given',
    'Follow-up duration for primary endpoint(s) in months | Overall',
    'Follow-up duration for primary endpoint(s) in months | Rx',
    'Follow-up duration for primary endpoint(s) in months | Control',
    'Included in MA'
  ],
}

export default function ColumnToggle({ allColumns, visibleColumns, setVisibleColumns, chatOpen, onToggleChat }) {
  const [open, setOpen] = useState(false)

  const toggle = (col) => {
    setVisibleColumns(prev => ({ ...prev, [col]: !prev[col] }))
  }

  const checkAll = () => {
    setVisibleColumns(prev => Object.fromEntries(Object.keys(prev).map(k => [k, true])))
  }

  const resetAll = () => {
    const defaults = ['NCT', 'PubMed ID', 'Trial name', 'Author', 'Year',
      'Originial publication or Follow-up', 'Cancer type', 'Name of ICI', 'Class of ICI']
    setVisibleColumns(prev => Object.fromEntries(Object.keys(prev).map(k => [k, defaults.includes(k)])))
  }

  return (
    <div className="column-toggle-bar">
      <button className="columns-btn" onClick={() => setOpen(!open)}>
        <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
          <rect x="0" y="0" width="4" height="14" rx="1"/>
          <rect x="5" y="0" width="4" height="14" rx="1"/>
          <rect x="10" y="0" width="4" height="14" rx="1"/>
        </svg>
        Columns
        <svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor" style={{marginLeft:4}}>
          <path d={open ? "M2 7l3-3 3 3" : "M2 3l3 3 3-3"}/>
        </svg>
      </button>
      <button className="columns-btn secondary" onClick={checkAll}>Check All</button>
      <button className="columns-btn secondary" onClick={resetAll}>Reset All</button>

      {/* Chatbot toggle button — blue when open, outlined when closed */}
      <button
        className={`columns-btn chatbot-toggle-btn ${chatOpen ? 'chat-active' : 'secondary'}`}
        onClick={onToggleChat}
        title={chatOpen ? 'Close assistant' : 'Open assistant'}
      >
        <svg width="15" height="15" viewBox="0 0 16 16" fill="currentColor">
          <path d="M2 2h12a1 1 0 011 1v7a1 1 0 01-1 1H9l-3 3v-3H2a1 1 0 01-1-1V3a1 1 0 011-1z"/>
        </svg>
        Assistant
        {chatOpen && (
          <span className="chat-open-dot" />
        )}
      </button>

      {open && (
        <>
          <div className="columns-overlay" onClick={() => setOpen(false)} />
          <div className="columns-dropdown">
            <div className="columns-dropdown-header">
              <span>Toggle Columns</span>
              <button onClick={() => setOpen(false)} className="close-btn">✕</button>
            </div>
            <div className="columns-groups">
              {Object.entries(COLUMN_GROUPS).map(([group, cols]) => (
                <div key={group} className="column-group">
                  <div className="column-group-title">{group}</div>
                  <div className="column-checkboxes">
                    {cols.filter(c => allColumns.includes(c)).map(col => (
                      <label key={col} className="col-check-item">
                        <input
                          type="checkbox"
                          checked={!!visibleColumns[col]}
                          onChange={() => toggle(col)}
                        />
                        <span className="col-check-custom" />
                        <span>{col}</span>
                      </label>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}