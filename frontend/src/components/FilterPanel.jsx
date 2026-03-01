//frontend/src/components/FillerPannel.jsx
import { useState } from 'react'

function RadioGroup({ label, name, options, value, onChange }) {
  return (
    <div className="filter-group">
      <div className="filter-label">{label}</div>
      <div className="radio-list">
        {options.map(opt => (
          <label key={opt} className={`radio-item ${value === opt ? 'active' : ''}`}>
            <input type="radio" name={name} value={opt} checked={value === opt} onChange={() => onChange(opt)} />
            <span className="radio-custom" />
            <span className="radio-text">{opt}</span>
          </label>
        ))}
      </div>
    </div>
  )
}

function SelectFilter({ label, options, value, onChange }) {
  return (
    <div className="filter-group">
      <div className="filter-label">{label}</div>
      <div className="select-wrapper">
        <select value={value} onChange={e => onChange(e.target.value)}>
          {options.map(opt => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
        <svg className="select-arrow" width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
          <path d="M2 4l4 4 4-4"/>
        </svg>
      </div>
    </div>
  )
}

export default function FilterPanel({ filters, setFilters, filterOptions, onReset }) {
  const [mobileOpen, setMobileOpen] = useState(false)
  const update = (key) => (val) => setFilters(prev => ({ ...prev, [key]: val }))

  const activeCount = Object.values(filters).filter(v => v !== 'All').length

  const content = (
    <>
      <RadioGroup label="ICI Class" name="iciClass"
        options={['All', 'PD1', 'PD-L1', 'CTLA-4']}
        value={filters.iciClass} onChange={update('iciClass')} />
      <SelectFilter label="ICI Name"
        options={filterOptions.iciName || ['All']}
        value={filters.iciName} onChange={update('iciName')} />
      <SelectFilter label="Cancer Type"
        options={filterOptions.cancerType || ['All']}
        value={filters.cancerType} onChange={update('cancerType')} />
      <RadioGroup label="Type of Therapy" name="typeOfTherapy"
        options={['All', 'Combination', 'Monotherapy']}
        value={filters.typeOfTherapy} onChange={update('typeOfTherapy')} />
      <SelectFilter label="Control Arm"
        options={filterOptions.controlArm || ['All']}
        value={filters.controlArm} onChange={update('controlArm')} />
      <SelectFilter label="Clinical Setting"
        options={filterOptions.clinicalSetting || ['All']}
        value={filters.clinicalSetting} onChange={update('clinicalSetting')} />
      <RadioGroup label="Trial Phase" name="trialPhase"
        options={['All', 'Phase 2', 'Phase 3']}
        value={filters.trialPhase} onChange={update('trialPhase')} />
      <RadioGroup label="Type of Study" name="typeOfStudy"
        options={['All', 'Follow-up', 'Original publication']}
        value={filters.typeOfStudy} onChange={update('typeOfStudy')} />
      <SelectFilter label="Primary Endpoint"
        options={filterOptions.primaryEndpoint || ['All']}
        value={filters.primaryEndpoint} onChange={update('primaryEndpoint')} />
      <RadioGroup label="Included in MA" name="includedInMA"
        options={['All', 'Yes', 'No']}
        value={filters.includedInMA} onChange={update('includedInMA')} />
    </>
  )

  return (
    <>
      {/* Mobile toggle button */}
      <button className="mobile-filter-toggle" onClick={() => setMobileOpen(true)}>
        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
          <path d="M1 3h14v1.5L9.5 10v5l-3-1.5V10L1 4.5V3z"/>
        </svg>
        Filters
        {activeCount > 0 && <span className="filter-badge">{activeCount}</span>}
      </button>

      {/* Desktop sidebar */}
      <aside className="filter-panel">
        <div className="filter-panel-header">
          <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
            <path d="M1 3h14v1.5L9.5 10v5l-3-1.5V10L1 4.5V3z"/>
          </svg>
          Filters
          <button className="reset-btn" onClick={onReset}>
            <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
              <path d="M6 1a5 5 0 100 10A5 5 0 006 1zm1.5 7.5L6 7l-1.5 1.5-1-1L5 6 3.5 4.5l1-1L6 5l1.5-1.5 1 1L7 6l1.5 1.5-1 1z"/>
            </svg>
            Reset
          </button>
        </div>
        <div className="filter-panel-body">{content}</div>
      </aside>

      {/* Mobile drawer */}
      {mobileOpen && (
        <>
          <div className="mobile-drawer-overlay" onClick={() => setMobileOpen(false)} />
          <div className="mobile-drawer">
            <div className="mobile-drawer-header">
              <span>Filters</span>
              <div style={{display:'flex', gap:'8px', alignItems:'center'}}>
                <button className="reset-btn" onClick={() => { onReset(); }}>Reset All</button>
                <button className="mobile-drawer-close" onClick={() => setMobileOpen(false)}>✕</button>
              </div>
            </div>
            <div className="mobile-drawer-body">{content}</div>
            <div className="mobile-drawer-footer">
              <button className="mobile-apply-btn" onClick={() => setMobileOpen(false)}>
                Apply Filters
              </button>
            </div>
          </div>
        </>
      )}
    </>
  )
}