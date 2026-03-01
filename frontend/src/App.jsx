//frontend/src/App.jsx
import { useState, useEffect, useMemo, useRef } from 'react'
import Header from './components/Header'
import HeroBlock from './components/HeroBlock'
import FilterPanel from './components/FilterPanel'
import DataTable from './components/DataTable'
import ColumnToggle from './components/ColumnToggle'
import ChatBot from './components/ChatBot'
import './App.css'

async function loadDatabase() {
  if (!window.initSqlJs) {
    await new Promise((resolve, reject) => {
      const script = document.createElement('script')
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.10.2/sql-wasm.js'
      script.onload = resolve
      script.onerror = () => reject(new Error('Failed to load sql.js from CDN'))
      document.head.appendChild(script)
    })
  }
  const SQL = await window.initSqlJs({
    locateFile: filename =>
      `https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.10.2/${filename}`
  })
  const response = await fetch('/database.db')
  if (!response.ok) throw new Error(`Could not fetch database.db (HTTP ${response.status})`)
  const buffer = await response.arrayBuffer()
  const db = new SQL.Database(new Uint8Array(buffer))
  const result = db.exec('SELECT * FROM clinical_trials')
  if (!result.length) return { data: [], db }
  const { columns, values } = result[0]
  const data = values.map(row =>
    Object.fromEntries(columns.map((col, i) => [col, row[i]]))
  )
  return { data, db }
}

export default function App() {
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(true)
  const [dbError, setDbError] = useState(null)
  const [chatOpen, setChatOpen] = useState(false)
  const dbRef = useRef(null)

  // SQL query mode — when active, table shows SQL results instead of filtered data
  const [sqlMode, setSqlMode] = useState(false)
  const [sqlData, setSqlData] = useState([])
  const [sqlColumns, setSqlColumns] = useState([])
  const [sqlLabel, setSqlLabel] = useState('')

  const [filters, setFilters] = useState({
    iciClass: 'All',
    iciName: 'All',
    cancerType: 'All',
    trialPhase: 'All',
    typeOfTherapy: 'All',
    controlArm: 'All',
    clinicalSetting: 'All',
    typeOfStudy: 'All',
    primaryEndpoint: 'All',
    includedInMA: 'All',
  })

  const [visibleColumns, setVisibleColumns] = useState({
    'NCT': true,
    'PubMed ID': true,
    'Trial name': true,
    'Author': true,
    'Year': true,
    'Trial phase': false,
    'Number of arms': false,
    'Total sample size': false,
    'Originial publication or Follow-up': true,
    'Cancer type': true,
    'Treatment regimen': false,
    'Name of ICI': true,
    'Class of ICI': true,
    'Monotherapy/combination': false,
    'Type of combination': false,
    'Control regimen': false,
    'Type of control': false,
    'Lines of treatment': false,
    'Clincal setting in relation to surgery': false,
    'Primary endpoint': false,
    'Priamry Multiple, composite, or co-primary endpoints?': false,
    'Secondary endpoint': false,
    'Is PD-L1 positivity inclusion criteria': false,
    'Is any other biomarker used for inclusion': false,
    'Type of follow-up given': false,
    'Follow-up duration for primary endpoint(s) in months | Overall': false,
    'Follow-up duration for primary endpoint(s) in months | Rx': false,
    'Follow-up duration for primary endpoint(s) in months | Control': false,
    'Type of therapy': false,
    'Control arm': false,
    'Clinical setting': false,
    'Included in MA': false,
  })

  useEffect(() => {
    loadDatabase()
      .then(({ data, db }) => {
        setData(data)
        dbRef.current = db
        setLoading(false)
      })
      .catch(err => {
        setDbError(err.message)
        setLoading(false)
      })
  }, [])

  const getDistinct = (col) => {
    const vals = [...new Set(data.map(r => r[col]).filter(Boolean).map(v => v.toString().trim()))].sort()
    return ['All', ...vals]
  }

  const filterOptions = useMemo(() => ({
    iciClass: getDistinct('Class of ICI'),
    iciName: getDistinct('Name of ICI'),
    cancerType: getDistinct('Cancer type'),
    trialPhase: getDistinct('Trial phase'),
    typeOfTherapy: getDistinct('Type of therapy'),
    controlArm: getDistinct('Control arm'),
    clinicalSetting: getDistinct('Clinical setting'),
    typeOfStudy: getDistinct('Originial publication or Follow-up'),
    primaryEndpoint: getDistinct('Primary endpoint'),
    includedInMA: getDistinct('Included in MA'),
  }), [data])

  const filteredData = useMemo(() => {
    return data.filter(row => {
      if (filters.iciClass !== 'All' && (row['Class of ICI'] || '').trim() !== filters.iciClass) return false
      if (filters.iciName !== 'All' && (row['Name of ICI'] || '').trim() !== filters.iciName) return false
      if (filters.cancerType !== 'All' && (row['Cancer type'] || '').trim() !== filters.cancerType) return false
      if (filters.trialPhase !== 'All' && (row['Trial phase'] || '').trim() !== filters.trialPhase) return false
      if (filters.typeOfTherapy !== 'All' && (row['Type of therapy'] || '').trim() !== filters.typeOfTherapy) return false
      if (filters.controlArm !== 'All' && (row['Control arm'] || '').trim() !== filters.controlArm) return false
      if (filters.clinicalSetting !== 'All' && (row['Clinical setting'] || '').trim() !== filters.clinicalSetting) return false
      if (filters.typeOfStudy !== 'All' && (row['Originial publication or Follow-up'] || '').trim() !== filters.typeOfStudy) return false
      if (filters.primaryEndpoint !== 'All' && (row['Primary endpoint'] || '').trim() !== filters.primaryEndpoint) return false
      if (filters.includedInMA !== 'All' && (row['Included in MA'] || '').trim() !== filters.includedInMA) return false
      return true
    })
  }, [data, filters])

  const resetFilters = () => {
    setFilters(Object.fromEntries(Object.keys(filters).map(k => [k, 'All'])))
  }

  // Called by ChatBot when a SQL query runs successfully
  const handleQueryResult = (rows, columns, label) => {
    const cols = Array.isArray(columns) ? columns : []
    const rawRows = Array.isArray(rows) ? rows : []

    // Chat SQL execution returns sqlite-style arrays; DataTable expects objects.
    const normalizedRows = rawRows.map((row, idx) => {
      if (row && typeof row === 'object' && !Array.isArray(row)) return row
      const arr = Array.isArray(row) ? row : []
      return Object.fromEntries(cols.map((c, i) => [c, arr[i]]))
    })

    setSqlData(normalizedRows)
    setSqlColumns(cols)
    setSqlLabel(label)
    setSqlMode(true)
  }

  // Called when user clicks "Reset table" in chatbot
  const handleClearQuery = () => {
    setSqlMode(false)
    setSqlData([])
    setSqlColumns([])
    setSqlLabel('')
  }

  // In SQL mode, show only the columns the SQL selected
  const sqlVisibleColumns = useMemo(() => {
    if (!sqlColumns.length) return {}
    return Object.fromEntries(sqlColumns.map(c => [c, true]))
  }, [sqlColumns])

  const allColumns = Object.keys(visibleColumns)

  // What the table actually shows
  const tableData = sqlMode ? sqlData : filteredData
  const tableVisibleColumns = sqlMode ? sqlVisibleColumns : visibleColumns

  return (
    <div className="app">
      <Header />
      <HeroBlock />
      <main className="main-layout">
        <FilterPanel
          filters={filters}
          setFilters={setFilters}
          filterOptions={filterOptions}
          onReset={resetFilters}
        />
        <div className={`data-and-chat ${chatOpen ? 'chat-is-open' : ''}`}>
          <div className="content-area">
            <ColumnToggle
              allColumns={allColumns}
              visibleColumns={visibleColumns}
              setVisibleColumns={setVisibleColumns}
              chatOpen={chatOpen}
              onToggleChat={() => setChatOpen(o => !o)}
            />

            {/* Yellow banner shown when table is in SQL query mode */}
            {sqlMode && (
              <div className="sql-mode-banner">
                <div className="sql-mode-banner-left">
                  <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M2 2h12v2H2zM2 7h8v2H2zM2 12h10v2H2z"/>
                  </svg>
                  <span>SQL Query Result: <strong>{sqlLabel}</strong></span>
                </div>
                <button className="sql-mode-clear-btn" onClick={handleClearQuery}>
                  ✕ Clear · Show all records
                </button>
              </div>
            )}

            <div className="table-header-bar">
              <span className="record-count">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" style={{marginRight:'6px',verticalAlign:'middle'}}>
                  <rect x="0" y="0" width="7" height="7" rx="1"/><rect x="9" y="0" width="7" height="7" rx="1"/>
                  <rect x="0" y="9" width="7" height="7" rx="1"/><rect x="9" y="9" width="7" height="7" rx="1"/>
                </svg>
                {sqlMode ? 'Query Results' : 'Data of Studies'} &nbsp;|&nbsp;
                <strong>{tableData.length}</strong> Records
                {sqlMode && <span className="sql-mode-tag">SQL</span>}
              </span>
            </div>

            {loading ? (
              <div className="loading">
                <div className="loading-spinner" />
                <p>Loading database...</p>
              </div>
            ) : dbError ? (
              <div className="db-error">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 2a10 10 0 110 20A10 10 0 0112 2zm1 13h-2v2h2v-2zm0-8h-2v6h2V7z"/>
                </svg>
                <div>
                  <strong>Could not load database.db</strong>
                  <p>Make sure <code>database.db</code> is placed in the <code>public/</code> folder.</p>
                  <p style={{fontSize:'0.75rem', opacity:0.7}}>{dbError}</p>
                </div>
              </div>
            ) : (
              <DataTable
                data={tableData}
                visibleColumns={tableVisibleColumns}
              />
            )}
          </div>

          {chatOpen && (
            <ChatBot
              onClose={() => setChatOpen(false)}
              onQueryResult={handleQueryResult}
              onClearQuery={handleClearQuery}
              db={dbRef.current}
            />
          )}
        </div>
      </main>

      <footer className="footer">
        <div className="footer-inner">
          <div className="footer-columns">
            <div className="footer-col footer-col-brand">
              <div className="footer-logo">
                <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
                  <rect width="36" height="36" rx="4" fill="rgba(255,255,255,0.15)"/>
                  <path d="M18 6L30 28H6L18 6Z" fill="white" opacity="0.9"/>
                </svg>
                
                <span>FD-NL2SQL</span>
              </div>
            
              <p>FD-NL2SQL is a natural-language-to-SQL assistant for oncology clinical-trial databases. It translates everyday research questions into executable SQL so users can query biomarkers, endpoints, interventions, eligibility criteria, and follow-up outcomes without needing SQL or schema expertise.</p>
            </div>
            <div className="footer-col">
             
              <h4>About This Resource</h4>
              <p>This interface supports ad-hoc, multi-constraint analysis of a curated oncology trial dataset (Phase 2/3 studies, ICI-focused variables, trial metadata, and endpoint information). It is designed for interactive exploration: generate SQL, inspect results, refine queries, and iterate quickly.</p>
            </div>
            <div className="footer-col">
             
              <h4>Contact</h4>
              <p>For project questions, collaboration requests, or data/method clarification, contact the FD-NL2SQL team. If you use this resource in research outputs, please cite the FD-NL2SQL project and associated manuscript/materials.</p>
            </div>
          </div>
          <div className="footer-bottom">
           
            <span>© 2026 FD-NL2SQL. For research purposes only.</span>
            <span className="footer-links">
              <a href="#">Terms of Use</a>
              <a href="#">Privacy Policy</a>
              <a href="#">Accessibility</a>
            </span>
          </div>
        </div>
      </footer>
    </div>
  )
}
