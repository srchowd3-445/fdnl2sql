//frontend/src/components/DataTable.jsx
import { useState, useMemo } from 'react'

const PUBMED_BASE = 'https://pubmed.ncbi.nlm.nih.gov/'
const NCT_BASE = 'https://clinicaltrials.gov/ct2/show/'

const CLASS_COLORS = {
  'PD1': { bg: '#e8f4fd', color: '#1a6fa8', border: '#90caf9' },
  'PD-L1': { bg: '#e8f7ee', color: '#1a7a3a', border: '#81c784' },
  'CTLA-4': { bg: '#fff3e0', color: '#b85c00', border: '#ffb74d' },
  'PD-L1, CTLA-4': { bg: '#f3e8fd', color: '#6a1a9a', border: '#ce93d8' },
  'PD1, CTLA-4': { bg: '#fce4ec', color: '#a81a3a', border: '#f48fb1' },
}

function Badge({ value, colorMap }) {
  const style = colorMap?.[value] || { bg: '#f0f4f8', color: '#4a5568', border: '#cbd5e0' }
  return (
    <span className="badge" style={{
      backgroundColor: style.bg,
      color: style.color,
      border: `1px solid ${style.border}`,
    }}>
      {value}
    </span>
  )
}

function SortIcon({ direction }) {
  if (!direction) return (
    <svg width="10" height="12" viewBox="0 0 10 12" fill="currentColor" opacity="0.3">
      <path d="M5 0L9 4H1L5 0zM5 12L1 8H9L5 12z"/>
    </svg>
  )
  return (
    <svg width="10" height="12" viewBox="0 0 10 12" fill="currentColor" opacity="0.8">
      {direction === 'asc'
        ? <path d="M5 0L9 4H1L5 0z"/>
        : <path d="M5 12L1 8H9L5 12z"/>}
    </svg>
  )
}

export default function DataTable({ data, visibleColumns }) {
  const [sortConfig, setSortConfig] = useState({ key: null, direction: null })
  const [page, setPage] = useState(1)
  const [rowsPerPage, setRowsPerPage] = useState(25)

  const cols = Object.entries(visibleColumns).filter(([, v]) => v).map(([k]) => k)

  const handleSort = (col) => {
    setSortConfig(prev => {
      if (prev.key !== col) return { key: col, direction: 'asc' }
      if (prev.direction === 'asc') return { key: col, direction: 'desc' }
      return { key: null, direction: null }
    })
  }

  const sortedData = useMemo(() => {
    if (!sortConfig.key) return data
    return [...data].sort((a, b) => {
      const av = a[sortConfig.key] ?? ''
      const bv = b[sortConfig.key] ?? ''
      const cmp = String(av).localeCompare(String(bv), undefined, { numeric: true })
      return sortConfig.direction === 'asc' ? cmp : -cmp
    })
  }, [data, sortConfig])

  const totalPages = Math.ceil(sortedData.length / rowsPerPage)
  const pageData = sortedData.slice((page - 1) * rowsPerPage, page * rowsPerPage)

  const getColWidth = (col) => {
    if (col === 'NCT') return '120px'
    if (col === 'PubMed ID') return '100px'
    if (col === 'Year') return '70px'
    if (col === 'Class of ICI') return '100px'
    if (col === 'Name of ICI') return '140px'
    if (col === 'Cancer type') return '130px'
    if (col === 'Originial publication or Follow-up') return '120px'
    return '160px'
  }

  const renderCell = (row, col) => {
    const val = row[col]
    if (val === null || val === undefined || val === '') return <span style={{ color: '#9aa5b1' }}>—</span>

    if (col === 'PubMed ID') {
      return <a href={`${PUBMED_BASE}${val}`} target="_blank" rel="noopener" className="table-link">{val}</a>
    }
    if (col === 'NCT') {
      return <a href={`${NCT_BASE}${val}`} target="_blank" rel="noopener" className="table-link nct-link">{val}</a>
    }
    if (col === 'Class of ICI') {
      return <Badge value={val} colorMap={CLASS_COLORS} />
    }
    if (col === 'Originial publication or Follow-up') {
      return (
        <span className={`study-type-badge ${val === 'Follow-up' ? 'followup' : 'original'}`}>
          {val}
        </span>
      )
    }
    if (col === 'Trial phase') {
      return <span className="phase-badge">{val}</span>
    }
    if (col === 'Included in MA') {
      return val === 'Yes'
        ? <span className="yes-badge">Yes</span>
        : <span className="no-badge">No</span>
    }
    const str = String(val)
    if (str.length > 40) return <span title={str}>{str.substring(0, 38)}…</span>
    return str
  }

  return (
    <div className="table-container">
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              {cols.map(col => (
                <th
                  key={col}
                  style={{ minWidth: getColWidth(col) }}
                  onClick={() => handleSort(col)}
                  className="sortable-th"
                >
                  <div className="th-inner">
                    <span>{col}</span>
                    <SortIcon direction={sortConfig.key === col ? sortConfig.direction : null} />
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageData.length === 0 ? (
              <tr>
                <td colSpan={cols.length} className="empty-row">
                  No records match the selected filters.
                </td>
              </tr>
            ) : pageData.map((row, i) => (
              <tr key={`${row['NCT']}-${row['PubMed ID']}-${i}`} className={i % 2 === 0 ? 'row-even' : 'row-odd'}>
                {cols.map(col => (
                  <td key={col}>{renderCell(row, col)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="pagination">
        <div className="pagination-info">
          Showing {Math.min((page - 1) * rowsPerPage + 1, sortedData.length)}–{Math.min(page * rowsPerPage, sortedData.length)} of {sortedData.length} records
        </div>
        <div className="pagination-controls">
          <label className="rows-label">
            Rows per page:
            <select value={rowsPerPage} onChange={e => { setRowsPerPage(Number(e.target.value)); setPage(1) }}>
              {[10, 25, 50, 100].map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
          <button className="page-btn" onClick={() => setPage(1)} disabled={page === 1}>«</button>
          <button className="page-btn" onClick={() => setPage(p => p - 1)} disabled={page === 1}>‹</button>
          {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
            let p
            if (totalPages <= 5) p = i + 1
            else if (page <= 3) p = i + 1
            else if (page >= totalPages - 2) p = totalPages - 4 + i
            else p = page - 2 + i
            return (
              <button key={p} className={`page-btn ${p === page ? 'active' : ''}`} onClick={() => setPage(p)}>{p}</button>
            )
          })}
          <button className="page-btn" onClick={() => setPage(p => p + 1)} disabled={page === totalPages}>›</button>
          <button className="page-btn" onClick={() => setPage(totalPages)} disabled={page === totalPages}>»</button>
        </div>
      </div>
    </div>
  )
}
