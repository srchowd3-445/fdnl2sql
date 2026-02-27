import { useEffect, useRef, useState } from 'react'

const ORCHESTRATOR_API_BASE = (import.meta.env.VITE_ORCHESTRATOR_API_BASE || '').trim().replace(/\/$/, '')
const CHAT_SQL_ENDPOINT = `${ORCHESTRATOR_API_BASE}/api/chat_sql`

async function callChatSqlOrchestrator(question) {
  const response = await fetch(CHAT_SQL_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })

  let payload = null
  try {
    payload = await response.json()
  } catch {
    // ignore JSON parse errors and handle via status below
  }

  if (!response.ok) {
    const msg = payload?.error || payload?.message || `Orchestrator API error ${response.status}`
    throw new Error(msg)
  }

  if (!payload || typeof payload !== 'object') {
    throw new Error('Orchestrator API returned an invalid response.')
  }

  return payload
}

function executeSqlOnLoadedDb(db, sql) {
  if (!db) {
    throw new Error('Database is not loaded yet.')
  }
  const resultSets = db.exec(sql)
  if (!Array.isArray(resultSets) || resultSets.length === 0) {
    return { rows: [], columns: [] }
  }

  const first = resultSets[0]
  const columns = Array.isArray(first.columns) ? first.columns : []
  const values = Array.isArray(first.values) ? first.values : []

  const rows = values.map((row) =>
    Object.fromEntries(columns.map((col, i) => [col, row[i]]))
  )

  return { rows, columns }
}

function confidenceText(confidence) {
  if (typeof confidence !== 'number' || Number.isNaN(confidence)) return 'NA'
  return confidence.toFixed(6)
}

export default function ChatBot({ onClose, onQueryResult, onClearQuery, db }) {
  const [messages, setMessages] = useState([
    {
      id: 0,
      from: 'bot',
      text:
        "Ask a clinical trials question. I'll run the NL2SQL orchestrator, generate SQL, execute it on the local database, and show results in the table.",
    },
  ])
  const [input, setInput] = useState('')
  const [typing, setTyping] = useState(false)

  const bottomRef = useRef(null)
  const inputRef = useRef(null)
  const idRef = useRef(1)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [messages, typing])

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  const addMessage = (from, payload) => {
    const msg = { id: idRef.current++, from, ...payload }
    setMessages((prev) => [...prev, msg])
    return msg
  }

  const send = async () => {
    const text = input.trim()
    if (!text || typing) return

    addMessage('user', { text })
    setInput('')
    setTyping(true)

    try {
      const orchestrator = await callChatSqlOrchestrator(text)

      const finalSql = String(orchestrator.final_sql || '').trim()
      if (!finalSql) {
        throw new Error(orchestrator.error || 'Orchestrator did not return SQL.')
      }

      const { rows, columns } = executeSqlOnLoadedDb(db, finalSql)

      if (typeof onQueryResult === 'function') {
        onQueryResult(rows, columns, text)
      }

      const confidence = orchestrator.confidence_overall
      const statusText = `Generated SQL and executed successfully. Returned ${rows.length} row(s). Confidence: ${confidenceText(confidence)}.`

      addMessage('bot', {
        text: statusText,
        sql: finalSql,
        confidence,
      })
    } catch (err) {
      addMessage('bot', {
        text: `❌ ${err?.message || 'Unknown error'}`,
        isError: true,
      })
    } finally {
      setTyping(false)
    }
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  const renderBubble = (msg) => {
    if (msg.isError) {
      return <div className="chat-bubble error-bubble">{msg.text}</div>
    }

    if (msg.sql) {
      return (
        <div className="chat-bubble sql-bubble">
          <div>{msg.text}</div>
          <div className="sql-block-label">Predicted SQL</div>
          <pre className="sql-code">{msg.sql}</pre>
        </div>
      )
    }

    return (
      <div className="chat-bubble">
        {msg.text.split('\n').map((line, i, arr) => (
          <span key={i}>
            {line}
            {i < arr.length - 1 && <br />}
          </span>
        ))}
      </div>
    )
  }

  return (
    <div className="chatbot-panel">
      <div className="chatbot-header">
        <div className="chatbot-header-left">
          <div className="chatbot-avatar">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="white">
              <path d="M8 1a5 5 0 00-5 5v1H2a1 1 0 000 2h1v1a5 5 0 0010 0v-1h1a1 1 0 000-2h-1V6a5 5 0 00-5-5z" />
            </svg>
          </div>
          <div>
            <div className="chatbot-title">MAYO-AIM2 Assistant</div>
            <div className="chatbot-status">
              <span className="status-dot" />
              {ORCHESTRATOR_API_BASE ? 'Orchestrator API configured' : 'Using /api/chat_sql'}
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          {typeof onClearQuery === 'function' && (
            <button className="chatbot-close-btn" onClick={onClearQuery} title="Clear SQL table mode">
              Clear
            </button>
          )}
          <button className="chatbot-close-btn" onClick={onClose} title="Close">
            <svg
              width="14"
              height="14"
              viewBox="0 0 14 14"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
            >
              <path d="M1 1l12 12M13 1L1 13" />
            </svg>
          </button>
        </div>
      </div>

      <div className="chatbot-messages">
        {messages.map((msg) => (
          <div key={msg.id} className={`chat-message ${msg.from}`}>
            {msg.from === 'bot' && (
              <div className="bot-icon">
                <svg width="12" height="12" viewBox="0 0 16 16" fill="white">
                  <path d="M8 1a5 5 0 00-5 5v1H2a1 1 0 000 2h1v1a5 5 0 0010 0v-1h1a1 1 0 000-2h-1V6a5 5 0 00-5-5z" />
                </svg>
              </div>
            )}
            {msg.from === 'user' ? (
              <div className="chat-bubble user-bubble">{msg.text}</div>
            ) : (
              renderBubble(msg)
            )}
          </div>
        ))}

        {typing && (
          <div className="chat-message bot">
            <div className="bot-icon">
              <svg width="12" height="12" viewBox="0 0 16 16" fill="white">
                <path d="M8 1a5 5 0 00-5 5v1H2a1 1 0 000 2h1v1a5 5 0 0010 0v-1h1a1 1 0 000-2h-1V6a5 5 0 00-5-5z" />
              </svg>
            </div>
            <div className="chat-bubble typing-bubble">
              <span className="dot" />
              <span className="dot" />
              <span className="dot" />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="chatbot-input-area">
        <textarea
          ref={inputRef}
          className="chatbot-input"
          placeholder="Ask about the clinical trials dataset..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKey}
          rows={1}
          disabled={typing}
        />
        <button
          className={`chatbot-send-btn ${input.trim() && !typing ? 'active' : ''}`}
          onClick={send}
          disabled={!input.trim() || typing}
        >
          {typing ? (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor" style={{ animation: 'spin 1s linear infinite' }}>
              <path d="M7 1a6 6 0 11-4.24 1.76" />
            </svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
              <path d="M1 1l14 7-14 7V9.5l10-1.5-10-1.5V1z" />
            </svg>
          )}
        </button>
      </div>
    </div>
  )
}
