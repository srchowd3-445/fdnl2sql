import { useState, useEffect, useRef } from 'react'

const API_BASE = (import.meta.env.VITE_ORCHESTRATOR_PROXY_TARGET || "").trim().replace(/\/$/, "")
const apiUrl = (path) => (API_BASE ? `${API_BASE}${path}` : path)

// Pipeline API call
async function callPipeline(userMessage) {
  const response = await fetch(apiUrl('/api/chat-query'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: userMessage, skip_exec: 0, preview_rows: 20 }),
  })

  if (!response.ok) {
    let errText = `Pipeline API error ${response.status}`
    try {
      const data = await response.json()
      const detail = data?.detail
      if (typeof detail === 'string') {
        errText = detail
      } else if (detail?.message) {
        errText = detail.message
      }
    } catch {
      // Keep fallback message
    }
    console.error("Pipeline API call failed", { status: response.status, errText })
    throw new Error(errText)
  }

  return response.json()
}

async function saveSeedFeedback(question, predictedSql) {
  const response = await fetch(apiUrl('/api/seed-feedback'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, predicted_sql: predictedSql }),
  })
  if (!response.ok) {
    let errText = `Seed save API error ${response.status}`
    try {
      const data = await response.json()
      errText = data?.detail?.message || data?.detail || errText
    } catch {
      // Keep fallback message.
    }
    throw new Error(String(errText))
  }
  return response.json()
}

// Autocomplete helpers
async function loadSeedQuestions() {
  try {
    const res = await fetch('/seed_questions.json')
    const data = await res.json()
    return data.map(d => d.original_question)
  } catch {
    return []
  }
}

function findSuggestion(input, questions) {
  if (!input || input.length < 4) return ''
  const lower = input.toLowerCase()
  const match = questions.find(q => q.toLowerCase().startsWith(lower))
  return match || ''
}

// Satisfaction meter
function computeSatisfaction(ratings) {
  if (!ratings.length) return null
  const score = ratings.reduce((acc, r) => {
    if (r === 'good') return acc + 1
    if (r === 'neutral') return acc + 0.5
    return acc
  }, 0)
  return score / ratings.length
}

function SatisfactionBar({ ratings }) {
  if (!ratings.length) return null
  const score = computeSatisfaction(ratings)
  const pct = Math.round(score * 100)
  const color = score >= 0.7 ? '#22c55e' : score >= 0.4 ? '#f59e0b' : '#ef4444'
  const label = score >= 0.7 ? 'Satisfied' : score >= 0.4 ? 'Mixed' : 'Unsatisfied'
  const pulseClass = score >= 0.7 ? 'pulse-green' : score < 0.4 ? 'pulse-red' : 'pulse-amber'
  return (
    <div className={`satisfaction-bar-wrap ${pulseClass}`}>
      <div className="satisfaction-bar-label">
        <span className="satisfaction-text">{label}</span>
        <span className="satisfaction-pct">{pct}%</span>
      </div>
      <div className="satisfaction-track">
        <div className="satisfaction-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <div className="satisfaction-counts">
        <span className="sc-good">✔ {ratings.filter(r => r === 'good').length}</span>
        <span className="sc-neutral">● {ratings.filter(r => r === 'neutral').length}</span>
        <span className="sc-bad">✕ {ratings.filter(r => r === 'bad').length}</span>
      </div>
    </div>
  )
}

// Feedback buttons
function FeedbackButtons({ messageId, onRate, onChooseEditTarget }) {
  const [rating, setRating] = useState(null)
  const [animating, setAnimating] = useState(null)

  const buttons = [
    { value: 'good', label: 'Helpful', symbol: '✓', activeColor: '#22c55e' },
    { value: 'neutral', label: 'Okay', symbol: '●', activeColor: '#d97706' },
    { value: 'bad', label: 'Unhelpful', symbol: '✕', activeColor: '#ef4444' },
  ]

  const handleRate = value => {
    if (rating) return
    setAnimating(value)
    setTimeout(() => {
      setRating(value)
      setAnimating(null)
      onRate(messageId, value)
      if (value === 'good') console.log('User liked the output')
      if (value === 'neutral') {
        console.log('User wants to edit output')
        onChooseEditTarget(messageId)
      }
      if (value === 'bad') console.log('User did not like the output')
    }, 350)
  }

  if (rating) {
    const chosen = buttons.find(b => b.value === rating)
    return (
      <div className="feedback-thankyou" style={{ color: chosen.activeColor }}>
        <span className="feedback-thankyou-icon">{chosen.symbol}</span>
        <span>Thanks for the feedback</span>
      </div>
    )
  }

  return (
    <div className="feedback-buttons">
      <span className="feedback-prompt">Was this helpful?</span>
      {buttons.map(btn => (
        <button
          key={btn.value}
          className={`feedback-btn fb-${btn.value} ${animating === btn.value ? 'fb-animating' : ''}`}
          onClick={() => handleRate(btn.value)}
          title={btn.label}
        >
          {btn.symbol}
        </button>
      ))}
    </div>
  )
}

function executeSqlOnClient(db, sql) {
  if (!db || !sql) return { columns: [], rows: [] }
  const out = db.exec(sql)
  if (!Array.isArray(out) || out.length === 0) return { columns: [], rows: [] }
  const first = out[0] || {}
  return {
    columns: Array.isArray(first.columns) ? first.columns : [],
    rows: Array.isArray(first.values) ? first.values : [],
  }
}

// Main ChatBot component
export default function ChatBot({ onClose, db, onQueryResult, onClearQuery }) {
  const [messages, setMessages] = useState([
    {
      id: 0,
      from: 'bot',
      text: "Hi! Ask a clinical-trial question and I'll generate SQL and execute it.",
      showFeedback: false,
    },
  ])
  const [input, setInput] = useState('')
  const [typing, setTyping] = useState(false)
  const [ratings, setRatings] = useState([])
  const [showSatisfaction, setShowSatisfaction] = useState(false)

  const [editingId, setEditingId] = useState(null)
  const [editTarget, setEditTarget] = useState('sql')
  const [editDraft, setEditDraft] = useState('')
  const [editError, setEditError] = useState('')

  const [seedQuestions, setSeedQuestions] = useState([])
  const [suggestion, setSuggestion] = useState('')
  const [apiHealthy, setApiHealthy] = useState(null)

  const bottomRef = useRef(null)
  const inputRef = useRef(null)
  const idRef = useRef(1)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [messages, typing])

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  useEffect(() => {
    if (ratings.length > 0) setShowSatisfaction(true)
  }, [ratings])

  useEffect(() => {
    loadSeedQuestions().then(setSeedQuestions)
  }, [])

  useEffect(() => {
    fetch(apiUrl('/health'))
      .then(r => setApiHealthy(r.ok))
      .catch(() => setApiHealthy(false))
  }, [])

  const addMessage = (from, payload) => {
    const msg = { id: idRef.current++, from, ...payload }
    setMessages(prev => [...prev, msg])
    return msg
  }

  const handleRate = async (messageId, value) => {
    setRatings(prev => [...prev, value])
    if (value === 'bad') {
      const msg = messages.find(m => m.id === messageId)
      const question = String(msg?.sourceQuestion || '').trim()
      const previousSql = String(msg?.predictedSql || msg?.text || '').trim()
      if (!question) return

      const rerunPrompt = [
        question,
        '',
        'IMPORTANT: The previous SQL answer was not acceptable.',
        'You MUST provide a different SQL answer than the prior attempt.',
        'Do not repeat the same SQL string.',
        'If the previous SQL used an incorrect filter/value/operator, correct it.',
        '',
        'Previous SQL:',
        previousSql || '(none)',
      ].join('\n')

      setTyping(true)
      try {
        const result = await callPipeline(rerunPrompt)
        const finalSql = result?.final_sql || ''
        if (result?.error || !finalSql) {
          console.error("Pipeline returned error in rerun", { result, finalSql })
          addMessage('bot', {
            text: `Pipeline error: ${result?.error || 'No SQL generated.'}`,
            isError: true,
            showFeedback: false,
          })
          return
        }

        let exec = result?.execution_preview || {}
        if (db && finalSql) {
          try {
            const local = executeSqlOnClient(db, finalSql)
            exec = { error: null, columns: local.columns, rows: local.rows }
          } catch (e) {
            exec = { ...exec, error: String(e) }
          }
        }

        if (!exec?.error && typeof onQueryResult === 'function') {
          onQueryResult(
            Array.isArray(exec.rows) ? exec.rows : [],
            Array.isArray(exec.columns) ? exec.columns : [],
            question
          )
        }

        addMessage('bot', {
          text: finalSql,
          showFeedback: true,
          sqlOnly: true,
          sourceQuestion: question,
          predictedSql: finalSql,
        })
      } catch (e) {
        console.error("Pipeline request exception in rerun", e)
        addMessage('bot', { text: `❌ ${String(e)}`, isError: true, showFeedback: false })
      } finally {
        setTyping(false)
      }
      return
    }

    if (value !== 'good') return

    const msg = messages.find(m => m.id === messageId)
    const question = String(msg?.sourceQuestion || '').trim()
    const predictedSql = String(msg?.predictedSql || '').trim()
    if (!question || !predictedSql) return

    try {
      await saveSeedFeedback(question, predictedSql)
      setSeedQuestions(prev => {
        if (prev.some(q => q.toLowerCase() === question.toLowerCase())) return prev
        return [...prev, question]
      })
    } catch (e) {
      console.error('Failed to save seed feedback:', e)
    }
  }

  const openEditChooser = (messageId) => {
    setMessages(prev =>
      prev.map(m =>
        m.id === messageId
          ? {
              ...m,
              chooseEditTarget: true,
            }
          : m
      )
    )
  }

  const chooseEditTarget = (messageId, target) => {
    const msg = messages.find(m => m.id === messageId)
    const seedText = target === 'question' ? String(msg?.sourceQuestion || '') : String(msg?.predictedSql || msg?.text || '')
    setMessages(prev =>
      prev.map(m =>
        m.id === messageId
          ? {
              ...m,
              chooseEditTarget: false,
            }
          : m
      )
    )
    setEditingId(messageId)
    setEditTarget(target)
    setEditDraft(seedText)
    setEditError('')
  }

  const handleSavePromptDecision = async (messageId, shouldSave) => {
    const msg = messages.find(m => m.id === messageId)
    const question = String(msg?.sourceQuestion || '').trim()
    const predictedSql = String(msg?.predictedSql || '').trim()

    if (!msg) return

    if (!shouldSave) {
      setMessages(prev =>
        prev.map(m =>
          m.id === messageId
            ? {
                ...m,
                savePromptResolved: true,
                savePromptChoice: 'no',
                text: 'Edited SQL not saved to seeds.',
              }
            : m
        )
      )
      return
    }

    if (!question || !predictedSql) {
      setMessages(prev =>
        prev.map(m =>
          m.id === messageId
            ? {
                ...m,
                savePromptResolved: true,
                savePromptChoice: 'error',
                text: 'Could not save: missing question or SQL.',
              }
            : m
        )
      )
      return
    }

    try {
      await saveSeedFeedback(question, predictedSql)
      setSeedQuestions(prev => {
        if (prev.some(q => q.toLowerCase() === question.toLowerCase())) return prev
        return [...prev, question]
      })
      setMessages(prev =>
        prev.map(m =>
          m.id === messageId
            ? {
                ...m,
                savePromptResolved: true,
                savePromptChoice: 'yes',
                text: 'Edited SQL saved to seed questions.',
              }
            : m
        )
      )
    } catch (e) {
      setMessages(prev =>
        prev.map(m =>
          m.id === messageId
            ? {
                ...m,
                savePromptResolved: true,
                savePromptChoice: 'error',
                text: `Failed to save edited SQL: ${String(e)}`,
                isError: true,
              }
            : m
        )
      )
    }
  }

  const handleInputChange = e => {
    const val = e.target.value
    setInput(val)
    setSuggestion(findSuggestion(val, seedQuestions))
    e.target.style.height = 'auto'
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px'
  }

  const handleKeyDown = e => {
    if (e.key === 'Tab' && suggestion) {
      e.preventDefault()
      setInput(suggestion)
      setSuggestion('')
      return
    }
    if (e.key === 'Escape') {
      setSuggestion('')
      return
    }
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  const send = async () => {
    const text = input.trim()
    if (!text || typing) return
    setSuggestion('')
    addMessage('user', { text, showFeedback: false })
    setInput('')
    setTyping(true)

    try {
      const result = await callPipeline(text)
      const finalSql = result?.final_sql || ''
      const pipelineExec = result?.execution_preview || {}
      let exec = pipelineExec

      // Prefer local DB execution so table and chatbot stay consistent.
      if (finalSql && db) {
        try {
          const local = executeSqlOnClient(db, finalSql)
          exec = { error: null, columns: local.columns, rows: local.rows }
        } catch (e) {
          exec = { ...pipelineExec, error: String(e) }
        }
      }

      if (result?.error || !finalSql) {
        console.error("Pipeline returned error in send", { result, finalSql })
        addMessage('bot', {
          text: `Pipeline error: ${result?.error || 'No SQL generated.'}`,
          isError: true,
          showFeedback: false,
        })
        return
      }

      if (finalSql && !exec?.error && typeof onQueryResult === 'function') {
        onQueryResult(
          Array.isArray(exec.rows) ? exec.rows : [],
          Array.isArray(exec.columns) ? exec.columns : [],
          text
        )
      }

      addMessage('bot', {
        text: finalSql,
        showFeedback: true,
        sqlOnly: true,
        sourceQuestion: text,
        predictedSql: finalSql,
      })
    } catch (err) {
      console.error("Pipeline request exception in send", err)
      addMessage('bot', { text: `❌ ${err.message}`, isError: true, showFeedback: false })
    } finally {
      setTyping(false)
    }
  }

  const renderBubble = msg => {
    if (msg.isError) return <div className="chat-bubble error-bubble">{msg.text}</div>

    if (editingId === msg.id) {
      return (
        <div className="edit-bubble-wrap">
          <textarea
            className="edit-bubble-textarea"
            value={editDraft}
            onChange={e => {
              setEditDraft(e.target.value)
              if (editError) setEditError('')
            }}
            rows={6}
            autoFocus
          />
          {editError && <div className="error-bubble" style={{ marginTop: 6 }}>{editError}</div>}
          <div className="edit-bubble-actions">
            <button
              className="edit-save-btn"
              onClick={async () => {
                const sql = String(editDraft || '').trim()
                if (!sql) {
                  setEditError('SQL cannot be empty.')
                  return
                }
                if (!db) {
                  if (editTarget === 'sql') {
                    setEditError('Database is not loaded yet. Try again in a moment.')
                    return
                  }
                }
                try {
                  if (editTarget === 'question') {
                    const editedQuestion = sql
                    setTyping(true)
                    const result = await callPipeline(editedQuestion)
                    const finalSql = result?.final_sql || ''
                    if (result?.error || !finalSql) {
                      throw new Error(`Pipeline error: ${result?.error || 'No SQL generated.'}`)
                    }

                    let exec = result?.execution_preview || {}
                    if (db && finalSql) {
                      try {
                        const local = executeSqlOnClient(db, finalSql)
                        exec = { error: null, columns: local.columns, rows: local.rows }
                      } catch (e) {
                        exec = { ...exec, error: String(e) }
                      }
                    }

                    if (!exec?.error && typeof onQueryResult === 'function') {
                      onQueryResult(
                        Array.isArray(exec.rows) ? exec.rows : [],
                        Array.isArray(exec.columns) ? exec.columns : [],
                        editedQuestion
                      )
                    }

                    addMessage('bot', {
                      text: finalSql,
                      showFeedback: true,
                      sqlOnly: true,
                      sourceQuestion: editedQuestion,
                      predictedSql: finalSql,
                    })
                  } else {
                    const local = executeSqlOnClient(db, sql)
                    if (typeof onQueryResult === 'function') {
                      onQueryResult(local.rows, local.columns, 'Edited SQL')
                    }
                    addMessage('bot', {
                      text: sql,
                      showFeedback: true,
                      sqlOnly: true,
                      sourceQuestion: msg.sourceQuestion || '',
                      predictedSql: sql,
                    })
                  }
                  setEditingId(null)
                  setEditTarget('sql')
                  setEditDraft('')
                  setEditError('')
                } catch (e) {
                  setEditError(`Failed to apply edit: ${String(e)}`)
                } finally {
                  setTyping(false)
                }
              }}
            >
              Save
            </button>
            <button
              className="edit-cancel-btn"
              onClick={() => {
                setEditingId(null)
                setEditTarget('sql')
                setEditDraft('')
                setEditError('')
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      )
    }

    if (msg.savePrompt) {
      return (
        <div className="chat-bubble">
          <div>{msg.text}</div>
          {msg.savePromptResolved ? null : (
            <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
              <button className="edit-save-btn" onClick={() => handleSavePromptDecision(msg.id, true)}>
                Yes
              </button>
              <button className="edit-cancel-btn" onClick={() => handleSavePromptDecision(msg.id, false)}>
                No
              </button>
            </div>
          )}
        </div>
      )
    }

    if (msg.chooseEditTarget) {
      return (
        <div className="chat-bubble">
          <div>Edit this result by changing the question or the SQL?</div>
          <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
            <button className="edit-save-btn" onClick={() => chooseEditTarget(msg.id, 'question')}>
              Edit Question
            </button>
            <button className="edit-cancel-btn" onClick={() => chooseEditTarget(msg.id, 'sql')}>
              Edit SQL
            </button>
          </div>
        </div>
      )
    }

    return (
      <div className="chat-bubble">
        {msg.sqlOnly ? (
          <pre className="sql-preview">{msg.text}</pre>
        ) : (
          msg.text.split('\n').map((line, i, arr) => (
            <span key={i}>
              {line}
              {i < arr.length - 1 && <br />}
            </span>
          ))
        )}
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
            <div className="chatbot-title">Clinical Assistant</div>
            <div className="chatbot-status">
              <span className="status-dot" />
              {apiHealthy === null ? 'Pipeline checking...' : apiHealthy ? 'Pipeline connected' : 'Pipeline unavailable'}
            </div>
          </div>
        </div>
        <button className="chatbot-close-btn" onClick={onClose} title="Close">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M1 1l12 12M13 1L1 13" />
          </svg>
        </button>
      </div>

      {showSatisfaction && <SatisfactionBar ratings={ratings} />}

      {apiHealthy === false && (
        <div className="chatbot-warning">
          <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 1L1 14h14L8 1zm0 3l4.5 8h-9L8 4zm-1 3v2h2V7H7zm0 3v2h2v-2H7z" />
          </svg>
          Start API server on port 9001 (or set <code>VITE_ORCHESTRATOR_PROXY_TARGET</code>).
        </div>
      )}
      {typeof onClearQuery === 'function' && (
        <div className="chatbot-warning" style={{ marginTop: 8 }}>
          <button className="sql-mode-clear-btn" onClick={onClearQuery}>
            Clear SQL Table Mode
          </button>
        </div>
      )}

      <div className="chatbot-messages">
        {messages.map(msg => (
          <div key={msg.id} className={`chat-message ${msg.from}`}>
            {msg.from === 'bot' && (
              <div className="bot-icon">
                <svg width="12" height="12" viewBox="0 0 16 16" fill="white">
                  <path d="M8 1a5 5 0 00-5 5v1H2a1 1 0 000 2h1v1a5 5 0 0010 0v-1h1a1 1 0 000-2h-1V6a5 5 0 00-5-5z" />
                </svg>
              </div>
            )}
            <div className="msg-content-wrap">
              {msg.from === 'user' ? <div className="chat-bubble user-bubble">{msg.text}</div> : renderBubble(msg)}
              {msg.from === 'bot' && msg.showFeedback && (
                <FeedbackButtons
                  messageId={msg.id}
                  onRate={handleRate}
                  onChooseEditTarget={openEditChooser}
                />
              )}
            </div>
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
        {suggestion && (
          <div
            className="suggestion-chip"
            onClick={() => {
              setInput(suggestion)
              setSuggestion('')
              inputRef.current?.focus()
            }}
          >
            <div className="suggestion-chip-inner">
              <svg width="11" height="11" viewBox="0 0 16 16" fill="currentColor" style={{ flexShrink: 0, opacity: 0.5 }}>
                <path d="M8 1a7 7 0 110 14A7 7 0 018 1zm0 1.5a5.5 5.5 0 100 11 5.5 5.5 0 000-11zM7 5h2v4H7V5zm0 5h2v2H7v-2z" />
              </svg>
              <span className="suggestion-text">{suggestion}</span>
            </div>
            <kbd className="suggestion-tab-key">Tab ↵</kbd>
          </div>
        )}

        <div className="chatbot-input-row">
          <textarea
            ref={inputRef}
            className="chatbot-input"
            placeholder="Ask about the trials..."
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={typing}
          />
          <button className={`chatbot-send-btn ${input.trim() && !typing ? 'active' : ''}`} onClick={send} disabled={!input.trim() || typing}>
            {typing ? (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" style={{ animation: 'spin 0.8s linear infinite' }}>
                <path d="M12 2a10 10 0 0110 10" strokeLinecap="round" />
              </svg>
            ) : (
              <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                <path d="M1 1l14 7-14 7V9.5l10-1.5-10-1.5V1z" />
              </svg>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
