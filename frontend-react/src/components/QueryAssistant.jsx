import { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Bot, User, Copy, Check, Sparkles, ChevronDown } from 'lucide-react';
import { askQuery } from '../services/api';

/* ── Markdown-like renderer (bold, bullets, numbered lists) ── */
function RichText({ text }) {
  if (!text) return null;

  const lines = text.split('\n');
  const elements = [];

  lines.forEach((line, i) => {
    // Numbered list
    const numberedMatch = line.match(/^\s*(\d+)\.\s+(.*)/);
    if (numberedMatch) {
      elements.push(
        <div key={i} style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.25rem' }}>
          <span style={{ color: 'var(--primary)', fontWeight: 700, minWidth: '1.4rem' }}>{numberedMatch[1]}.</span>
          <span dangerouslySetInnerHTML={{ __html: inlineMd(numberedMatch[2]) }} />
        </div>
      );
      return;
    }

    // Bullet list (• or -)
    const bulletMatch = line.match(/^\s*[•-]\s+(.*)/);
    if (bulletMatch) {
      elements.push(
        <div key={i} style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.25rem' }}>
          <span style={{ color: 'var(--accent)', marginTop: '2px' }}>•</span>
          <span dangerouslySetInnerHTML={{ __html: inlineMd(bulletMatch[1]) }} />
        </div>
      );
      return;
    }

    // Empty line → spacer
    if (!line.trim()) {
      elements.push(<div key={i} style={{ height: '0.5rem' }} />);
      return;
    }

    // Normal line
    elements.push(
      <div key={i} style={{ marginBottom: '0.1rem' }}
        dangerouslySetInnerHTML={{ __html: inlineMd(line) }}
      />
    );
  });

  return <div style={{ lineHeight: 1.6 }}>{elements}</div>;
}

// Inline markdown: **bold**, *italic*, `code`
function inlineMd(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong style="color:#e6edf3">$1</strong>')
    .replace(/\*(.+?)\*/g, '<em style="color:#bc8cff">$1</em>')
    .replace(/`([^`]+)`/g, '<code style="background:rgba(255,255,255,0.1);padding:0.1em 0.35em;border-radius:4px;font-size:0.85em;font-family:monospace">$1</code>');
}

/* ── Suggestion Chips ── */
const SUGGESTIONS = [
  "Give me a summary of the dataset",
  "What is the total sales?",
  "Show me top 5 products",
  "Monthly trend analysis",
  "Are there any anomalies?",
  "What are the AI insights?",
  "Which region performs best?",
  "Forecast next month",
];

/* ── Typing Animation Hook ── */
function useTypingEffect(text, speed = 8, enabled = true) {
  const [displayed, setDisplayed] = useState('');
  const [done, setDone] = useState(false);

  useEffect(() => {
    if (!enabled || !text) {
      setDisplayed(text || '');
      setDone(true);
      return;
    }
    setDisplayed('');
    setDone(false);

    // Chunk reveal (fast batches for no-lag effect)
    let idx = 0;
    const CHUNK = 5; // chars per tick
    const id = setInterval(() => {
      idx = Math.min(idx + CHUNK, text.length);
      setDisplayed(text.slice(0, idx));
      if (idx >= text.length) {
        clearInterval(id);
        setDone(true);
      }
    }, speed);
    return () => clearInterval(id);
  }, [text, speed, enabled]);

  return { displayed, done };
}

/* ── Single Bot Message with typing ── */
function BotMessage({ msg, isLatest }) {
  const { displayed, done } = useTypingEffect(msg.content, 6, isLatest);
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(msg.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const shownText = isLatest ? displayed : msg.content;

  return (
    <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start' }}>
      {/* Avatar */}
      <div style={{
        width: 36, height: 36, borderRadius: '50%', flexShrink: 0,
        background: 'linear-gradient(135deg, #1f6feb, #bc8cff)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        boxShadow: '0 0 12px rgba(88,166,255,0.3)',
      }}>
        <Bot size={18} color="#fff" />
      </div>

      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{
          background: 'rgba(22,27,34,0.85)',
          border: '1px solid rgba(48,54,61,0.9)',
          borderRadius: '4px 14px 14px 14px',
          padding: '0.9rem 1.1rem',
          position: 'relative',
          boxShadow: '0 2px 12px rgba(0,0,0,0.25)',
        }}>
          <RichText text={shownText} />
          {isLatest && !done && (
            <span style={{
              display: 'inline-block', width: 8, height: 8,
              background: 'var(--primary)', borderRadius: '50%',
              marginLeft: 4, verticalAlign: 'middle',
              animation: 'cursorBlink 0.8s ease-in-out infinite',
            }} />
          )}

          {/* Meta row */}
          {done && (
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              marginTop: '0.6rem', paddingTop: '0.5rem',
              borderTop: '1px solid rgba(48,54,61,0.5)',
            }}>
              <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
                {msg.intent && msg.intent !== 'unknown' && msg.intent !== 'error' && (
                  <span style={{
                    fontSize: '0.68rem', padding: '0.15rem 0.5rem',
                    borderRadius: '20px', background: 'rgba(88,166,255,0.12)',
                    color: 'var(--primary)', border: '1px solid rgba(88,166,255,0.2)',
                  }}>
                    {msg.intent}
                  </span>
                )}
                {msg.confidence != null && (
                  <span style={{
                    fontSize: '0.68rem', padding: '0.15rem 0.5rem',
                    borderRadius: '20px', background: 'rgba(188,140,255,0.1)',
                    color: 'var(--accent)', border: '1px solid rgba(188,140,255,0.2)',
                  }}>
                    {Math.round(msg.confidence * 100)}% confidence
                  </span>
                )}
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>{msg.time}</span>
                <button onClick={handleCopy} title="Copy" style={{
                  background: 'none', border: 'none', cursor: 'pointer',
                  color: copied ? 'var(--success)' : 'var(--text-muted)',
                  padding: '0.2rem', borderRadius: '4px', transition: 'color 0.2s',
                  display: 'flex', alignItems: 'center',
                }}>
                  {copied ? <Check size={13} /> : <Copy size={13} />}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ── User Message ── */
function UserMessage({ msg }) {
  return (
    <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start', flexDirection: 'row-reverse' }}>
      <div style={{
        width: 36, height: 36, borderRadius: '50%', flexShrink: 0,
        background: 'linear-gradient(135deg, #58a6ff, #1f6feb)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        <User size={18} color="#fff" />
      </div>

      <div style={{
        background: 'linear-gradient(135deg, rgba(31,111,235,0.25), rgba(88,166,255,0.15))',
        border: '1px solid rgba(88,166,255,0.25)',
        borderRadius: '14px 4px 14px 14px',
        padding: '0.9rem 1.1rem',
        maxWidth: '75%',
        boxShadow: '0 2px 8px rgba(88,166,255,0.08)',
      }}>
        <p style={{ margin: 0, lineHeight: 1.55 }}>{msg.content}</p>
        <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: '0.4rem', textAlign: 'right' }}>{msg.time}</div>
      </div>
    </div>
  );
}

/* ── Typing Dots (Loading) ── */
function TypingDots() {
  return (
    <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
      <div style={{
        width: 36, height: 36, borderRadius: '50%',
        background: 'linear-gradient(135deg, #1f6feb, #bc8cff)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        boxShadow: '0 0 12px rgba(88,166,255,0.3)',
      }}>
        <Bot size={18} color="#fff" />
      </div>
      <div style={{
        background: 'rgba(22,27,34,0.85)', border: '1px solid rgba(48,54,61,0.9)',
        borderRadius: '4px 14px 14px 14px', padding: '0.9rem 1.2rem',
        display: 'flex', gap: '0.4rem', alignItems: 'center',
      }}>
        {[0, 1, 2].map(i => (
          <span key={i} style={{
            width: 7, height: 7, borderRadius: '50%',
            background: 'var(--primary)',
            animation: `dotPulse 1.2s ease-in-out ${i * 0.2}s infinite`,
            display: 'inline-block',
          }} />
        ))}
      </div>
    </div>
  );
}

/* ── Main Component ── */
const QueryAssistant = ({ datasetId }) => {
  const getTime = () => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  const [messages, setMessages] = useState([{
    id: 0, role: 'bot',
    content: "👋 Hello! I'm your **AI Data Assistant**. I've analyzed your dataset and I'm ready to answer questions.\n\nAsk me about totals, trends, top performers, anomalies, forecasts, or anything about your data!",
    time: getTime(),
    intent: 'greeting', confidence: 1.0,
  }]);
  const [input, setInput]         = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const [msgIdCounter, setMsgIdCounter]       = useState(1);

  const bottomRef  = useRef(null);
  const inputRef   = useRef(null);
  const listRef    = useRef(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSend = useCallback(async (text) => {
    const msg = (text || input).trim();
    if (!msg || !datasetId || isLoading) return;

    setInput('');
    setShowSuggestions(false);
    const userMsgId = msgIdCounter;
    setMsgIdCounter(c => c + 1);

    setMessages(prev => [...prev, {
      id: userMsgId, role: 'user', content: msg, time: getTime()
    }]);
    setIsLoading(true);

    try {
      const response = await askQuery(datasetId, msg);
      let answer = response?.answer || "I'm sorry, I couldn't compute an answer for that.";
      
      // Check for image-related errors
      if (answer.toLowerCase().includes("cannot read image") || 
          answer.toLowerCase().includes("does not support image") ||
          answer.toLowerCase().includes("model does not support image input")) {
        answer = "⚠️ **Image Input Not Supported**\n\nThe AI model does not support image input. Please ask questions using text only. For example:\n\n- 'What is the total revenue by region?'\n- 'Show me top 5 products'\n- 'What are the monthly trends?'\n\nYou can also switch to a text-capable model in the backend settings if available.";
      }
      
      const botMsgId = msgIdCounter + 1;
      setMsgIdCounter(c => c + 2);

      setMessages(prev => [...prev, {
        id: botMsgId, role: 'bot', content: answer, time: getTime(),
        intent:     response?.intent,
        confidence: response?.confidence,
        isLatest:   true,
      }]);
    } catch (err) {
      console.error('[QueryAssistant]', err);
      const errMsgId = msgIdCounter + 1;
      setMsgIdCounter(c => c + 2);
      setMessages(prev => [...prev, {
        id: errMsgId, role: 'bot', time: getTime(),
        content: '⚠️ **Connection error.** Could not reach the query engine. Please ensure the backend is running and the dataset has been processed.',
        intent: 'error',
      }]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  }, [input, datasetId, isLoading, msgIdCounter]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Auto-resize textarea
  const handleInputChange = (e) => {
    setInput(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
  };

  return (
    <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden', position: 'relative' }}>

      {/* ── Inline Styles ── */}
      <style>{`
        @keyframes dotPulse {
          0%, 80%, 100% { opacity: 0.3; transform: scale(0.85); }
          40% { opacity: 1; transform: scale(1.1); }
        }
        @keyframes cursorBlink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0; }
        }
        @keyframes suggestionFadeIn {
          from { opacity: 0; transform: translateY(6px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .chat-input:focus {
          border-color: var(--primary) !important;
          box-shadow: 0 0 0 3px rgba(88,166,255,0.15) !important;
        }
        .suggestion-chip:hover {
          background: rgba(88,166,255,0.2) !important;
          border-color: rgba(88,166,255,0.5) !important;
          transform: translateY(-1px);
          color: #fff !important;
        }
        .send-btn:hover:not(:disabled) {
          transform: scale(1.05);
          box-shadow: 0 4px 16px rgba(88,166,255,0.4) !important;
        }
        .send-btn:active:not(:disabled) { transform: scale(0.97); }
      `}</style>

      {/* ── Header ── */}
      <div style={{
        padding: '1rem 1.25rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        borderBottom: '1px solid var(--border-color)',
        background: 'rgba(22,27,34,0.6)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{
            width: 40, height: 40, borderRadius: '50%',
            background: 'linear-gradient(135deg, #1f6feb, #bc8cff)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 0 16px rgba(88,166,255,0.35)',
          }}>
            <Sparkles size={20} color="#fff" />
          </div>
          <div>
            <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 700 }}>AI Data Assistant</h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginTop: '2px' }}>
              <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#3fb950', display: 'inline-block' }} />
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Online · Analyzing dataset</span>
            </div>
          </div>
        </div>

        <button
          onClick={() => setShowSuggestions(s => !s)}
          title="Toggle suggestions"
          style={{
            background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border-color)',
            borderRadius: '8px', padding: '0.35rem 0.6rem', cursor: 'pointer',
            color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.35rem',
            fontSize: '0.75rem', transition: 'all 0.2s',
          }}
        >
          <ChevronDown size={14} style={{ transform: showSuggestions ? 'rotate(180deg)' : '', transition: 'transform 0.2s' }} />
          Suggestions
        </button>
      </div>

      {/* ── Suggestions Chips ── */}
      {showSuggestions && (
        <div style={{
          padding: '0.75rem 1.25rem', display: 'flex', flexWrap: 'wrap', gap: '0.5rem',
          borderBottom: '1px solid var(--border-color)',
          background: 'rgba(0,0,0,0.15)',
          animation: 'suggestionFadeIn 0.3s ease forwards',
        }}>
          {SUGGESTIONS.map((s, i) => (
            <button
              key={i}
              className="suggestion-chip"
              onClick={() => handleSend(s)}
              disabled={isLoading}
              style={{
                background: 'rgba(88,166,255,0.08)', border: '1px solid rgba(88,166,255,0.2)',
                borderRadius: '20px', padding: '0.3rem 0.75rem',
                color: 'var(--text-muted)', fontSize: '0.78rem', cursor: 'pointer',
                transition: 'all 0.2s', fontFamily: 'var(--font-family)',
                whiteSpace: 'nowrap',
              }}
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* ── Messages List ── */}
      <div
        ref={listRef}
        style={{
          flexGrow: 1, overflowY: 'auto', padding: '1.25rem',
          display: 'flex', flexDirection: 'column', gap: '1.25rem',
        }}
      >
        {messages.map((msg, idx) => {
          const isLatestBot = msg.role === 'bot' && idx === messages.length - 1;
          if (msg.role === 'user') {
            return <UserMessage key={msg.id} msg={msg} />;
          }
          return <BotMessage key={msg.id} msg={msg} isLatest={isLatestBot} />;
        })}

        {isLoading && <TypingDots />}
        <div ref={bottomRef} />
      </div>

      {/* ── Input Bar ── */}
      <div style={{
        padding: '1rem 1.25rem', borderTop: '1px solid var(--border-color)',
        background: 'rgba(13,17,23,0.7)',
      }}>
        <div style={{
          display: 'flex', gap: '0.75rem', alignItems: 'flex-end',
          background: 'rgba(22,27,34,0.8)', border: '1px solid var(--border-color)',
          borderRadius: '12px', padding: '0.6rem 0.75rem',
          transition: 'border-color 0.2s, box-shadow 0.2s',
        }}
          onFocus={() => {}}
          onBlur={() => {}}
        >
          <textarea
            ref={inputRef}
            rows={1}
            className="chat-input"
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything about your dataset..."
            disabled={isLoading}
            style={{
              flex: 1, background: 'transparent', border: 'none', outline: 'none',
              color: '#e6edf3', fontSize: '0.95rem', fontFamily: 'var(--font-family)',
              resize: 'none', lineHeight: 1.55, minHeight: '24px', maxHeight: '120px',
              padding: 0,
            }}
          />
          <button
            className="send-btn"
            onClick={() => handleSend()}
            disabled={!input.trim() || isLoading}
            style={{
              background: input.trim() && !isLoading
                ? 'linear-gradient(135deg, #1f6feb, #58a6ff)'
                : 'rgba(255,255,255,0.05)',
              border: 'none', borderRadius: '8px',
              width: 36, height: 36, display: 'flex', alignItems: 'center', justifyContent: 'center',
              cursor: input.trim() && !isLoading ? 'pointer' : 'not-allowed',
              transition: 'all 0.2s', flexShrink: 0,
              boxShadow: input.trim() && !isLoading ? '0 2px 8px rgba(88,166,255,0.3)' : 'none',
            }}
          >
            <Send size={16} color={input.trim() && !isLoading ? '#fff' : 'var(--text-muted)'} />
          </button>
        </div>
        <p style={{ margin: '0.4rem 0 0', fontSize: '0.7rem', color: 'var(--text-muted)', textAlign: 'center' }}>
          Press Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  );
};

export default QueryAssistant;
