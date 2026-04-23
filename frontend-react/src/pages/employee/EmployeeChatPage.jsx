import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { Send, Bot, User, Copy, Check, Sparkles, LayoutDashboard, Trash2, Database } from 'lucide-react';
import { askQuery, getDatasets, startChatSession, endChatSession, requestPermission } from '../../services/api';

import EmployeeLayout from '../../layout/EmployeeLayout';

function RichText({ text }) {
  if (!text) return null;
  const lines = text.split('\n');
  return (
    <div style={{ lineHeight: 1.6 }}>
      {lines.map((line, i) => {
        const numbered = line.match(/^\s*(\d+)\.\s+(.*)/);
        if (numbered) return <div key={i} style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.25rem' }}><span style={{ color: 'var(--primary)', fontWeight: 700, minWidth: '1.4rem' }}>{numbered[1]}.</span><span dangerouslySetInnerHTML={{ __html: inlineMd(numbered[2]) }} /></div>;
        const bullet = line.match(/^\s*[•-]\s+(.*)/);
        if (bullet) return <div key={i} style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.25rem' }}><span style={{ color: 'var(--accent)', marginTop: 2 }}>•</span><span dangerouslySetInnerHTML={{ __html: inlineMd(bullet[1]) }} /></div>;
        if (!line.trim()) return <div key={i} style={{ height: '0.5rem' }} />;
        return <div key={i} style={{ marginBottom: '0.1rem' }} dangerouslySetInnerHTML={{ __html: inlineMd(line) }} />;
      })}
    </div>
  );
}

function inlineMd(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong style="color:#e6edf3">$1</strong>')
    .replace(/\*(.+?)\*/g, '<em style="color:#bc8cff">$1</em>')
    .replace(/`([^`]+)`/g, '<code style="background:rgba(255,255,255,0.1);padding:0.1em 0.35em;border-radius:4px;font-size:0.85em;font-family:monospace">$1</code>');
}

// Generic suggestions that work for any dataset
const BASE_SUGGESTIONS = [
  { icon: '📊', q: 'Give me a summary of this dataset' },
  { icon: '🔢', q: 'What columns are available?' },
  { icon: '📈', q: 'What is the total of the main numeric column?' },
  { icon: '⭐', q: 'Show the top 10 records by value' },
  { icon: '📉', q: 'What is the average of each numeric column?' },
  { icon: '🔍', q: 'How many rows are in this dataset?' },
  { icon: '⚠️', q: 'Are there any missing or null values?' },
  { icon: '📋', q: 'What are the unique categories?' },
];

const buildSmartSuggestions = (datasetInfo) => {
  const headers = datasetInfo?.headers || [];
  const lowerHeaders = headers.map(h => String(h).toLowerCase());
  const has = (pattern) => lowerHeaders.some(h => pattern.test(h));

  const suggestions = [
    { icon: '💡', q: 'What can I do to improve sales?' },
    { icon: '🎯', q: 'What should I focus on next?' },
    { icon: '📊', q: 'Give me a summary of this dataset' },
  ];

  if (has(/sales|revenue|turnover/)) {
    suggestions.push(
      { icon: '📈', q: 'Which product or category is driving the most sales?' },
      { icon: '🚀', q: 'How can I increase revenue?' },
    );
  }

  if (has(/profit|margin|earnings/)) {
    suggestions.push(
      { icon: '💰', q: 'What is the profit margin and how can I improve it?' },
      { icon: '🧪', q: 'Which products or categories are most profitable?' },
    );
  }

  if (has(/region|state|country|city|territory/)) {
    suggestions.push(
      { icon: '🗺️', q: 'Which region is underperforming?' },
      { icon: '📍', q: 'What is the best region to focus on next?' },
    );
  }

  if (has(/product|item|sku|category|subcategory|sub-category/)) {
    suggestions.push(
      { icon: '🏷️', q: 'What products should I promote more?' },
      { icon: '🛒', q: 'Which product category is losing momentum?' },
    );
  }

  if (has(/customer|client|buyer/)) {
    suggestions.push(
      { icon: '👥', q: 'Which customers are most valuable?' },
      { icon: '🔁', q: 'How can I improve customer retention?' },
    );
  }

  if (has(/date|month|year|time/)) {
    suggestions.push(
      { icon: '🕒', q: 'What are the monthly sales trends?' },
      { icon: '📅', q: 'When do sales peak and dip?' },
    );
  }

  if (has(/discount|rebate|markdown/)) {
    suggestions.push(
      { icon: '🏷️', q: 'Are discounts hurting profit?' },
      { icon: '📉', q: 'What discount strategy should I use?' },
    );
  }

  BASE_SUGGESTIONS.forEach(item => suggestions.push(item));

  const deduped = [];
  const seen = new Set();
  for (const item of suggestions) {
    if (!seen.has(item.q)) {
      seen.add(item.q);
      deduped.push(item);
    }
  }

  return deduped.slice(0, 10);
};

function getResponse(q) {
  return {
    text: "Please select a dataset from the dropdown above before asking questions.",
  };
}

function buildResponseHTML(r) {
  let html = `<div style="margin-bottom:10px">${r.text}</div>`;
  if (r.require_permission) {
    html += `<div style="margin-top:12px;padding:12px;background:rgba(210,153,34,0.1);border:1px solid rgba(210,153,34,0.4);border-radius:8px;text-align:center;">
      <div style="font-size:12px;color:var(--warning);margin-bottom:8px">You need <strong>${r.require_permission}</strong> access to perform this operation.</div>
      <button onclick="window.requestAccess('${r.require_permission}')" style="background:var(--warning);color:#000;border:none;padding:6px 12px;border-radius:4px;font-size:11px;font-weight:600;cursor:pointer;font-family:inherit">Request Access</button>
    </div>`;
  }

  if (r.table) {
    html += `<div style="margin-top:8px;border-radius:8px;overflow:hidden;border:1px solid rgba(48,54,61,0.8)"><table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:11px"><thead><tr>${r.table.cols.map(c => `<th style="background:rgba(13,17,23,0.95);padding:7px 10px;text-align:left;color:#8b949e;font-size:9px;letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid rgba(48,54,61,0.8)">${c}</th>`).join('')}</tr></thead><tbody>${r.table.rows.map(row => `<tr>${row.map(c => `<td style="padding:7px 10px;border-bottom:1px solid rgba(48,54,61,0.4);color:#c9d1d9">${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  }
  if (r.code) html += `<div style="background:rgba(13,17,23,0.95);border:1px solid rgba(48,54,61,0.8);border-radius:8px;padding:10px 12px;margin-top:8px;font-family:monospace;font-size:10px;color:#a5b4fc;overflow-x:auto"># Generated query\n${r.code}</div>`;
  if (r.insight) html += `<div style="margin-top:10px;padding:10px 12px;background:rgba(88,166,255,0.06);border-left:3px solid var(--primary);border-radius:0 8px 8px 0;font-size:12px;color:var(--text-main);line-height:1.6"><strong style="color:var(--primary)">✦ Insight: </strong>${r.insight}</div>`;
  return html;
}

function TypingDots() {
  return (
    <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
      <div style={{ width: 28, height: 28, borderRadius: 8, background: 'rgba(88,166,255,0.08)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
        <Bot size={14} color="var(--primary)" />
      </div>
      <div style={{ background: 'rgba(22,27,34,0.85)', border: '1px solid rgba(48,54,61,0.9)', borderRadius: '3px 12px 12px 12px', padding: '11px 14px', display: 'flex', gap: 4, alignItems: 'center' }}>
        {[0, 1, 2].map(i => (
          <span key={i} style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--primary)', animation: `dotPulse 1.2s ease-in-out ${i * 0.2}s infinite`, display: 'inline-block' }} />
        ))}
      </div>
    </div>
  );
}

const EmployeeChatPage = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const datasetId = searchParams.get('ds') || null;
  const [availableDatasets, setAvailableDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [datasetInfo, setDatasetInfo] = useState(null); // { headers, columnTypes, totalRows }
  const [messages, setMessages] = useState([{
    id: 0, role: 'ai',
    content: "👋 Hello! I'm your **AI Data Assistant**. Ask me about **totals**, **trends**, **top performers**, **anomalies**, **forecasts**, or anything about your data!\n\nSelect a dataset from the dropdown above and try one of the suggested questions on the left →",
    time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
  }]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedCols, setSelectedCols] = useState([]);
  const [queryHistory, setQueryHistory] = useState([]);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);
  const msgId = useRef(1);
  const api = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api';
  const smartSuggestions = useMemo(() => buildSmartSuggestions(datasetInfo), [datasetInfo]);

  // Fetch real column/row info for the selected dataset
  const fetchDatasetInfo = useCallback(async (dsId) => {
    if (!dsId) return;
    try {
      const token = sessionStorage.getItem('token');
      const res = await fetch(`${api}/cleaned-data/${dsId}?limit=1`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      const data = await res.json();
      if (data.success) {
        setDatasetInfo({
          headers: (data.headers || []).filter(h => h !== 'Unnamed: 0.1' && h !== 'Unnamed: 0'),
          columnTypes: data.columnTypes || {},
          totalRows: data.totalRows || 0,
        });
      }
    } catch (e) {
      console.warn('Could not load dataset info:', e.message);
    }
  }, [api]);

  useEffect(() => {
    const loadDatasets = async () => {
      try {
        const res = await getDatasets();
        if (res.success && res.data) {
          const readyDatasets = res.data.filter(d => d.status === 'completed' || d.status === 'ready' || d.status === 'cleaned');
          setAvailableDatasets(readyDatasets);

          if (!datasetId && readyDatasets.length > 0) {
            setSelectedDataset(readyDatasets[0]);
            fetchDatasetInfo(readyDatasets[0].dataset_id || readyDatasets[0].id);
          } else if (datasetId) {
            const selected = readyDatasets.find(d => (d.dataset_id || d.id) === datasetId);
            if (selected) { setSelectedDataset(selected); fetchDatasetInfo(datasetId); }
          }
        }
      } catch (err) {
        console.warn('Could not load datasets:', err.message);
      }
    };
    loadDatasets();
  }, [datasetId, fetchDatasetInfo]);


  useEffect(() => {
    if (selectedDataset) {
      const dsId = selectedDataset.dataset_id || selectedDataset.id;
      startChatSession(dsId).catch(console.error);
      
      return () => {
        endChatSession(dsId, 'closed').catch(console.error);
      };
    }
  }, [selectedDataset]);

  // Handle Dynamic Permission Request from HTML button
  useEffect(() => {
    window.requestAccess = async (type) => {
      const dsId = selectedDataset?.dataset_id || selectedDataset?.id;
      if (!dsId) return;
      
      try {
        const res = await requestPermission(dsId, type);
        if (res.success) {
          alert(`Success: Your request for ${type} access on ${selectedDataset.name} has been sent to the administrator.`);
        } else {
          alert(`Error: ${res.message || 'Failed to submit request'}`);
        }
      } catch (err) {
        alert("Failed to send request. Please try again.");
      }
    };
    return () => { delete window.requestAccess; };
  }, [selectedDataset]);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, isLoading]);


  const handleSend = useCallback(async (text) => {
    const msg = (text || input).trim();
    if (!msg || isLoading) return;
    
    const currentDs = selectedDataset;
    const dsId = currentDs?.dataset_id || currentDs?.id;
    
    setInput('');
    const id = msgId.current++;
    setMessages(prev => [...prev, { id, role: 'user', content: msg, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }]);
    setIsLoading(true);
    
    // Add to query history
    if (!queryHistory.includes(msg)) {
      setQueryHistory(prev => [msg, ...prev.slice(0, 4)]);
    }

    try {
      // Try to get real response from backend
      if (dsId) {
        const response = await askQuery(dsId, msg);
        let answer = response?.answer || "I'm sorry, I couldn't compute an answer for that.";
        const answerAlreadyHasSuggestions = /good follow-up questions|try asking next/i.test(answer);
        const suggested = Array.isArray(response?.suggested_questions) && response.suggested_questions.length > 0 && !answerAlreadyHasSuggestions
          ? `\n\n**Try asking next:**\n${response.suggested_questions.slice(0, 5).map(q => `• ${q}`).join('\n')}`
          : '';
        
        // Check for image-related errors
        if (answer.toLowerCase().includes("cannot read image") || 
            answer.toLowerCase().includes("does not support image") ||
            answer.toLowerCase().includes("model does not support image input")) {
          answer = "⚠️ **Image Input Not Supported**\n\nThe AI model does not support image input. Please ask questions using text only. For example:\n\n- 'What is the total revenue by region?'\n- 'Show me top 5 products'\n- 'What are the monthly trends?'\n\nYou can also switch to a text-capable model in the backend settings if available.";
        }
        
        const bid = msgId.current++;
        setMessages(prev => [...prev, { id: bid, role: 'ai', content: buildResponseHTML({ text: `${answer}${suggested}`, require_permission: response?.require_permission }), time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }]);
      } else {
        // Fallback to mock response
        const r = getResponse(msg);
        const bid = msgId.current++;
        setMessages(prev => [...prev, { id: bid, role: 'ai', content: buildResponseHTML(r), time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }]);
      }
    } catch (err) {
      console.error('Chat error:', err);
      const bid = msgId.current++;
      setMessages(prev => [...prev, { 
        id: bid, role: 'ai', 
        content: buildResponseHTML({ 
          text: "⚠️ **Connection Error**\n\nCould not reach the AI service. Please ensure the backend is running and try again.",
          insight: "Make sure the RAG server is running: `python ml_engine/rag_server.py`"
        }), 
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) 
      }]);
    }
    
    setIsLoading(false);
  }, [input, isLoading, selectedDataset, queryHistory]);

  const handleKeyDown = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } };
  const toggleCol = (col) => setSelectedCols(prev => prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col]);
  const clearChat = () => {
    setMessages([{ id: 0, role: 'ai', content: 'Chat cleared. Ask a new question about your dataset.', time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }]);
    msgId.current = 1;
  };

  const colTypeColors = { numeric: { border: 'rgba(63,185,80,0.2)', color: 'var(--success)', label: 'NUM' }, categorical: { border: 'rgba(188,140,255,0.2)', color: 'var(--accent)', label: 'CAT' }, date: { border: 'rgba(210,153,34,0.2)', color: 'var(--warning)', label: 'DATE' } };

  // Derive live columns from fetched dataset info
  const liveColumns = datasetInfo?.headers?.map(h => ({
    name: h,
    type: datasetInfo.columnTypes?.[h] || 'categorical',
  })) || [];

  return (
    <EmployeeLayout>
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Context Panel */}
        <div style={{ width: 260, background: 'rgba(22,27,34,0.7)', borderRight: '1px solid var(--border-color)', display: 'flex', flexDirection: 'column', flexShrink: 0, overflow: 'hidden' }}>
          <div style={{ padding: '16px 16px 12px', borderBottom: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#fff', marginBottom: 2 }}>Dataset Context</div>
            <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)' }}>Active for this session</div>
          </div>
          <div style={{ flex: 1, overflowY: 'auto', padding: 14 }}>
            {/* Dataset Info — Live from API */}
            <div style={{ background: 'rgba(13,17,23,0.95)', borderRadius: 10, padding: '12px 14px', marginBottom: 14 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: '#fff', marginBottom: 4, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {selectedDataset?.name || 'No dataset selected'}
              </div>
              <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)', lineHeight: 1.8 }}>
                {datasetInfo ? (
                  <>
                    <span style={{ color: 'var(--primary)' }}>{datasetInfo.totalRows.toLocaleString()}</span> rows ·{' '}
                    <span style={{ color: 'var(--primary)' }}>{datasetInfo.headers.length}</span> cols<br />
                    Status: <span style={{ color: 'var(--success)' }}>Cleaned ✓</span>
                  </>
                ) : (
                  <span style={{ color: 'var(--text-muted)' }}>{selectedDataset ? 'Loading...' : 'Select a dataset'}</span>
                )}
              </div>
            </div>

            {/* Columns — Live from dataset */}
            <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)', letterSpacing: 1.5, textTransform: 'uppercase', marginBottom: 8 }}>Columns</div>
            <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 8, color: 'var(--text-muted)', marginBottom: 6 }}>Click to reference in query</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginBottom: 6 }}>
              {liveColumns.length > 0 ? liveColumns.map(col => {
                const tc = colTypeColors[col.type] || colTypeColors['categorical'];
                const isSelected = selectedCols.includes(col.name);
                return (
                  <span
                    key={col.name}
                    onClick={() => toggleCol(col.name)}
                    style={{
                      fontFamily: "'DM Mono', monospace", fontSize: 9, padding: '3px 8px', borderRadius: 5,
                      cursor: 'pointer', transition: 'all 0.15s',
                      border: `1px solid ${isSelected ? 'var(--primary)' : tc.border}`,
                      background: isSelected ? 'rgba(88,166,255,0.08)' : 'rgba(255,255,255,0.04)',
                      color: isSelected ? 'var(--primary)' : tc.color,
                    }}
                  >
                    {col.name}
                  </span>
                );
              }) : (
                <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)', fontStyle: 'italic' }}>
                  {selectedDataset ? 'Loading columns...' : 'No dataset selected'}
                </span>
              )}
            </div>
            <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 8, color: 'var(--text-muted)', marginTop: 4 }}>
              <span style={{ color: 'var(--success)' }}>■</span> numeric &nbsp;
              <span style={{ color: 'var(--accent)' }}>■</span> categorical &nbsp;
              <span style={{ color: 'var(--warning)' }}>■</span> date
            </div>

            {/* Suggested Questions */}
            <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)', letterSpacing: 1.5, textTransform: 'uppercase', marginBottom: 8, marginTop: 16 }}>Suggested Questions</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
              {smartSuggestions.map((s, i) => (
                <div
                  key={i}
                  onClick={() => handleSend(s.q)}
                  style={{
                    padding: '8px 11px', borderRadius: 8, fontSize: 11.5, color: 'var(--text-main)',
                    border: '1px solid var(--border-color)', background: 'rgba(13,17,23,0.95)',
                    cursor: 'pointer', transition: 'all 0.15s', lineHeight: 1.4,
                  }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--primary)'; e.currentTarget.style.background = 'rgba(88,166,255,0.08)'; }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border-color)'; e.currentTarget.style.background = 'rgba(13,17,23,0.95)'; }}
                >
                  <span style={{ fontSize: 11, marginRight: 5 }}>{s.icon}</span>{s.q}
                </div>
              ))}
            </div>

            {/* Query History */}
            <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)', letterSpacing: 1.5, textTransform: 'uppercase', marginBottom: 8, marginTop: 16 }}>Query History</div>
            <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)', lineHeight: 2 }}>
              {queryHistory.map((q, i) => (
                <div key={i} style={{ cursor: 'pointer', color: 'var(--text-main)' }} onClick={() => handleSend(q)}>→ {q}</div>
              ))}
            </div>
          </div>
        </div>

        {/* Chat Main */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {/* Chat Topbar */}
          <div style={{
            padding: '14px 20px', background: 'rgba(13,17,23,0.8)', backdropFilter: 'blur(12px)',
            borderBottom: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', gap: 12, flexShrink: 0,
          }}>
            <div style={{ fontSize: 20 }}>◎</div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 15, fontWeight: 600, color: '#fff' }}>DataInsights Chatbot</div>
              {availableDatasets.length > 1 ? (
                <select className="emp-filter-select" value={selectedDataset?.dataset_id || selectedDataset?.id || ''}
                  onChange={(e) => {
                    const ds = availableDatasets.find(d => (d.dataset_id || d.id) === e.target.value);
                    if (ds) {
                      isInitialized?.current && (isInitialized.current = false);
                      setSelectedDataset(ds);
                      setDatasetInfo(null);
                      fetchDatasetInfo(ds.dataset_id || ds.id);
                      setMessages([{ id: 0, role: 'ai', content: `👋 Switched to **${ds.name}**. Ask me about **totals**, **trends**, **top performers**, or anything about your data!`, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }]);
                      msgId.current = 1;
                    }
                  }} style={{ marginTop: 4, fontSize: 10, minWidth: 150 }}>
                  {availableDatasets.map(ds => <option key={ds.dataset_id || ds.id} value={ds.dataset_id || ds.id}>{ds.name}</option>)}
                </select>
              ) : (
                <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)', marginTop: 2 }}>
                  {selectedDataset?.name || 'No dataset selected'}
                </div>
              )}
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="emp-btn emp-btn-ghost emp-btn-sm" onClick={clearChat}><Trash2 size={12} /> Clear</button>
              <button className="emp-btn emp-btn-ghost emp-btn-sm" onClick={() => navigate('/employee/dashboard')}><LayoutDashboard size={12} /> Dashboard</button>
            </div>
          </div>

          {/* Messages */}
          <div style={{ flex: 1, overflowY: 'auto', padding: 20, display: 'flex', flexDirection: 'column', gap: 16 }}>
            {messages.map((msg, idx) => {
              const isUser = msg.role === 'user';
              const isImageError = !isUser && msg.content && (msg.content.toLowerCase().includes("does not support image") || msg.content.toLowerCase().includes("cannot read image") || msg.content.toLowerCase().includes("image input not supported"));
              return (
                <div key={msg.id} style={{ display: 'flex', gap: 10, flexDirection: isUser ? 'row-reverse' : 'row', animation: 'adminFadeUp 0.25s ease', justifyContent: isImageError ? 'center' : 'flex-start' }}>
                  {!isImageError && (
                  <div style={{
                    width: 28, height: 28, borderRadius: isUser ? 7 : 8, flexShrink: 0,
                    background: isUser ? 'rgba(255,255,255,0.06)' : 'rgba(88,166,255,0.08)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center', marginTop: 2,
                  }}>
                    {isUser ? <User size={14} color="var(--text-muted)" /> : <Bot size={14} color="var(--primary)" />}
                  </div>
                  )}
                  <div style={{ maxWidth: isImageError ? '70%' : '75%' }}>
                    {isImageError ? (
                      <div style={{
                        padding: '20px 24px', fontSize: 13, lineHeight: 1.6,
                        borderRadius: 16,
                        background: 'linear-gradient(135deg, rgba(210,153,34,0.15), rgba(248,81,73,0.1))',
                        border: '1px solid rgba(210,153,34,0.4)',
                        color: '#fff',
                        textAlign: 'center',
                        boxShadow: '0 4px 16px rgba(0,0,0,0.3)',
                      }}>
                        <RichText text={msg.content} />
                      </div>
                    ) : (
                    <div style={{
                      padding: '11px 14px', fontSize: 13, lineHeight: 1.65,
                      borderRadius: isUser ? '12px 3px 12px 12px' : '3px 12px 12px 12px',
                      background: isUser ? 'linear-gradient(135deg, rgba(31,111,235,0.25), rgba(88,166,255,0.15))' : 'rgba(22,27,34,0.85)',
                      border: `1px solid ${isUser ? 'rgba(88,166,255,0.25)' : 'rgba(48,54,61,0.9)'}`,
                      color: isUser ? '#fff' : 'var(--text-main)',
                      textAlign: isUser ? 'right' : 'left',
                      boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
                    }}>
                      {isUser ? <p style={{ margin: 0 }}>{msg.content}</p> : <RichText text={msg.content} />}
                    </div>
                    )}
                    {!isImageError && <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)', marginTop: 4, padding: '0 2px', textAlign: isUser ? 'right' : 'left' }}>{msg.time}</div>}
                  </div>
                </div>
              );
            })}
            {isLoading && <TypingDots />}
            <div ref={bottomRef} />
          </div>

          {/* Input Bar */}
          <div style={{ padding: '14px 16px', borderTop: '1px solid var(--border-color)', background: 'rgba(13,17,23,0.95)', flexShrink: 0 }}>
            <div style={{
              display: 'flex', gap: 10, alignItems: 'flex-end',
              background: 'rgba(22,27,34,0.8)', border: '1px solid var(--border-color)',
              borderRadius: 12, padding: '10px 14px', transition: 'border-color 0.2s',
            }}>
              <textarea
                ref={inputRef}
                rows={1}
                value={input}
                onChange={e => { setInput(e.target.value); e.target.style.height = 'auto'; e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px'; }}
                onKeyDown={handleKeyDown}
                placeholder="Ask anything… e.g. What is the average revenue per region?"
                disabled={isLoading}
                style={{
                  flex: 1, background: 'transparent', border: 'none', outline: 'none',
                  color: '#e6edf3', fontSize: 13, fontFamily: 'var(--font-family)',
                  resize: 'none', maxHeight: 120, lineHeight: 1.5,
                }}
              />
              <button
                onClick={() => handleSend()}
                disabled={!input.trim() || isLoading}
                style={{
                  width: 42, height: 42, borderRadius: 10,
                  background: input.trim() && !isLoading ? 'linear-gradient(135deg, var(--secondary), var(--primary))' : 'rgba(255,255,255,0.05)',
                  border: 'none', cursor: input.trim() && !isLoading ? 'pointer' : 'not-allowed',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  transition: 'all 0.15s', flexShrink: 0,
                  boxShadow: input.trim() && !isLoading ? '0 2px 8px rgba(88,166,255,0.3)' : 'none',
                }}
              >
                <Send size={16} color={input.trim() && !isLoading ? '#fff' : 'var(--text-muted)'} />
              </button>
            </div>
            <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)', marginTop: 7, display: 'flex', gap: 12 }}>
              <span>⏎ Send · Shift+⏎ New line</span>
              <span style={{ marginLeft: 'auto' }}>Results are computed from your actual dataset</span>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes dotPulse {
          0%, 80%, 100% { opacity: 0.3; transform: scale(0.85); }
          40% { opacity: 1; transform: scale(1.1); }
        }
      `}</style>
    </EmployeeLayout>
  );
};

export default EmployeeChatPage;
