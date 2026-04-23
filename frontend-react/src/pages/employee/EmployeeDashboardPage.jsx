import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, PieChart, Pie, AreaChart, Area
} from 'recharts';
import {
  Upload, Database, BarChart3, MessageSquare, Sparkles, FileText,
  ArrowRight, Clock, CheckCircle2, AlertCircle, TrendingUp,
  Loader, RefreshCw, Activity, Layers, HardDrive
} from 'lucide-react';
import EmployeeLayout from '../../layout/EmployeeLayout';
import { getDatasets, getUserStats } from '../../services/api';

const COLORS = ['#58a6ff', '#3fb950', '#bc8cff', '#d29922', '#f85149', '#79c0ff', '#d2a8ff', '#ffa657'];

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: 'rgba(22,27,34,0.96)', border: '1px solid var(--border-color)',
      borderRadius: 8, padding: '8px 12px', fontSize: 12, maxWidth: 220,
    }}>
      {label && <div style={{ color: 'var(--text-muted)', marginBottom: 4, fontSize: 11 }}>{label}</div>}
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || '#58a6ff', fontWeight: 600 }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toLocaleString() : p.value}
        </div>
      ))}
    </div>
  );
};

const StatusBadge = ({ status }) => {
  const config = {
    cleaned: { color: '#3fb950', bg: 'rgba(63,185,80,0.1)', icon: CheckCircle2, label: 'Cleaned' },
    processing: { color: '#d29922', bg: 'rgba(210,153,34,0.1)', icon: Loader, label: 'Cleaning' },
    cleaning: { color: '#d29922', bg: 'rgba(210,153,34,0.1)', icon: Loader, label: 'Cleaning' },
    failed: { color: '#f85149', bg: 'rgba(248,81,73,0.1)', icon: AlertCircle, label: 'Failed' },
    not_cleaned: { color: 'rgba(139,148,158,0.8)', bg: 'rgba(139,148,158,0.1)', icon: Database, label: 'Not Cleaned' },
    // Legacy
    completed: { color: '#3fb950', bg: 'rgba(63,185,80,0.1)', icon: CheckCircle2, label: 'Cleaned' },
    ready: { color: '#3fb950', bg: 'rgba(63,185,80,0.1)', icon: CheckCircle2, label: 'Cleaned' },
  };
  const c = config[status] || config.cleaned;
  const Icon = c.icon;
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 4, padding: '3px 10px',
      borderRadius: 20, background: c.bg, color: c.color,
      fontSize: 10, fontWeight: 600, fontFamily: "'DM Mono', monospace",
    }}>
      <Icon size={10} style={status === 'processing' ? { animation: 'spin 1s linear infinite' } : {}} />
      {c.label}
    </span>
  );
};

const EmployeeDashboardPage = () => {
  const navigate = useNavigate();
  const userName = sessionStorage.getItem('userName') || 'Employee';
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    loadDashboard();
  }, []);

  const loadDashboard = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await getDatasets();
      if (res.success && res.data) {
        setDatasets(res.data);
      }
    } catch (err) {
      console.warn('Dashboard load error:', err.message);
    }
    setLoading(false);
  };

  // Compute stats from real datasets
  const totalDatasets = datasets.length;
  const readyDatasets = datasets.filter(d => d.status === 'completed' || d.status === 'ready' || d.status === 'cleaned');
  const processingDatasets = datasets.filter(d => d.status === 'processing' || d.status === 'cleaning');
  const failedDatasets = datasets.filter(d => d.status === 'failed');

  // Status distribution for pie chart
  const statusData = [
    { name: 'Cleaned', value: readyDatasets.length, color: '#3fb950' },
    { name: 'Processing', value: processingDatasets.length, color: '#d29922' },
    { name: 'Failed', value: failedDatasets.length, color: '#f85149' },
  ].filter(d => d.value > 0);

  // Timeline data — group datasets by creation date
  const timelineData = (() => {
    const grouped = {};
    datasets.forEach(d => {
      const date = d.created_at ? new Date(d.created_at).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' }) : 'Unknown';
      grouped[date] = (grouped[date] || 0) + 1;
    });
    return Object.entries(grouped)
      .map(([date, count]) => ({ date, uploads: count }))
      .slice(-7);
  })();

  // Greeting based on time of day
  const hour = new Date().getHours();
  const greeting = hour < 12 ? 'Good Morning' : hour < 17 ? 'Good Afternoon' : 'Good Evening';

  // Quick actions
  const quickActions = [
    { icon: Database, label: 'My Datasets', desc: 'View all processed datasets', path: '/employee/datasets', color: '#3fb950' },
    { icon: Sparkles, label: 'Analysis', desc: 'Deep-dive into your data', path: '/employee/analysis', color: '#bc8cff' },
    { icon: BarChart3, label: 'Visualization', desc: 'Charts, graphs & filters', path: '/employee/visualization', color: '#d29922' },
    { icon: MessageSquare, label: 'AI Chatbot', desc: 'Ask questions about data', path: '/employee/chat', color: '#79c0ff' },
    { icon: FileText, label: 'Summary', desc: 'Executive report generation', path: '/employee/summary', color: '#d2a8ff' },
  ];

  return (
    <EmployeeLayout>
      <div className="emp-topbar">
        <div>
          <div className="emp-topbar-title">Dashboard</div>
          <div className="emp-topbar-sub">Enterprise Analytics Overview</div>
        </div>
        <div className="emp-topbar-actions">
          <button className="emp-btn emp-btn-ghost emp-btn-sm" onClick={loadDashboard}>
            <RefreshCw size={12} /> Refresh
          </button>
        </div>
      </div>

      <div className="emp-content">

        {/* ── Welcome Banner ── */}
        <div className="glass-panel" style={{
          padding: '28px 32px', marginBottom: 20,
          background: 'linear-gradient(135deg, rgba(88,166,255,0.08) 0%, rgba(188,140,255,0.05) 100%)',
          borderLeft: '3px solid #58a6ff',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div>
              <div style={{ fontSize: 22, fontWeight: 700, color: '#fff', marginBottom: 4 }}>
                {greeting}, {userName} 👋
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: 13, lineHeight: 1.6 }}>
                {totalDatasets === 0
                  ? 'Welcome to DataInsights.ai! Upload your first dataset to get started with automated analytics.'
                  : `You have ${totalDatasets} dataset${totalDatasets !== 1 ? 's' : ''} — ${readyDatasets.length} cleaned and ready for analysis${processingDatasets.length > 0 ? `, ${processingDatasets.length} in progress` : ''}.`
                }
              </div>
            </div>
          </div>
        </div>

        {/* ── KPI Stats Cards ── */}
        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: 12, marginBottom: 20,
        }}>
          {[
            { icon: Database, label: 'Total Datasets', value: totalDatasets, color: '#58a6ff', sub: 'All time' },
            { icon: CheckCircle2, label: 'Cleaned Data', value: readyDatasets.length, color: '#3fb950', sub: totalDatasets > 0 ? `${Math.round((readyDatasets.length/totalDatasets)*100)}%` : '0%' },
            { icon: Loader, label: 'Processing', value: processingDatasets.length, color: '#d29922', sub: 'In queue' },
            { icon: AlertCircle, label: 'Failed', value: failedDatasets.length, color: '#f85149', sub: totalDatasets > 0 ? `${Math.round((failedDatasets.length/totalDatasets)*100)}%` : '0%' },
          ].map((kpi, i) => {
            const Icon = kpi.icon;
            return (
              <div key={i} className="glass-panel" style={{
                padding: '18px 20px',
                animation: `adminFadeUp 0.4s ease both`,
                animationDelay: `${i * 0.06}s`,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <Icon size={16} color={kpi.color} />
                  <span style={{
                    fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase',
                    fontFamily: "'DM Mono', monospace", letterSpacing: 0.5,
                  }}>{kpi.label}</span>
                </div>
                <div style={{ fontSize: 28, fontWeight: 700, color: kpi.color }}>{kpi.value}</div>
                <div style={{ fontSize: 9, color: 'var(--text-muted)', marginTop: 4, fontFamily: "'DM Mono', monospace" }}>{kpi.sub}</div>
              </div>
            );
          })}
        </div>

        {/* ── Quick Stats Summary ── */}
        {totalDatasets > 0 && (
          <div className="glass-panel" style={{ padding: '14px 20px', marginBottom: 20, display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Activity size={14} color="#58a6ff" />
              <span style={{ fontSize: 12, fontWeight: 600, color: '#fff' }}>Dashboard Analytics</span>
            </div>
            <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: '#58a6ff' }}>{totalDatasets}</div>
                <div style={{ fontSize: 9, color: 'var(--text-muted)', fontFamily: "'DM Mono', monospace" }}>Total Datasets</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: '#3fb950' }}>{readyDatasets.length}</div>
                <div style={{ fontSize: 9, color: 'var(--text-muted)', fontFamily: "'DM Mono', monospace" }}>Cleaned</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: '#d29922' }}>{processingDatasets.length}</div>
                <div style={{ fontSize: 9, color: 'var(--text-muted)', fontFamily: "'DM Mono', monospace" }}>Processing</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: '#bc8cff' }}>{timelineData.length > 0 ? timelineData.reduce((a,b) => a + b.uploads, 0) : 0}</div>
                <div style={{ fontSize: 9, color: 'var(--text-muted)', fontFamily: "'DM Mono', monospace" }}>This Week</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: '#f85149' }}>{failedDatasets.length}</div>
                <div style={{ fontSize: 9, color: 'var(--text-muted)', fontFamily: "'DM Mono', monospace" }}>Failed</div>
              </div>
            </div>
          </div>
        )}

        {/* ── Quick Actions Grid ── */}
        <div style={{ marginBottom: 20 }}>
          <div style={{
            fontSize: 14, fontWeight: 700, color: '#fff', marginBottom: 12,
            display: 'flex', alignItems: 'center', gap: 8,
          }}>
            <Activity size={16} color="#58a6ff" /> Quick Actions
          </div>
          <div style={{
            display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
            gap: 10,
          }}>
            {quickActions.map((action, i) => {
              const Icon = action.icon;
              return (
                <div key={i} className="glass-panel" onClick={() => navigate(action.path)}
                  style={{
                    padding: '18px 20px', cursor: 'pointer',
                    transition: 'all 0.2s ease', borderLeft: `2px solid ${action.color}`,
                    animation: `adminFadeUp 0.4s ease both`,
                    animationDelay: `${0.3 + i * 0.05}s`,
                  }}
                  onMouseEnter={e => {
                    e.currentTarget.style.background = 'rgba(88,166,255,0.05)';
                    e.currentTarget.style.transform = 'translateY(-2px)';
                  }}
                  onMouseLeave={e => {
                    e.currentTarget.style.background = '';
                    e.currentTarget.style.transform = '';
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
                    <Icon size={18} color={action.color} />
                    <span style={{ fontSize: 13, fontWeight: 600, color: '#fff' }}>{action.label}</span>
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.4 }}>{action.desc}</div>
                  <div style={{ marginTop: 8, fontSize: 10, color: action.color, display: 'flex', alignItems: 'center', gap: 4 }}>
                    Open <ArrowRight size={10} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* ── Charts Row ── */}
        {totalDatasets > 0 && (
          <div style={{
            display: 'grid', gridTemplateColumns: statusData.length > 1 ? '1.5fr 1fr' : '1fr',
            gap: 12, marginBottom: 20,
          }}>
            {/* Upload Timeline */}
            {timelineData.length > 1 && (
              <div className="glass-panel" style={{ padding: '16px 20px' }}>
                <div style={{ fontSize: 13, fontWeight: 700, color: '#fff', marginBottom: 4 }}>Upload Activity</div>
                <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)', marginBottom: 12 }}>
                  Datasets uploaded over time
                </div>
                <div style={{ height: 160 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={timelineData}>
                      <defs>
                        <linearGradient id="dashGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#58a6ff" stopOpacity={0.5} />
                          <stop offset="50%" stopColor="#58a6ff" stopOpacity={0.25} />
                          <stop offset="100%" stopColor="#58a6ff" stopOpacity={0.02} />
                        </linearGradient>
                        <linearGradient id="dashStroke" x1="0" y1="0" x2="1" y2="0">
                          <stop offset="0%" stopColor="#58a6ff" />
                          <stop offset="100%" stopColor="#3fb950" />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                      <XAxis dataKey="date" tick={{ fill: '#3d4f6e', fontSize: 9 }} />
                      <YAxis tick={{ fill: '#3d4f6e', fontSize: 9 }} allowDecimals={false} />
                      <Tooltip content={<CustomTooltip />} />
                      <Area type="monotone" dataKey="uploads" name="Uploads" stroke="url(#dashStroke)" fill="url(#dashGrad)" strokeWidth={3} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Status Distribution */}
            {statusData.length > 1 && (
              <div className="glass-panel" style={{ padding: '16px 20px' }}>
                <div style={{ fontSize: 13, fontWeight: 700, color: '#fff', marginBottom: 4 }}>Dataset Status</div>
                <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)', marginBottom: 12 }}>
                  Current pipeline distribution
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                  <div style={{ width: 130, height: 130 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <defs>
                          {statusData.map((entry, i) => (
                            <linearGradient key={i} id={`dashPieGrad${i}`} x1="0" y1="0" x2="1" y2="1">
                              <stop offset="0%" stopColor={entry.color} stopOpacity={1} />
                              <stop offset="100%" stopColor={entry.color} stopOpacity={0.6} />
                            </linearGradient>
                          ))}
                        </defs>
                        <Pie data={statusData} dataKey="value" nameKey="name" innerRadius={40} outerRadius={60} paddingAngle={3}
                          label={({ name, percent }) => `${(percent * 100).toFixed(0)}%`}
                          labelLine={{ stroke: '#6b7694', strokeWidth: 1 }}>
                          {statusData.map((entry, i) => (
                            <Cell key={i} fill={`url(#dashPieGrad${i})`} stroke="rgba(22,27,34,0.5)" strokeWidth={2} />
                          ))}
                        </Pie>
                        <Tooltip content={<CustomTooltip />} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 10, flex: 1 }}>
                    {statusData.map((s, i) => {
                      const pct = totalDatasets > 0 ? Math.round((s.value / totalDatasets) * 100) : 0;
                      return (
                        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11 }}>
                          <div style={{ width: 10, height: 10, borderRadius: 3, background: s.color, flexShrink: 0 }} />
                          <span style={{ color: 'var(--text-muted)', flex: 1 }}>{s.name}</span>
                          <span style={{ fontFamily: "'DM Mono', monospace", fontWeight: 700, color: s.color }}>{s.value} <span style={{ fontSize: 9, opacity: 0.6 }}>({pct}%)</span></span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Recent Datasets Table ── */}
        <div className="glass-panel" style={{ overflow: 'hidden', marginBottom: 20 }}>
          <div style={{
            padding: '14px 20px', borderBottom: '1px solid var(--border-color)',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          }}>
            <div>
              <div style={{ fontSize: 14, fontWeight: 700, color: '#fff' }}>Recent Datasets</div>
              <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)', marginTop: 2 }}>
                {totalDatasets} total · {readyDatasets.length} cleaned
              </div>
            </div>
            <button className="emp-btn emp-btn-ghost emp-btn-sm" onClick={() => navigate('/employee/datasets')}>
              View All <ArrowRight size={10} />
            </button>
          </div>

          {loading ? (
            <div style={{ padding: '3rem', textAlign: 'center' }}>
              <Loader size={32} color="var(--primary)" style={{ animation: 'spin 1s linear infinite', marginBottom: 12 }} />
              <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>Loading datasets...</p>
            </div>
          ) : datasets.length === 0 ? (
            <div style={{ padding: '3rem', textAlign: 'center' }}>
              <p style={{ color: 'var(--text-muted)', fontSize: 13, marginBottom: 16 }}>No datasets uploaded yet</p>
            </div>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Dataset Name</th>
                    <th style={thStyle}>Status</th>
                    <th style={thStyle}>Uploaded</th>
                    <th style={thStyle}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {datasets.slice(0, 8).map((ds, i) => {
                    const dsId = ds.dataset_id || ds.id;
                    const dsName = ds.name || ds.filename || 'Unnamed';
                    const isReady = ds.status === 'completed' || ds.status === 'ready' || ds.status === 'cleaned';
                    return (
                      <tr key={dsId || i} style={{
                        animation: `adminSlideIn 0.4s cubic-bezier(0.16,1,0.3,1) ${0.1 + i * 0.04}s both`,
                      }}>
                        <td style={tdStyle}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <HardDrive size={14} color="#58a6ff" />
                            <div>
                              <div style={{ fontWeight: 600, color: '#fff', fontSize: 12 }}>{dsName}</div>
                              <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)' }}>
                                ID: {dsId?.slice(0, 8)}...
                              </div>
                            </div>
                          </div>
                        </td>
                        <td style={tdStyle}><StatusBadge status={ds.status} /></td>
                        <td style={{ ...tdStyle, fontFamily: "'DM Mono', monospace", fontSize: 10, color: 'var(--text-muted)' }}>
                          {ds.created_at ? new Date(ds.created_at).toLocaleDateString('en-IN', {
                            day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit'
                          }) : '—'}
                        </td>
                        <td style={tdStyle}>
                          <div style={{ display: 'flex', gap: 4 }}>
                            {isReady ? (
                              <button className="emp-btn emp-btn-ghost emp-btn-sm" style={{ fontSize: 9, padding: '4px 10px' }}
                                onClick={() => navigate(`/employee/visualization?ds=${dsId}&name=${encodeURIComponent(dsName)}`)}>
                                <BarChart3 size={10} /> View
                              </button>
                            ) : (
                              <button className="emp-btn emp-btn-primary emp-btn-sm" style={{ fontSize: 9, padding: '4px 10px' }}
                                onClick={() => navigate(`/employee/cleaning?ds=${dsId}&name=${encodeURIComponent(dsName)}`)}>
                                <Sparkles size={10} /> Clean
                              </button>
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* ── AI Chatbot CTA ── */}
        {readyDatasets.length > 0 && (
          <div className="glass-panel" onClick={() => navigate('/employee/chat')} style={{
            padding: '20px 24px', cursor: 'pointer',
            background: 'linear-gradient(135deg, rgba(88,166,255,0.06) 0%, rgba(63,185,80,0.04) 100%)',
            borderLeft: '3px solid #3fb950',
            transition: 'all 0.2s ease',
          }}
            onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-1px)'}
            onMouseLeave={e => e.currentTarget.style.transform = ''}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                <div style={{
                  width: 42, height: 42, borderRadius: 12,
                  background: 'rgba(63,185,80,0.15)', display: 'flex',
                  alignItems: 'center', justifyContent: 'center',
                }}>
                  <MessageSquare size={20} color="#3fb950" />
                </div>
                <div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: '#fff' }}>Ask AI about your data</div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                    "What is the total revenue by region?" — Get instant answers from {readyDatasets.length} dataset{readyDatasets.length !== 1 ? 's' : ''}
                  </div>
                </div>
              </div>
              <ArrowRight size={18} color="#3fb950" />
            </div>
          </div>
        )}

      </div>

      <style>{`
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
      `}</style>
    </EmployeeLayout>
  );
};

const thStyle = {
  padding: '10px 16px', textAlign: 'left', fontSize: 10,
  fontFamily: "'DM Mono', monospace", textTransform: 'uppercase',
  letterSpacing: 0.5, color: 'var(--text-muted)',
  borderBottom: '1px solid var(--border-color)',
  background: 'rgba(22,27,34,0.5)',
};

const tdStyle = {
  padding: '10px 16px', borderBottom: '1px solid rgba(255,255,255,0.03)',
  fontSize: 12, color: 'var(--text-main)',
};

export default EmployeeDashboardPage;
