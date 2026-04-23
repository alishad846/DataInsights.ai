import { useState, useEffect } from 'react';
import { Navigate } from 'react-router-dom';
import { Filter, X, Loader, Database, Users, Clock, Eye, Sparkles, BarChart3, FileText, Upload, Check, Shield } from 'lucide-react';
import AdminLayout from '../../layout/AdminLayout';
import { getActivityLogs, getActivityStats, getDatasetsAdmin } from '../../services/api';

function getMethodClass(event) {
  switch (event) {
    case 'LOGIN': return 'admin-method-get';
    case 'LOGOUT': return 'admin-method-get';
    case 'QUERY': return 'admin-method-get';
    case 'CLEAN_START': return 'admin-method-post';
    case 'CLEAN_DONE': return 'admin-method-post';
    case 'CLEAN_PAUSE': return 'admin-method-blocked';
    case 'CHAT_START': return 'admin-method-post';
    case 'CHAT_END': return 'admin-method-get';
    case 'PERM_REQUEST': return 'admin-method-post';
    case 'PERM_DENIED': return 'admin-method-blocked';
    case 'VISUALIZE': return 'admin-method-post';
    case 'UPLOAD': return 'admin-method-post';
    case 'TRAIN': return 'admin-method-post';
    case 'SESSION': return 'admin-method-get';
    default: return 'admin-method-get';

  }
}

function getEventIcon(event) {
  switch (event) {
    case 'LOGIN': return <Users size={12} />;
    case 'LOGOUT': return <Clock size={12} />;
    case 'QUERY': return <Database size={12} />;
    case 'CLEAN_START': return <Sparkles size={12} />;
    case 'CLEAN_DONE': return <Check size={12} />;
    case 'CLEAN_PAUSE': return <Clock size={12} />;
    case 'CHAT_START': return <FileText size={12} />;
    case 'CHAT_END': return <Clock size={12} />;
    case 'PERM_REQUEST': return <Shield size={12} />;
    case 'PERM_DENIED': return <Shield size={12} color="#f85149" />;
    case 'VISUALIZE': return <BarChart3 size={12} />;
    case 'UPLOAD': return <Upload size={12} />;
    case 'TRAIN': return <BarChart3 size={12} />;
    case 'SESSION': return <Clock size={12} />;
    default: return <FileText size={12} />;

  }
}

function getStatusBadge(status) {
  switch (status) {
    case 'ok': return <span className="admin-badge green">✓ OK</span>;
    case 'pending': return <span className="admin-badge amber">⏳ Pending</span>;
    case 'failed': return <span className="admin-badge red">✗ Failed</span>;
    default: return <span className="admin-badge gray">{status || 'OK'}</span>;
  }
}

function formatDate(dateStr) {
  if (!dateStr) return '—';
  const d = new Date(dateStr);
  return d.toLocaleString('en-IN', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });
}

function formatDuration(seconds) {
  if (!seconds || seconds === 0) return '0s';
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

export default function LogsPage() {
  const [loading, setLoading] = useState(true);
  const [logs, setLogs] = useState([]);
  const [users, setUsers] = useState([]);
  const [events, setEvents] = useState([]);
  const [datasets, setDatasets] = useState([]);
  
  const [employeeFilter, setEmployeeFilter] = useState('All');
  const [eventFilter, setEventFilter] = useState('all');
  const [statusFilter, setStatusFilter] = useState('all');
  const [datasetFilter, setDatasetFilter] = useState('all');

  const role = sessionStorage.getItem('role');
  const isAdmin = role === 'admin';

  const fetchData = async () => {
    setLoading(true);
    try {
      const [logsRes, statsRes] = await Promise.all([
        getActivityLogs({
          employee: employeeFilter !== 'All' ? employeeFilter : undefined,
          event: eventFilter !== 'all' ? eventFilter : undefined,
          status: statusFilter !== 'all' ? statusFilter : undefined,
          dataset: datasetFilter !== 'all' ? datasetFilter : undefined,
          includeSessions: 'true'
        }),
        getActivityStats()
      ]);
      
      if (logsRes.success) {
        setLogs(logsRes.logs || []);
      }
      
      if (statsRes.success) {
        setUsers(statsRes.users || []);
        setEvents(statsRes.events || []);
      }

      const datasetsRes = await getDatasetsAdmin();
      if (datasetsRes.success) {
        setDatasets(datasetsRes.data || []);
      }
    } catch (err) {
      console.error('Failed to fetch logs data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [employeeFilter, eventFilter, statusFilter, datasetFilter]);

  if (!isAdmin) return <Navigate to="/employee/datasets" replace />;

  const filteredLogs = logs;

  const hasFilters = employeeFilter !== 'All' || eventFilter !== 'all' || statusFilter !== 'all' || datasetFilter !== 'all';

  const clearFilters = () => {
    setEmployeeFilter('All');
    setEventFilter('all');
    setStatusFilter('all');
    setDatasetFilter('all');
  };

  const userOptions = ['All', ...users.map(u => u.user_name).filter(Boolean)];
  const eventOptions = ['all', ...events];
  const statusOptions = [
    { value: 'all', label: 'All Status' },
    { value: 'ok', label: 'Completed' },
    { value: 'pending', label: 'Pending' },
    { value: 'failed', label: 'Failed' }
  ];
  const datasetOptions = ['all', ...datasets.map(d => d.dataset_name).filter(Boolean)];

  const totalLogins = logs.filter(l => l.event_type === 'LOGIN').length;
  const totalCleans = logs.filter(l => l.event_type === 'CLEAN_DONE' || l.event_type === 'CLEAN_START' || l.event_type === 'CLEAN').length;
  const totalQueries = logs.filter(l => l.event_type === 'QUERY' || l.event_type === 'CHAT_START').length;
  const totalPermissionIssues = logs.filter(l => l.event_type === 'PERM_DENIED' || l.event_type === 'PERM_REQUEST').length;


  return (
    <AdminLayout title="Activity Logs" subtitle="Employee activity tracking with real-time logs">
      <div className="admin-three-col">
        <div className="admin-stat-card accent" style={{ padding: 16 }}>
          <div className="admin-stat-value" style={{ fontSize: 22 }}>{totalLogins}</div>
          <div className="admin-stat-label">Total Logins</div>
        </div>
        <div className="admin-stat-card green" style={{ padding: 16 }}>
          <div className="admin-stat-value" style={{ fontSize: 22 }}>{totalCleans}</div>
          <div className="admin-stat-label">Cleaning Actions</div>
        </div>
        <div className="admin-stat-card danger" style={{ padding: 16 }}>
          <div className="admin-stat-value" style={{ fontSize: 22 }}>{totalQueries}</div>
          <div className="admin-stat-label">AI Interactions</div>
        </div>
        <div className="admin-stat-card" style={{ padding: 16, background: 'linear-gradient(135deg, rgba(210,153,34,0.2) 0%, rgba(210,153,34,0.1) 100%)' }}>
          <div className="admin-stat-value" style={{ fontSize: 22, color: '#d29922' }}>{totalPermissionIssues}</div>
          <div className="admin-stat-label">Permission Events</div>
        </div>

      </div>

      <div className="admin-filter-bar">
        <Filter size={14} style={{ color: 'var(--text-muted)' }} />
        <select className="admin-filter-select" value={employeeFilter} onChange={e => setEmployeeFilter(e.target.value)}>
          {userOptions.map(e => <option key={e} value={e}>{e}</option>)}
        </select>
        <select className="admin-filter-select" value={eventFilter} onChange={e => setEventFilter(e.target.value)}>
          {eventOptions.map(e => <option key={e} value={e}>{e === 'all' ? 'All Events' : e}</option>)}
        </select>
        <select className="admin-filter-select" value={statusFilter} onChange={e => setStatusFilter(e.target.value)}>
          {statusOptions.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
        </select>
        <select className="admin-filter-select" value={datasetFilter} onChange={e => setDatasetFilter(e.target.value)}>
          {datasetOptions.map(d => <option key={d} value={d}>{d === 'all' ? 'All Datasets' : d}</option>)}
        </select>
        {hasFilters && (
          <button className="admin-btn admin-btn-ghost admin-btn-sm" onClick={clearFilters} style={{ marginLeft: 'auto' }}>
            <X size={12} /> Clear Filters
          </button>
        )}
        <button className="admin-btn admin-btn-ghost admin-btn-sm" style={{ marginLeft: hasFilters ? 8 : 'auto' }} onClick={fetchData}>
          <Loader size={12} style={loading ? { animation: 'spin 1s linear infinite' } : {}} /> Refresh
        </button>
      </div>

      <div className="admin-section-header">
        <div>
          <div className="admin-section-title">All Events</div>
          <div className="admin-section-sub">Showing {filteredLogs.length} events</div>
        </div>
      </div>
      <div className="admin-table-wrap">
        {loading ? (
          <div style={{ padding: '40px', textAlign: 'center' }}>
            <Loader size={24} style={{ animation: 'spin 1s linear infinite', color: 'var(--primary)' }} />
          </div>
        ) : filteredLogs.length === 0 ? (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
            No activity logs found. Employee activities will appear here after login.
          </div>
        ) : (
          <>
            <div className="admin-table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Date & Time</th>
                    <th>Employee</th>
                    <th>Event</th>
                    <th>Dataset</th>
                    <th>Detail</th>
                    <th>Status</th>
                    <th>Duration</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredLogs.map((log, i) => (
                    <tr
                      key={log.log_id || i}
                      style={{ animation: `adminSlideIn 0.35s cubic-bezier(0.16,1,0.3,1) ${i * 0.04}s both` }}
                    >
                      <td style={{ fontFamily: "'DM Mono', monospace", fontSize: 11, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                        {formatDate(log.created_at)}
                      </td>
                      <td>
                        <div className="admin-user-cell">
                          <div className="admin-u-avatar" style={{ background: '#58a6ff', width: 22, height: 22, fontSize: 9 }}>
                            {log.user_name ? log.user_name.charAt(0).toUpperCase() : 'U'}
                          </div>
                          <div>
                            <span style={{ fontSize: 12 }}>{log.user_name || 'Unknown'}</span>
                            <div style={{ fontSize: 9, color: 'var(--text-muted)' }}>{log.user_email}</div>
                          </div>
                        </div>
                      </td>
                      <td>
                        <span className={`admin-log-method ${getMethodClass(log.event_type)}`}>
                          {getEventIcon(log.event_type)}
                          {log.event_type}
                        </span>
                      </td>
                      <td style={{ fontSize: 12 }}>{log.dataset_name || '—'}</td>
                      <td style={{
                        maxWidth: 220,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                        color: 'var(--text-main)',
                        fontSize: 11,
                        fontFamily: "'DM Mono', monospace"
                      }}>
                        {log.detail || log.event_description || '—'}
                      </td>
                        <td>{getStatusBadge(log.status)}</td>
                      <td style={{ fontFamily: "'DM Mono', monospace", fontSize: 11, color: log.event_type === 'SESSION' ? '#3fb950' : 'var(--text-muted)' }}>
                        {log.event_type === 'SESSION' ? formatDuration(log.duration_seconds) : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {filteredLogs.length > 0 && (
              <div className="admin-pagination">
                <div className="admin-page-info">Showing 1–{filteredLogs.length} events</div>
              </div>
            )}
          </>
        )}
      </div>

      <style>{`
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .admin-log-method {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 10px;
          font-weight: 600;
          font-family: "'DM Mono', monospace";
        }
        .admin-method-get {
          background: rgba(88, 166, 255, 0.15);
          color: #58a6ff;
        }
        .admin-method-post {
          background: rgba(63, 185, 80, 0.15);
          color: #3fb950;
        }
        .admin-method-blocked {
          background: rgba(248, 81, 73, 0.15);
          color: #f85149;
        }
      `}</style>
    </AdminLayout>
  );
}
