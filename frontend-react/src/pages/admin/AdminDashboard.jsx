import { useState, useEffect } from 'react';
import { Navigate, useNavigate } from 'react-router-dom';
import { Users, Database, Zap, AlertCircle, TrendingUp, ArrowUpRight, RefreshCw, Loader, BarChart3, Sparkles } from 'lucide-react';
import AdminLayout from '../../layout/AdminLayout';
import { getUsers, getDatasets, getUserStats, getQueryVolume, getActivityLogs, getDatasetAssignments } from '../../services/api';
import AssignUserModal from '../../components/admin/AssignUserModal';

const API_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api';

function AnimatedNumber({ value, duration = 1200 }) {
  const [display, setDisplay] = useState('0');
  useEffect(() => {
    const numStr = value.toString().replace(/,/g, '');
    const target = parseInt(numStr, 10);
    if (isNaN(target)) { setDisplay(value); return; }
    const start = performance.now();
    const format = (n) => n.toLocaleString();
    const tick = (now) => {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(format(Math.round(target * eased)));
      if (progress < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [value, duration]);
  return <span>{display}</span>;
}

export default function AdminDashboard() {
  const navigate = useNavigate();
  const [hoveredBar, setHoveredBar] = useState(null);
  const [animatedBars, setAnimatedBars] = useState(false);
  const [employees, setEmployees] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [activityLogs, setActivityLogs] = useState([]);
  const [stats, setStats] = useState({ total: 0, active: 0, byRole: { admin: 0, employee: 0 } });
  const [loading, setLoading] = useState(true);
  const [roleFilter, setRoleFilter] = useState('all');
  const [queryChartData, setQueryChartData] = useState([10, 10, 10, 10, 10, 10, 10]);
  const [queryDayLabels, setQueryDayLabels] = useState(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']);
  const [assignModal, setAssignModal] = useState(null);
  const [datasetAssignments, setDatasetAssignments] = useState({});

  const role = sessionStorage.getItem('role');
  if (role !== 'admin') return <Navigate to="/datasets" />;

  const getAuthHeaders = () => {
    const token = sessionStorage.getItem('token');
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  useEffect(() => {
    const t = setTimeout(() => setAnimatedBars(true), 400);
    return () => clearTimeout(t);
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [usersRes, datasetsRes, statsRes, volRes, logsRes] = await Promise.all([
        getUsers(roleFilter),
        getDatasets(),
        getUserStats(),
        getQueryVolume(7).catch(() => ({ success: false })),
        getActivityLogs({ includeSessions: 'true' }).catch(() => ({ success: false }))
      ]);

      setEmployees(usersRes.users || []);
      setDatasets(datasetsRes.data || []);
      setActivityLogs(logsRes?.logs || []);
      setStats(statsRes.stats || { total: 0, active: 0, byRole: { admin: 0, employee: 0 } });
      if (volRes?.success && Array.isArray(volRes.normalized) && volRes.normalized.length > 0) {
        setQueryChartData(volRes.normalized);
        if (Array.isArray(volRes.days) && volRes.days.length === volRes.normalized.length) {
          setQueryDayLabels(volRes.days);
        }
      }
    } catch (err) {
      console.error('Failed to fetch data:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchAssignments = async (dsId) => {
    try {
      const res = await getDatasetAssignments(dsId);
      if (res.success) {
        setDatasetAssignments(prev => ({ ...prev, [dsId]: res.users }));
      }
    } catch (err) {
      console.warn(`Failed to fetch assignments for ${dsId}`);
    }
  };

  useEffect(() => {
    if (datasets.length > 0) {
      datasets.slice(0, 8).forEach(ds => {
        fetchAssignments(ds.dataset_id || ds.id);
      });
    }
  }, [datasets]);

  useEffect(() => {
    fetchData();
  }, [roleFilter]);

  const handleRoleChange = async (email, newRole) => {
    setEmployees(prev => prev.map(emp => emp.email === email ? { ...emp, role: newRole } : emp));
    try {
      const { updateUserRole } = await import('../../services/api');
      await updateUserRole(email, newRole);
      fetchData();
    } catch (err) {
      console.error('Failed to update role:', err);
      fetchData();
    }
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case 'completed':
      case 'ready':
      case 'cleaned':
        return <span className="admin-badge green">● Cleaned</span>;
      case 'processing':
      case 'cleaning':
        return <span className="admin-badge yellow">● Cleaning</span>;
      case 'failed':
        return <span className="admin-badge red">● Failed</span>;
      default:
        return <span className="admin-badge gray">○ Not Cleaned</span>;
    }
  };

  const getDatasetStatus = (ds) => {
    const s = ds.status || ds.upload_status;
    if (s === 'completed' || s === 'ready' || s === 'cleaned') return 'cleaned';
    if (s === 'processing' || s === 'cleaning') return 'cleaning';
    if (s === 'failed') return 'failed';
    return 'not_cleaned';
  };

  const formatSize = (bytes) => {
    if (!bytes) return '0 B';
    const mb = bytes / (1024 * 1024);
    if (mb >= 1) return `${mb.toFixed(1)} MB`;
    const kb = bytes / 1024;
    return `${kb.toFixed(1)} KB`;
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return '—';
    const d = new Date(dateStr);
    return d.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
  };

  const recentDatasets = datasets.slice(0, 8).map(ds => ({
    id: ds.dataset_id || ds.id,
    name: ds.name || ds.filename || 'Unnamed',
    meta: `v${ds.version || 1} · ${ds.rows_count?.toLocaleString() || '—'} rows · ${ds.columns_count || '—'} cols`,
    uploader: ds.uploaded_by?.charAt(0).toUpperCase() || 'U',
    uploaderName: ds.uploaded_by || 'Unknown',
    uploaderColor: '#58a6ff',
    status: getDatasetStatus(ds),
    size: formatSize(ds.file_size || ds.size),
  }));

  const cleanedDatasetsCount = datasets.filter(d => d.status === 'completed' || d.status === 'ready' || d.status === 'cleaned').length;
  const cleaningDatasetsCount = datasets.filter(d => d.status === 'processing' || d.status === 'cleaning').length;
  const notCleanedDatasetsCount = datasets.filter(d => d.status === 'not_cleaned' || !d.status).length;

  return (
    <AdminLayout title="Dashboard" subtitle={`Welcome, ${sessionStorage.getItem('userName') || 'Admin'}`}>
      {/* Stat Cards */}
      <div className="admin-stat-grid">
        <div className="admin-stat-card accent">
          <Users size={22} style={{ marginBottom: 12, color: 'var(--primary)' }} />
          <div className="admin-stat-value admin-count-animate"><AnimatedNumber value={String(stats.total)} /></div>
          <div className="admin-stat-label">Total Users</div>
          <div className="admin-stat-delta admin-delta-up">
            <ArrowUpRight size={12} /> {stats.byRole?.employee || 0} employees
          </div>
        </div>
        <div className="admin-stat-card accent">
          <Database size={22} style={{ marginBottom: 12, color: 'var(--primary)' }} />
          <div className="admin-stat-value admin-count-animate"><AnimatedNumber value={String(datasets.length)} /></div>
          <div className="admin-stat-label">Total Datasets</div>
          <div className="admin-stat-delta admin-delta-up">
            <ArrowUpRight size={12} /> {cleanedDatasetsCount} cleaned
          </div>
        </div>
        <div className="admin-stat-card green">
          <Zap size={22} style={{ marginBottom: 12, color: 'var(--success)' }} />
          <div className="admin-stat-value admin-count-animate"><AnimatedNumber value={String(cleanedDatasetsCount)} /></div>
          <div className="admin-stat-label">Cleaned Assets</div>
          <div className="admin-stat-delta admin-delta-up">
            <TrendingUp size={12} /> {cleaningDatasetsCount} cleaning
          </div>
        </div>
        <div className="admin-stat-card gray">
          <Database size={22} style={{ marginBottom: 12, color: 'var(--text-muted)' }} />
          <div className="admin-stat-value admin-count-animate"><AnimatedNumber value={String(notCleanedDatasetsCount)} /></div>
          <div className="admin-stat-label">Not Cleaned</div>
          <div className="admin-stat-delta">
             Awaiting processing
          </div>
        </div>
      </div>

      {/* Row 1: Employees + Activity */}
      <div className="admin-two-col">
        {/* Employee Table */}
        <div>
          <div className="admin-section-header">
            <div>
              <div className="admin-section-title">Employees</div>
              <div className="admin-section-sub">{stats.total} total · {stats.active || stats.total} active</div>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="admin-btn admin-btn-ghost admin-btn-sm" onClick={() => navigate('/admin/employees')}>
                View All →
              </button>
            </div>
          </div>
          <div className="admin-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>User</th>
                  <th>Role</th>
                  <th>Datasets</th>
                  <th>Status</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  Array.from({ length: 4 }).map((_, i) => (
                    <tr key={i}>
                      <td><div style={{ width: 120, height: 14, background: 'rgba(255,255,255,0.06)', borderRadius: 4 }} /></td>
                      <td><div style={{ width: 80, height: 28, background: 'rgba(255,255,255,0.04)', borderRadius: 6 }} /></td>
                      <td><div style={{ width: 30, height: 14, background: 'rgba(255,255,255,0.04)', borderRadius: 4 }} /></td>
                      <td><div style={{ width: 60, height: 20, background: 'rgba(255,255,255,0.04)', borderRadius: 10 }} /></td>
                      <td></td>
                    </tr>
                  ))
                ) : employees.length === 0 ? (
                  <tr>
                    <td colSpan={5} style={{ textAlign: 'center', padding: '32px', color: 'var(--text-muted)' }}>
                      No users found for this filter
                    </td>
                  </tr>
                ) : (
                  employees.slice(0, 5).map((emp, i) => (
                    <tr key={emp.email} style={{ animation: `adminSlideIn 0.4s cubic-bezier(0.16,1,0.3,1) ${0.1 + i * 0.05}s both` }}>
                      <td>
                        <div className="admin-user-cell">
                          <div className="admin-u-avatar" style={{ background: emp.color || '#58a6ff' }}>{emp.initials || emp.full_name?.charAt(0).toUpperCase() || '??'}</div>
                          <div>
                            <div className="admin-u-name">{emp.full_name || 'Unknown'}</div>
                            <div className="admin-u-email">{emp.email}</div>
                          </div>
                        </div>
                      </td>
                      <td>
                        <span style={{ fontSize: 11, color: '#fff', textTransform: 'capitalize' }}>{emp.role || 'employee'}</span>
                      </td>

                      <td style={{ fontFamily: "'DM Mono', monospace", fontSize: '12px' }}>{emp.datasets_count || 0}</td>
                      <td>
                        <span className={`admin-badge ${emp.is_active ? 'green' : 'amber'}`}>
                          {emp.is_active ? 'Active' : 'Pending'}
                        </span>
                      </td>

                      <td><button className="admin-btn admin-btn-ghost admin-btn-sm">⋯</button></td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Activity Feed */}
        <div>
          <div className="admin-section-header">
            <div>
              <div className="admin-section-title">Recent Activity</div>
              <div className="admin-section-sub">Last 24 hours</div>
            </div>
            <button className="admin-btn admin-btn-ghost admin-btn-sm" onClick={() => navigate('/admin/logs')}>All Logs →</button>
          </div>
          <div className="admin-table-wrap">
            <div className="admin-activity-list">
              {activityLogs.slice(0, 5).map((log, i) => {
                const statusColor = log.status === 'ok' ? '#3fb950' : 
                                    log.status === 'pending' ? '#d29922' : '#f85149';
                return (
                  <div
                    key={log.log_id || i}
                    className="admin-activity-item"
                    style={{ animation: `adminSlideIn 0.4s cubic-bezier(0.16,1,0.3,1) ${0.15 + i * 0.06}s both` }}
                  >
                    <div className="admin-act-dot" style={{ background: statusColor }} />
                    <div className="admin-act-text">
                       {/* Make sure userName is valid string to prevent crashing */}
                      <strong>{String(log.user_name || 'System')}</strong> {log.detail || log.event_description || log.event_type}
                    </div>
                    <div className="admin-act-time">{formatDate(log.created_at)}</div>
                  </div>
                );
              })}
              {activityLogs.length === 0 && (
                <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)' }}>
                  No activity logs yet
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Row 2: Datasets + Storage/Charts */}
      <div className="admin-two-col">
        {/* Recent Datasets - Optimized High Density List matching screenshot */}
        <div>
          <div className="admin-section-header">
            <div>
              <div className="admin-section-title">Recent Datasets</div>
              <div className="admin-section-sub">All company uploads</div>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="admin-btn admin-btn-primary admin-btn-sm" onClick={() => navigate('/admin/upload')}>
                Upload Dataset
              </button>
              <button className="admin-btn admin-btn-ghost admin-btn-sm" onClick={() => navigate('/datasets')}>
                View All →
              </button>
            </div>
          </div>
          <div className="admin-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>DATASET</th>
                  <th>UPLOADED BY</th>
                  <th>STATUS</th>
                  <th>SIZE</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                   Array.from({ length: 4 }).map((_, i) => (
                    <tr key={i}>
                      <td colSpan={4}><div style={{ height: 40, width: '100%', background: 'rgba(255,255,255,0.02)', borderRadius: 4 }} /></td>
                    </tr>
                  ))
                ) : datasets.length === 0 ? (
                  <tr>
                    <td colSpan={4} style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
                      No datasets uploaded yet
                    </td>
                  </tr>
                ) : (
                  recentDatasets.slice(0, 5).map((ds, i) => (
                    <tr key={ds.id} style={{ animation: `adminSlideIn 0.4s cubic-bezier(0.16,1,0.3,1) ${0.2 + i * 0.05}s both` }}>
                      <td>
                        <div style={{ padding: '4px 0' }}>
                          <div style={{ fontSize: '14px', fontWeight: 600, color: '#fff', marginBottom: '2px' }}>{ds.name}</div>
                          <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{ds.meta}</div>
                        </div>
                      </td>
                      <td>
                        <div className="admin-user-cell" style={{ gap: '10px' }}>
                          <div className="admin-u-avatar admin-u-avatar-sm" style={{ background: ds.uploaderColor }}>{ds.uploader}</div>
                          <div className="admin-u-name" style={{ fontSize: '13px' }}>{ds.uploaderName}</div>
                        </div>
                      </td>
                      <td>
                        <div style={{ display: 'inline-flex', alignItems: 'center', padding: '4px 12px', borderRadius: '12px', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)', fontSize: '11px', fontWeight: 500, color: ds.status === 'cleaned' ? '#3fb950' : ds.status === 'cleaning' ? '#d29922' : '#f85149' }}>
                          {ds.status === 'cleaned' ? 'Cleaned' : ds.status === 'cleaning' ? 'Cleaning' : ds.status}
                        </div>
                      </td>
                      <td>
                        <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>{ds.size}</div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Distribution + Chart */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div>
            <div className="admin-section-header">
              <div className="admin-section-title">Dataset Status Distribution</div>
            </div>
            <div className="admin-table-wrap" style={{ padding: '18px 20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 12 }}>
                <div>
                  <span style={{ fontSize: '18px', color: '#3fb950', fontWeight: 600 }}>{cleanedDatasetsCount}</span>
                  <span style={{ fontSize: '12px', color: 'var(--text-muted)', marginLeft: 8 }}>Cleaned</span>
                </div>
                <div>
                  <span style={{ fontSize: '18px', color: '#d29922', fontWeight: 600 }}>{cleaningDatasetsCount}</span>
                  <span style={{ fontSize: '12px', color: 'var(--text-muted)', marginLeft: 8 }}>Cleaning</span>
                </div>
                <div>
                  <span style={{ fontSize: '18px', color: 'var(--text-muted)', fontWeight: 600 }}>{notCleanedDatasetsCount}</span>
                  <span style={{ fontSize: '12px', color: 'var(--text-muted)', marginLeft: 8 }}>Not Cleaned</span>
                </div>
              </div>
              <div className="admin-storage-bar">
                <div className="admin-storage-used" style={{ width: `${datasets.length > 0 ? (cleanedDatasetsCount / datasets.length) * 100 : 0}%` }} />
              </div>
              <div style={{ fontFamily: "'DM Mono', monospace", fontSize: '10px', color: 'var(--text-muted)', marginTop: 6 }}>
                {datasets.length} total datasets
              </div>
            </div>
          </div>

          <div>
            <div className="admin-section-header">
              <div className="admin-section-title">Query volume (7 days)</div>
            </div>
            <div className="admin-table-wrap" style={{ padding: '14px 16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: "'DM Mono', monospace", fontSize: '10px', color: 'var(--text-muted)', marginBottom: 4 }}>
                {queryDayLabels.map(d => <span key={d}>{d}</span>)}
              </div>
              <div className="admin-mini-chart">
                {queryChartData.map((h, i) => (
                  <div
                    key={i}
                    className="admin-chart-bar"
                    style={{
                      height: animatedBars ? `${h}%` : '4%',
                      transitionDelay: `${i * 0.08}s`,
                      opacity: hoveredBar === i ? 1 : 0.7,
                    }}
                    onMouseEnter={() => setHoveredBar(i)}
                    onMouseLeave={() => setHoveredBar(null)}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .admin-u-avatar-sm { width: 24px; height: 24px; font-size: 10px; }
      `}</style>
    </AdminLayout>
  );
}
