import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, FileText, Eye, ChevronDown, ChevronUp, RefreshCw, BarChart3, Trash2, AlertTriangle, X, Sparkles, Plus, Database, Loader } from 'lucide-react';
import { getDatasets, deleteDataset, getDatasetPreview, getAvailableDatasetsToRequest, requestPermission } from '../../services/api';

import EmployeeLayout from '../../layout/EmployeeLayout';


const MOCK_DATASETS = [
  {
    id: 'ds-001', name: 'Customer_Data', type: 'xlsx', status: 'cleaned',
    source: 'acme-prod', path: '/data/crm/customers.xlsx',
    rows: 12450, cols: 24, size: '8.1 MB', version: 'v3',
    updated: '2 days ago',
    uploadedBy: 'John Admin',
    versions: [
      { tag: 'v3', desc: 'Cleaned · 12,450 rows', date: 'Jan 18 2025', current: true },
      { tag: 'v2', desc: 'Cleaned · 12,800 rows', date: 'Dec 4 2024' },
      { tag: 'v1', desc: 'Raw · 13,100 rows', date: 'Nov 20 2024' },
    ]
  },
  {
    id: 'ds-002', name: 'Q3_Sales_Report', type: 'csv', status: 'cleaning',
    source: 'acme-prod', path: '/data/sales/q3_2024.csv',
    rows: 4521, cols: 12, size: '2.4 MB', version: 'v1',
    updated: '12 min ago', cleaningProgress: 40, cleaningStep: '2/5 — Removing duplicates',
    uploadedBy: 'Sarah Manager',
    versions: [
      { tag: 'v1', desc: 'Cleaning in progress…', date: 'Jan 20 2025', active: true },
    ]
  },
  {
    id: 'ds-003', name: 'Finance_Q2_2024', type: 'xlsx', status: 'not_cleaned',
    source: 'acme-prod', path: '/data/finance/q2_2024.xlsx',
    rows: 3200, cols: 9, size: '1.8 MB', version: 'v1',
    updated: '5 days ago', versions: [],
    uploadedBy: 'John Admin',
  },
];

const typeIcons = { csv: '📄', xlsx: '📊', json: '🗂' };
const typeColors = {
  csv: { bg: 'rgba(63,185,80,0.1)', color: 'var(--success)' },
  xlsx: { bg: 'rgba(88,166,255,0.1)', color: 'var(--primary)' },
  json: { bg: 'rgba(210,153,34,0.1)', color: 'var(--warning)' },
};

const EmployeeDatasetsPage = () => {
  const navigate = useNavigate();
  const [datasets, setDatasets] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [typeFilter, setTypeFilter] = useState('all');
  const [statusFilter, setStatusFilter] = useState('all');
  const [previewModal, setPreviewModal] = useState(null);
  const [deleteConfirm, setDeleteConfirm] = useState(null);
  const [isDeleting, setIsDeleting] = useState(null);
  const [showRequestModal, setShowRequestModal] = useState(false);
  const [availableDatasets, setAvailableDatasets] = useState([]);
  const [isLoadingAvailable, setIsLoadingAvailable] = useState(false);
  const [requestSearchQuery, setRequestSearchQuery] = useState('');
  const [requestStatus, setRequestStatus] = useState({}); // {dsId: 'sending' | 'sent' | 'error'}



  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await getDatasets();
        if (res.success && res.data?.length > 0) {
          const mapped = res.data
            .filter((d) => d.dataset_id || d._id)
            .map((d, i) => ({
            id: d.dataset_id || d._id,
            dataset_id: d.dataset_id || d._id,
            name: d.name || d.dataset_name || d.filename?.replace(/\.\w+$/, '') || `Dataset ${i + 1}`,
            type: d.filename?.split('.').pop() || d.name?.split('.').pop() || 'csv',
            status: (() => {
              const s = d.status || d.upload_status || 'not_cleaned';
              if (s === 'ready' || s === 'completed' || s === 'cleaned') return 'cleaned';
              if (s === 'processing' || s === 'cleaning') return 'cleaning';
              if (s === 'failed') return 'failed';
              return 'not_cleaned';
            })(),
            source: 'server',
            path: d.filename,
            rows: d.rows_count || d.rows,
            cols: d.columns_count || d.columns,
            size: d.fileSize ? `${(d.fileSize / 1024 / 1024).toFixed(1)} MB` : '—',
            version: 'v1',
            updated: d.created_at ? new Date(d.created_at).toLocaleDateString() : 'recent',
            versions: [],
            uploadedBy: d.uploaded_by_name || d.uploaded_by_email || 'Admin',
            uploadedByEmail: d.uploaded_by_email,
            has_access: d.has_access,
          }));
          setDatasets(mapped);
        } else {
          setDatasets([]);
        }
      } catch (err) {
        console.warn('Failed to fetch datasets:', err.message);
        setDatasets([]);
      } finally {
        setIsLoading(false);
      }
    };
    fetchData();
  }, []);

  const fetchAvailable = async () => {
    setIsLoadingAvailable(true);
    try {
      const res = await getAvailableDatasetsToRequest();
      if (res.success) {
        setAvailableDatasets(res.data || []);
      }
    } catch (err) {
      console.error("Failed to fetch available datasets:", err);
    } finally {
      setIsLoadingAvailable(false);
    }
  };

  const handleRequestAccess = async (dsId) => {
    setRequestStatus(prev => ({ ...prev, [dsId]: 'sending' }));
    try {
      const res = await requestPermission(dsId, 'DATASET_ACCESS');
      if (res.success) {
        setRequestStatus(prev => ({ ...prev, [dsId]: 'sent' }));
        setTimeout(() => {
          setAvailableDatasets(prev => prev.filter(d => (d.dataset_id || d.id) !== dsId));
        }, 1500);
      } else {
        setRequestStatus(prev => ({ ...prev, [dsId]: 'error' }));
      }
    } catch (err) {
      setRequestStatus(prev => ({ ...prev, [dsId]: 'error' }));
    }
  };


  const openPreview = async (ds) => {
    const dsId = ds.dataset_id || ds.id;
    setPreviewModal({
      name: ds.name,
      type: ds.type,
      version: ds.version,
      rows: ds.rows,
      cols: ds.cols,
      size: ds.size,
      _datasetId: dsId,
      loading: true,
      data: [],
    });

    try {
      const res = await getDatasetPreview(dsId);
      if (res.success) {
        setPreviewModal(prev => ({
          ...prev,
          loading: false,
          data: res.data || [],
          headers: res.data && res.data.length > 0 ? Object.keys(res.data[0]) : [],
          rows: res.total_rows_previewed || (res.data ? res.data.length : 0),
          error: null,
        }));
      } else {
        setPreviewModal(prev => ({ 
          ...prev, 
          loading: false, 
          data: [], 
          error: res.message || 'Failed to load data' 
        }));
      }
    } catch (err) {
      console.warn('Preview error:', err.message);
      setPreviewModal(prev => ({ 
        ...prev, 
        loading: false, 
        data: [], 
        error: err.response?.data?.message || 'Unable to load preview. Please try again.' 
      }));
    }
  };

  const handleDelete = async (ds, e) => {
    e.stopPropagation();
    const dsId = ds.dataset_id || ds.id;
    setDeleteConfirm({ id: dsId, name: ds.name });
  };

  const confirmDelete = async () => {
    if (!deleteConfirm) return;
    setIsDeleting(deleteConfirm.id);
    try {
      const res = await deleteDataset(deleteConfirm.id);
      if (res.success) {
        setDatasets(datasets.filter(d => (d.dataset_id || d.id) !== deleteConfirm.id));
      } else {
        alert(res.message || 'Failed to delete dataset');
      }
    } catch (err) {
      console.error('Delete error:', err);
      alert('Failed to delete dataset');
    } finally {
      setIsDeleting(null);
      setDeleteConfirm(null);
    }
  };

  const cancelDelete = () => {
    setDeleteConfirm(null);
    setIsDeleting(null);
  };

  const filtered = datasets.filter(ds => {
    const matchesSearch = ds.name.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesType = typeFilter === 'all' || ds.type === typeFilter;
    const matchesStatus = statusFilter === 'all' || ds.status === statusFilter;
    return matchesSearch && matchesType && matchesStatus;
  });

  const StatusBadge = ({ status }) => {
    const config = {
      cleaned: { bg: 'rgba(63,185,80,0.1)', color: '#3fb950', label: '● Cleaned' },
      completed: { bg: 'rgba(63,185,80,0.1)', color: '#3fb950', label: '● Cleaned' },
      ready: { bg: 'rgba(63,185,80,0.1)', color: '#3fb950', label: '● Cleaned' },
      cleaning: { bg: 'rgba(210,153,34,0.1)', color: '#d29922', label: '⟳ Cleaning' },
      processing: { bg: 'rgba(210,153,34,0.1)', color: '#d29922', label: '⟳ Cleaning' },
      not_cleaned: { bg: 'rgba(139,148,158,0.1)', color: 'rgba(139, 148, 158, 0.8)', label: '○ Not Cleaned' },
    }[status] || { bg: 'rgba(139,148,158,0.1)', color: 'rgba(139, 148, 158, 0.8)', label: status };
    return (
      <span className="emp-status-badge" style={{ background: config.bg, color: config.color }}>
        {config.label}
      </span>
    );
  };

  return (
    <EmployeeLayout>
      <div className="emp-topbar">
        <div>
          <div className="emp-topbar-title">Company Datasets</div>
          <div className="emp-topbar-sub">Manage and explore your data assets</div>
        </div>
        <div className="emp-topbar-actions">
          <div className="emp-search-bar">
            <Search size={14} />
            <input
              type="text"
              placeholder="Search datasets…"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
            />
          </div>
          <button className="emp-btn emp-btn-primary" onClick={() => { setShowRequestModal(true); fetchAvailable(); }}>
            <Plus size={14} /> Request New Dataset
          </button>

        </div>
      </div>

      <div className="emp-content">
        {datasets.length > 0 && (
          <div className="glass-panel" style={{ padding: '12px 16px', marginBottom: 16, display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#3fb950' }} />
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              {datasets.length} dataset{datasets.length !== 1 ? 's' : ''} available
            </span>
          </div>
        )}

        <div className="emp-filters">
          <select className="emp-filter-select" value={typeFilter} onChange={e => setTypeFilter(e.target.value)}>
            <option value="all">All Types</option>
            <option value="csv">CSV</option>
            <option value="xlsx">Excel</option>
            <option value="json">JSON</option>
          </select>
          <select className="emp-filter-select" value={statusFilter} onChange={e => setStatusFilter(e.target.value)}>
            <option value="all">All Statuses</option>
            <option value="cleaned">Cleaned</option>
            <option value="cleaning">Cleaning</option>
            <option value="not_cleaned">Not Cleaned</option>
          </select>
          <div className="emp-filter-divider" />
          <div className="emp-filter-count">
            {filtered.length} dataset{filtered.length !== 1 ? 's' : ''}
          </div>
        </div>

        <div className="emp-dataset-grid">
          {isLoading ? (
            Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="emp-skeleton" style={{ animationDelay: `${i * 0.05}s` }}>
                <div className="emp-skeleton-title" />
                <div className="emp-skeleton-sub" />
                <div className="emp-skeleton-chips">
                  <div className="emp-skeleton-chip" />
                  <div className="emp-skeleton-chip" />
                </div>
              </div>
            ))
          ) : filtered.length === 0 ? (
            <div className="emp-empty">
              <FileText size={48} color="var(--text-muted)" />
              <div className="emp-empty-title">No datasets found</div>
              <div className="emp-empty-sub">Try adjusting your filters</div>
            </div>
          ) : (
            filtered.map((ds) => (
              <div key={ds.id} className="glass-panel" style={{ padding: 0, overflow: 'hidden' }}>
                <div style={{ padding: '16px 20px' }}>
                  <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 12 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                      <div style={{ 
                        width: 40, height: 40, borderRadius: 10, 
                        background: typeColors[ds.type]?.bg || 'rgba(139,148,158,0.1)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '1.25rem'
                      }}>
                        {typeIcons[ds.type] || '📄'}
                      </div>
                      <div>
                        <div style={{ fontSize: 14, fontWeight: 600, color: '#fff' }}>{ds.name}</div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                          {ds.type.toUpperCase()} · {ds.version}
                          {ds.uploadedBy && (
                            <span style={{ marginLeft: 8, color: 'var(--primary)' }}>
                              • Uploaded by <strong>{ds.uploadedBy}</strong>
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                    <StatusBadge status={ds.status} />
                  </div>

                  <div style={{ display: 'flex', gap: 20, marginBottom: 8 }}>
                    {ds.rows && (
                      <div>
                        <div style={{ fontSize: 16, fontWeight: 600, color: '#fff' }}>{ds.rows?.toLocaleString()}</div>
                        <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Rows</div>
                      </div>
                    )}
                    {ds.cols && (
                      <div>
                        <div style={{ fontSize: 16, fontWeight: 600, color: '#fff' }}>{ds.cols}</div>
                        <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Columns</div>
                      </div>
                    )}
                    {ds.size && ds.size !== '—' && (
                      <div>
                        <div style={{ fontSize: 16, fontWeight: 600, color: '#fff' }}>{ds.size}</div>
                        <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Size</div>
                      </div>
                    )}
                  </div>

                  <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: "'DM Mono', monospace" }}>
                    Updated {ds.updated}
                  </div>
                </div>

                <div style={{ borderTop: '1px solid var(--border-color)', padding: '10px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'rgba(13,17,23,0.5)' }}>
                  <div style={{ display: 'flex', gap: 6 }}>
                    <button className="emp-btn emp-btn-ghost emp-btn-sm"
                      onClick={(e) => { e.stopPropagation(); openPreview(ds); }}>
                      <Eye size={12} /> Preview
                    </button>
                    <button className="emp-btn emp-btn-ghost emp-btn-sm"
                      onClick={(e) => { 
                        e.stopPropagation(); 
                        const dsId = ds.dataset_id || ds.id;
                        navigate(`/employee/visualization?ds=${dsId}&name=${encodeURIComponent(ds.name)}`);
                      }}>
                      <BarChart3 size={12} /> Visualize
                    </button>
                    <button className="emp-btn emp-btn-primary emp-btn-sm"
                      onClick={(e) => { 
                        e.stopPropagation(); 
                        const dsId = ds.dataset_id || ds.id;
                        navigate(`/employee/cleaning?ds=${dsId}&name=${encodeURIComponent(ds.name)}`);
                      }}>
                      <Sparkles size={12} /> Clean
                    </button>
                  </div>
                    <button 
                      className="emp-btn emp-btn-ghost emp-btn-sm"
                      onClick={(e) => { e.stopPropagation(); handleDelete(ds, e); }}
                      style={{ color: 'var(--danger)' }}
                      disabled={isDeleting === (ds.dataset_id || ds.id)}
                    >
                      {isDeleting === (ds.dataset_id || ds.id) ? (
                        <RefreshCw size={12} className="spin" />
                      ) : (
                        <Trash2 size={12} />
                      )}
                    </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {previewModal && (
        <div className="emp-modal-overlay" onClick={() => setPreviewModal(null)}>
          <div className="glass-panel emp-modal" onClick={e => e.stopPropagation()}>
            <div className="emp-modal-header">
              <div style={{ flex: 1 }}>
                <div className="emp-modal-title">{previewModal.name}</div>
                <div className="emp-modal-subtitle">
                  First 50 rows • Read-only • Total: {previewModal.rows?.toLocaleString() || '—'} rows
                </div>
              </div>
              <button className="emp-btn emp-btn-ghost emp-btn-sm" onClick={() => setPreviewModal(null)}>
                <X size={14} />
              </button>
            </div>

            <div className="emp-modal-body" style={{ overflowX: 'auto', padding: '0 1.25rem' }}>
              {previewModal.loading ? (
                <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                  <div className="spin" style={{ marginBottom: '1rem' }}>
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" strokeWidth="2">
                      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                    </svg>
                  </div>
                  Loading preview...
                </div>
              ) : previewModal.error ? (
                <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--danger)' }}>
                  <div style={{ marginBottom: '0.5rem', fontSize: '1.5rem' }}>⚠️</div>
                  <div style={{ fontWeight: 500 }}>Unable to load preview</div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>{previewModal.error}</div>
                  <button 
                    className="emp-btn emp-btn-ghost emp-btn-sm" 
                    style={{ marginTop: '1rem' }}
                    onClick={() => {
                      const retryDs = { 
                        dataset_id: previewModal._datasetId, 
                        name: previewModal.name 
                      };
                      setPreviewModal({
                        name: previewModal.name,
                        type: previewModal.type,
                        version: previewModal.version,
                        rows: previewModal.rows,
                        cols: previewModal.cols,
                        size: previewModal.size,
                        loading: true,
                        data: [],
                      });
                      openPreview(retryDs);
                    }}
                  >
                    Try Again
                  </button>
                </div>
              ) : previewModal.data.length > 0 ? (
                <div style={{ minWidth: '100%', overflowX: 'auto' }}>
                  <table className="emp-modal-table">
                    <thead>
                      <tr>
                        <th style={{ width: '40px' }}>#</th>
                        {(previewModal.headers || Object.keys(previewModal.data[0])).map(col => (
                          <th key={col} style={{ whiteSpace: 'nowrap' }}>{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {previewModal.data.map((row, i) => (
                        <tr key={i}>
                          <td style={{ color: 'var(--text-muted)', fontSize: '11px' }}>{i + 1}</td>
                          {(previewModal.headers || Object.keys(previewModal.data[0])).map((col, ci) => (
                            <td key={ci} style={{ fontFamily: "'DM Mono', monospace", whiteSpace: 'nowrap', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                              {row[col] === null || row[col] === undefined || row[col] === '' ? <span style={{ color: 'var(--text-muted)', opacity: 0.5 }}>null</span> : String(row[col])}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                  No preview data available. The dataset may still be processing.
                </div>
              )}
            </div>

            <div className="emp-modal-footer" style={{ padding: '1rem 1.5rem' }}>
              <div className="emp-modal-footer-info" style={{ paddingLeft: '0.5rem' }}>
                {previewModal.loading ? 'Loading...' : `Showing first ${previewModal.data.length} rows`}
              </div>
              <div className="emp-modal-footer-actions">
                <button className="emp-btn emp-btn-ghost emp-btn-sm" onClick={() => setPreviewModal(null)}>Close</button>
                {/* Cleaning button hidden as requested */}
              </div>
            </div>
          </div>
        </div>
      )}

      {deleteConfirm && (
        <div className="emp-modal-overlay" onClick={cancelDelete}>
          <div className="glass-panel emp-modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 400 }}>
            <div className="emp-modal-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <AlertTriangle size={20} color="var(--danger)" />
                <div className="emp-modal-title">Delete Dataset?</div>
              </div>
            </div>
            <div style={{ padding: '1.25rem' }}>
              <p style={{ color: 'var(--text-muted)', marginBottom: '1rem' }}>
                This will permanently delete <strong style={{ color: '#fff' }}>{deleteConfirm.name}</strong> and all its files. This action cannot be undone.
              </p>
            </div>
            <div className="emp-modal-footer">
              <div className="emp-modal-footer-actions" style={{ justifyContent: 'flex-end', gap: '0.5rem' }}>
                <button className="emp-btn emp-btn-ghost emp-btn-sm" onClick={cancelDelete}>Cancel</button>
                <button 
                  className="emp-btn emp-btn-sm" 
                  onClick={confirmDelete}
                  style={{ background: 'var(--danger)', border: 'none', color: '#fff' }}
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {showRequestModal && (
        <div className="emp-modal-overlay" onClick={() => setShowRequestModal(false)}>
          <div className="glass-panel emp-modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 600 }}>
            <div className="emp-modal-header">
              <div style={{ flex: 1 }}>
                <div className="emp-modal-title">Request Dataset Access</div>
                <div className="emp-modal-subtitle">Browse all company datasets and request permission to view them</div>
              </div>
              <button className="emp-btn emp-btn-ghost emp-btn-sm" onClick={() => setShowRequestModal(false)}>
                <X size={14} />
              </button>
            </div>

            <div className="emp-modal-body" style={{ maxHeight: '60vh', overflowY: 'auto', padding: '1rem' }}>
              <div className="emp-search-bar" style={{ marginBottom: '1rem', width: '100%' }}>
                <Search size={14} />
                <input
                  type="text"
                  placeholder="Search and discover datasets…"
                  value={requestSearchQuery}
                  onChange={e => setRequestSearchQuery(e.target.value)}
                  style={{ width: '100%' }}
                />
              </div>

              {isLoadingAvailable ? (
                <div style={{ padding: '3rem', textAlign: 'center' }}>
                  <Loader className="spin" size={24} color="var(--primary)" />
                  <p style={{ marginTop: '1rem', color: 'var(--text-muted)' }}>Fetching available datasets...</p>
                </div>
              ) : availableDatasets.filter(ds => ds.name?.toLowerCase().includes(requestSearchQuery.toLowerCase())).length === 0 ? (
                <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                  <Database size={40} style={{ opacity: 0.3, marginBottom: '1rem' }} />
                  <p>{requestSearchQuery ? 'No datasets match your search.' : 'No new datasets available to request right now.'}</p>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  {availableDatasets.filter(ds => ds.name?.toLowerCase().includes(requestSearchQuery.toLowerCase())).map((ds) => {
                    const dsId = ds.dataset_id || ds.id;
                    const status = requestStatus[dsId];
                    return (

                      <div key={dsId} className="glass-panel" style={{ padding: '12px 16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'rgba(255,255,255,0.02)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                          <Database size={18} color="var(--primary)" />
                          <div>
                            <div style={{ fontSize: '13px', fontWeight: 600, color: '#fff' }}>{ds.name}</div>
                            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                              {ds.uploaded_by_name ? `Uploaded by ${ds.uploaded_by_name}` : 'Company Dataset'}
                            </div>
                          </div>
                        </div>
                        <button 
                          className={`emp-btn emp-btn-sm ${status === 'sent' ? 'emp-btn-success' : 'emp-btn-primary'}`}
                          style={{ minWidth: status === 'sending' ? '90px' : 'auto' }}
                          disabled={status === 'sending' || status === 'sent'}
                          onClick={() => handleRequestAccess(dsId)}
                        >
                          {status === 'sending' ? <Loader size={12} className="spin" /> : 
                           status === 'sent' ? 'Request Sent' : 
                           status === 'error' ? 'Retry' : 'Request Access'}
                        </button>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            <div className="emp-modal-footer">
              <div className="emp-modal-footer-actions" style={{ justifyContent: 'flex-end' }}>
                <button className="emp-btn emp-btn-ghost emp-btn-sm" onClick={() => setShowRequestModal(false)}>Close</button>
              </div>
            </div>
          </div>
        </div>
      )}

    </EmployeeLayout>
  );
};

export default EmployeeDatasetsPage;
