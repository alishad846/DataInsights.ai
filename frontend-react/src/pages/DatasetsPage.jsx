import { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

import { Database, Calendar, FileText, ChevronRight, Loader, BarChart3, Trash2, AlertTriangle, Sparkles, Users, Eye, Download } from 'lucide-react';
import { getDatasets, deleteDataset, getDatasetAssignments, downloadDataset } from '../services/api';
import MainLayout from '../layout/MainLayout';
import AdminLayout from '../layout/AdminLayout';
import AssignUserModal from '../components/admin/AssignUserModal';
import DatasetPreviewModal from '../components/DatasetPreviewModal';

const DatasetsPage = () => {
    const [datasets, setDatasets] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [deleteId, setDeleteId] = useState(null);
    const [showConfirm, setShowConfirm] = useState(null);
    const [assignModal, setAssignModal] = useState(null);
    const [previewDataset, setPreviewDataset] = useState(null);
    const [datasetAssignments, setDatasetAssignments] = useState({});
    const [highlightedId, setHighlightedId] = useState(null);
    
    const navigate = useNavigate();
    const location = useLocation();

    const role = sessionStorage.getItem('role');
    const isAdmin = role === 'admin';

    const fetchDatasets = async () => {
        setIsLoading(true);
        try {
            const res = await getDatasets();
            if (res.success) {
                setDatasets(res.data || []);
            }
        } catch (err) {
            console.error("Failed to fetch datasets:", err);
        } finally {
            setIsLoading(false);
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
        fetchDatasets();
    }, []);

    useEffect(() => {
        if (isAdmin && datasets.length > 0) {
            datasets.forEach(ds => {
                fetchAssignments(ds.dataset_id || ds.id);
            });

            // Handle highlighting from Permission resolution
            const params = new URLSearchParams(location.search);
            const highlightId = params.get('highlight');
            if (highlightId) {
                setHighlightedId(highlightId);
                // Scroll into view after a short delay
                setTimeout(() => {
                    const el = document.getElementById(`ds-card-${highlightId}`);
                    if (el) {
                        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                }, 500);
                
                // Remove highlight after 5 seconds
                setTimeout(() => setHighlightedId(null), 5000);
            }
        }
    }, [datasets, isAdmin, location.search]);


    const handleDelete = async (datasetId, e) => {
        e.stopPropagation();
        setDeleteId(datasetId);
        setShowConfirm(datasetId);
    };

    const confirmDelete = async () => {
        if (!deleteId) return;
        
        try {
            const res = await deleteDataset(deleteId);
            if (res.success) {
                setDatasets(datasets.filter(d => (d.dataset_id || d.id) !== deleteId));
            } else {
                alert(res.message || 'Failed to delete dataset');
            }
        } catch (err) {
            console.error("Delete error:", err);
            alert('Failed to delete dataset');
        } finally {
            setDeleteId(null);
            setShowConfirm(null);
        }
    };

    const cancelDelete = () => {
        setDeleteId(null);
        setShowConfirm(null);
    };

    const handleDownload = async (ds, e) => {
        e.stopPropagation();
        try {
            await downloadDataset(ds.dataset_id || ds.id, ds.name || ds.file_name);
        } catch (err) {
            alert('Download failed');
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
                return <span className="admin-badge gray">○ {status || 'Unknown'}</span>;
        }
    };

    const formatSize = (bytes) => {
        if (!bytes) return '0 B';
        const mb = bytes / (1024 * 1024);
        if (mb >= 1) return `${mb.toFixed(1)} MB`;
        const kb = bytes / 1024;
        return `${kb.toFixed(1)} KB`;
    };

    if (isLoading) {
        const Layout = isAdmin ? AdminLayout : MainLayout;
        return (
            <Layout title="Datasets" subtitle="Loading datasets...">
                <div className="flex-center" style={{ minHeight: '60vh', flexDirection: 'column' }}>
                    <Loader className="spinner" size={40} color="var(--primary)" />
                    <p style={{ marginTop: '1rem', color: 'var(--text-muted)' }}>Retrieving your workspace...</p>
                </div>
            </Layout>
        );
    }

    const renderAdminContent = () => (
        <div style={{ padding: '20px 0' }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '20px' }}>
                {datasets.map((ds) => {
                    const dsId = ds.dataset_id || ds.id;
                    const isHighlighted = highlightedId === dsId;
                    return (
                        <div 
                            key={dsId} 
                            id={`ds-card-${dsId}`}
                            className={`glass-panel ${isHighlighted ? 'admin-card-highlight' : ''}`} 
                            style={{ padding: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column', transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)' }}
                        >

                        <div style={{ padding: '20px', flex: 1 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                                    <div style={{ width: 40, height: 40, borderRadius: 10, background: 'rgba(88,166,255,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                        <Database size={20} color="var(--primary)" />
                                    </div>
                                    <div>
                                        <div style={{ fontSize: 16, fontWeight: 700, color: '#fff' }}>{ds.name}</div>
                                        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{formatSize(ds.file_size || ds.size)}</div>
                                    </div>
                                </div>
                                {getStatusBadge(ds.status)}
                            </div>
                            
                            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 20 }}>
                                {ds.rows_count?.toLocaleString() || '—'} rows · {ds.columns_count || '—'} cols · Uploaded {new Date(ds.created_at).toLocaleDateString()}
                            </div>
                            
                            <div style={{ marginBottom: 20 }}>
                                <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
                                    <Users size={12} /> Access Management
                                </div>
                                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                    <div style={{ display: 'flex', marginLeft: 4 }}>
                                        {datasetAssignments[ds.dataset_id || ds.id] && datasetAssignments[ds.dataset_id || ds.id].length > 0 ? (
                                            <div style={{ display: 'flex', alignItems: 'center' }}>
                                                {datasetAssignments[ds.dataset_id || ds.id].slice(0, 4).map((user, idx) => (
                                                    <div 
                                                        key={user.user_id} 
                                                        title={user.full_name}
                                                        style={{ 
                                                            width: 28, height: 28, borderRadius: '50%', 
                                                            background: 'var(--primary)', border: '2px solid var(--bg-dark)',
                                                            marginLeft: idx === 0 ? 0 : -10,
                                                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                            fontSize: 10, fontWeight: 700, color: '#fff',
                                                            zIndex: 5 - idx
                                                        }}
                                                    >
                                                        {user.full_name?.charAt(0)}
                                                    </div>
                                                ))}
                                                {datasetAssignments[ds.dataset_id || ds.id].length > 4 && (
                                                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 10 }}>
                                                        +{datasetAssignments[ds.dataset_id || ds.id].length - 4} more
                                                    </div>
                                                )}
                                            </div>
                                        ) : (
                                            <span style={{ fontSize: 12, color: 'var(--text-muted)', fontStyle: 'italic' }}>No users assigned</span>
                                        )}
                                    </div>
                                    <button 
                                        className="admin-btn admin-btn-ghost admin-btn-sm" 
                                        style={{ fontSize: 11, padding: '6px 12px' }}
                                        onClick={() => setAssignModal(ds)}
                                    >
                                        Assign Users
                                    </button>
                                </div>
                            </div>
                        </div>

                        <div style={{ background: 'rgba(255,255,255,0.02)', padding: '12px 20px', borderTop: '1px solid var(--border-color)', display: 'flex', gap: 10 }}>
                            <button className="admin-btn admin-btn-primary" style={{ flex: 1, height: 36, fontSize: 13 }}
                                onClick={() => navigate(ds.status === 'ready' || ds.status === 'completed' || ds.status === 'cleaned' ? `/employee/visualization?ds=${ds.dataset_id || ds.id}&name=${encodeURIComponent(ds.name)}` : `/employee/cleaning?ds=${ds.dataset_id || ds.id}&name=${encodeURIComponent(ds.name)}`)}>
                                {ds.status === 'ready' || ds.status === 'completed' || ds.status === 'cleaned' ? <><BarChart3 size={14} /> View Data</> : <><Sparkles size={14} /> Clean Data</>}
                            </button>
                            <button className="admin-btn admin-btn-ghost admin-btn-sm" style={{ padding: '0 10px', height: 36 }} onClick={(e) => { e.stopPropagation(); setPreviewDataset(ds); }} title="Quick Preview">
                                <Eye size={16} />
                            </button>
                            <button className="admin-btn admin-btn-ghost admin-btn-sm" style={{ padding: '0 10px', height: 36 }} onClick={(e) => handleDownload(ds, e)} title="Download CSV">
                                <Download size={16} />
                            </button>
                            <button className="admin-btn admin-btn-danger admin-btn-sm" style={{ padding: '0 10px', height: 36 }} onClick={(e) => handleDelete(ds.dataset_id || ds.id, e)}>
                                <Trash2 size={16} />
                            </button>
                        </div>
                    </div>
                );})}
            </div>

        </div>
    );

    const renderEmployeeContent = () => (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {datasets.map((dataset) => (
                <div 
                    key={dataset.dataset_id} 
                    className="glass-panel hover-scale" 
                    style={{ 
                        padding: '1.5rem', 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'space-between',
                        cursor: 'pointer',
                        transition: 'all 0.3s ease'
                    }}
                    onClick={() => navigate(`/dashboard/${dataset.dataset_id}`)}
                >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
                        <div style={{ 
                            background: (dataset.status === 'completed' || dataset.status === 'ready' || dataset.status === 'cleaned') ? 'rgba(63, 185, 80, 0.15)' : 'rgba(210, 153, 34, 0.15)', 
                            padding: '0.75rem', 
                            borderRadius: '10px' 
                        }}>
                            <FileText color={(dataset.status === 'completed' || dataset.status === 'ready' || dataset.status === 'cleaned') ? 'var(--secondary)' : 'var(--warning)'} />
                        </div>
                        <div>
                            <h4 style={{ margin: 0, fontSize: '1.1rem' }}>{dataset.name}</h4>
                            <div style={{ display: 'flex', gap: '1rem', marginTop: '0.4rem', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                                <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                                    <Calendar size={14} /> {new Date(dataset.created_at).toLocaleDateString()}
                                </span>
                                <span>{dataset.rows_count || 0} Rows • {dataset.columns_count || 0} Columns</span>
                                <span style={{ 
                                    textTransform: 'capitalize', 
                                    color: (dataset.status === 'completed' || dataset.status === 'ready' || dataset.status === 'cleaned') ? 'var(--secondary)' : 'var(--warning)',
                                    fontWeight: 600
                                }}>
                                    {dataset.status === 'completed' || dataset.status === 'ready' || dataset.status === 'cleaned' ? 'Cleaned' : dataset.status}
                                </span>
                            </div>
                        </div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <button 
                            className="btn-ghost"
                            onClick={(e) => {
                                e.stopPropagation();
                                setPreviewDataset(dataset);
                            }}
                            style={{ padding: '0.5rem', display: 'flex', alignItems: 'center', color: 'var(--primary)' }}
                            title="Preview Data"
                        >
                            <Eye size={18} />
                        </button>
                        {(dataset.status === 'completed' || dataset.status === 'ready' || dataset.status === 'cleaned') && (
                            <button 
                                className="btn-ghost"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    navigate(`/employee/visualization?ds=${dataset.dataset_id}&name=${encodeURIComponent(dataset.name)}`);
                                }}
                                style={{ padding: '0.5rem 1rem', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
                            >
                                <BarChart3 size={16} /> Visualize
                            </button>
                        )}
                        <button 
                            className="btn-ghost"
                            onClick={(e) => handleDelete(dataset.dataset_id, e)}
                            style={{ padding: '0.5rem', color: 'var(--danger)', display: 'flex', alignItems: 'center' }}
                            title="Delete dataset"
                        >
                            <Trash2 size={18} />
                        </button>
                        <ChevronRight size={20} color="var(--text-muted)" />
                    </div>
                </div>
            ))}
        </div>
    );

    const Layout = isAdmin ? AdminLayout : MainLayout;

    return (
        <Layout title="Datasets" subtitle={isAdmin ? "Manage and assign datasets to employees" : "Manage and analyze your historical data assets"}>
            <div className="view-enter">
                {showConfirm && (
                    <div style={{
                        position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
                        background: 'rgba(0,0,0,0.7)', display: 'flex', alignItems: 'center', justifyContent: 'center',
                        zIndex: 1000
                    }}>
                        <div className="glass-panel" style={{ padding: '2rem', maxWidth: 400 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
                                <AlertTriangle size={24} color="var(--danger)" />
                                <h3 style={{ margin: 0 }}>Delete Dataset?</h3>
                            </div>
                            <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
                                This will permanently delete the dataset and all its files. This action cannot be undone.
                            </p>
                            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'flex-end' }}>
                                <button onClick={cancelDelete} className="btn-ghost" style={{ padding: '0.5rem 1rem' }}>
                                    Cancel
                                </button>
                                <button onClick={confirmDelete} style={{
                                    background: 'var(--danger)', border: 'none', color: '#fff',
                                    padding: '0.5rem 1rem', borderRadius: 6, cursor: 'pointer'
                                }}>
                                    Delete
                                </button>
                            </div>
                        </div>
                    </div>
                )}

                {datasets.length === 0 ? (
                    <div className="glass-panel" style={{ padding: '4rem', textAlign: 'center' }}>
                        <Database size={48} color="var(--text-muted)" style={{ marginBottom: '1.5rem' }} />
                        <h3>No datasets found</h3>
                        <p style={{ color: 'var(--text-muted)', marginBottom: '2rem' }}>Upload your first dataset to start generating insights.</p>
                        {isAdmin && <button className="btn-primary" onClick={() => navigate('/admin/upload')}>Upload Dataset</button>}
                    </div>
                ) : (
                    isAdmin ? renderAdminContent() : renderEmployeeContent()
                )}
            </div>

            {assignModal && (
                <AssignUserModal 
                    dataset={assignModal} 
                    onClose={() => setAssignModal(null)} 
                    onUpdate={() => fetchAssignments(assignModal.dataset_id || assignModal.id)}
                />
            )}

            {previewDataset && (
                <DatasetPreviewModal 
                    dataset={previewDataset} 
                    onClose={() => setPreviewDataset(null)} 
                />
            )}

            <style>{`
                .admin-card-highlight {
                    border: 2px solid var(--secondary) !important;
                    box-shadow: 0 0 20px rgba(63, 185, 80, 0.4) !important;
                    transform: scale(1.02);
                    animation: pulse-border 2s infinite;
                }
                @keyframes pulse-border {
                    0% { border-color: var(--secondary); box-shadow: 0 0 10px rgba(63, 185, 80, 0.3); }
                    50% { border-color: #fff; box-shadow: 0 0 20px rgba(63, 185, 80, 0.6); }
                    100% { border-color: var(--secondary); box-shadow: 0 0 10px rgba(63, 185, 80, 0.3); }
                }
            `}</style>
        </Layout>

    );
};

export default DatasetsPage;
