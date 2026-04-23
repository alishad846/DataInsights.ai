import { useState, useEffect } from 'react';
import { X, Download, Table, Loader, AlertCircle } from 'lucide-react';
import { getDatasetPreview, downloadDataset } from '../services/api';

const DatasetPreviewModal = ({ dataset, onClose }) => {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState([]);
  const [error, setError] = useState(null);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    const fetchPreview = async () => {
      setLoading(true);
      try {
        const res = await getDatasetPreview(dataset.dataset_id);
        if (res.success) {
          setData(res.data);
        } else {
          setError(res.message || 'Failed to load preview');
        }
      } catch (err) {
        setError(err.response?.data?.message || 'Error fetching dataset preview');
      } finally {
        setLoading(false);
      }
    };

    if (dataset?.dataset_id) {
      fetchPreview();
    }
  }, [dataset]);

  const handleDownload = async () => {
    setDownloading(true);
    try {
      await downloadDataset(dataset.dataset_id, dataset.name || dataset.file_name);
    } catch (err) {
      console.error('Download failed:', err);
    } finally {
      setDownloading(false);
    }
  };

  const columns = data.length > 0 ? Object.keys(data[0]) : [];

  return (
    <div className="modal-overlay flex-center" style={{ 
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, 
      backgroundColor: 'rgba(0,0,0,0.8)', zIndex: 1000,
      backdropFilter: 'blur(4px)'
    }}>
      <div className="glass-panel" style={{ 
        width: '95%', maxWidth: '1200px', maxHeight: '90vh', 
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
        padding: 0
      }}>
        {/* Header */}
        <div style={{ 
          padding: '1.5rem 2rem', borderBottom: '1px solid var(--border-color)',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          backgroundColor: 'rgba(255,255,255,0.03)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <Table size={24} color="var(--primary)" />
            <div>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 600, margin: 0 }}>Dataset Preview</h2>
              <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', margin: 0 }}>
                First 50 rows • Read-only • Total: {dataset.rows_count || 'unknown'} rows
              </p>
            </div>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <button 
              onClick={handleDownload}
              disabled={downloading}
              className="btn-secondary"
              style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '0.6rem 1rem', fontSize: '13px' }}
            >
              {downloading ? <Loader size={16} className="spin" /> : <Download size={16} />}
              {downloading ? 'Downloading...' : 'Download Full CSV'}
            </button>
            <button onClick={onClose} style={{ 
              background: 'none', border: 'none', color: 'var(--text-muted)', 
              cursor: 'pointer', padding: '8px', borderRadius: '50%',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: 'background 0.2s'
            }} onMouseOver={e => e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)'} 
               onMouseOut={e => e.currentTarget.style.backgroundColor = 'transparent'}>
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Content */}
        <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          {loading ? (
            <div className="flex-center" style={{ height: '400px', flexDirection: 'column', gap: '16px' }}>
              <Loader size={48} className="spin" color="var(--primary)" />
              <p style={{ color: 'var(--text-muted)' }}>Fetching dataset preview...</p>
            </div>
          ) : error ? (
            <div className="flex-center" style={{ height: '400px', flexDirection: 'column', gap: '16px', textAlign: 'center' }}>
              <AlertCircle size={48} color="var(--danger)" />
              <h3 style={{ color: 'var(--danger)', margin: 0 }}>Access Denied or Error</h3>
              <p style={{ color: 'var(--text-muted)', maxWidth: '400px', fontSize: '0.9rem' }}>{error}</p>
            </div>
          ) : data.length === 0 ? (
            <div className="flex-center" style={{ height: '400px', flexDirection: 'column', gap: '16px' }}>
              <p style={{ color: 'var(--text-muted)' }}>No data found in this dataset.</p>
            </div>
          ) : (
            <div className="table-container" style={{ 
              flex: 1, 
              overflow: 'auto', 
              padding: '0', 
              borderTop: '1px solid var(--border-color)',
              margin: '0',
            }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                <thead style={{ position: 'sticky', top: 0, zIndex: 1, backgroundColor: '#161b22' }}>
                  <tr>
                    <th style={{ padding: '12px 16px', textAlign: 'left', fontWeight: 600, borderBottom: '1px solid var(--border-color)', color: 'var(--text-muted)', width: '40px' }}>#</th>
                    {columns.map(col => (
                      <th key={col} style={{ 
                        padding: '12px 16px', textAlign: 'left', fontWeight: 600,
                        borderBottom: '1px solid var(--border-color)', color: 'var(--primary)',
                        whiteSpace: 'nowrap'
                      }}>
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', backgroundColor: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.01)' }}>
                      <td style={{ padding: '10px 16px', color: 'rgba(255,255,255,0.2)', fontSize: '11px' }}>{i + 1}</td>
                      {columns.map(col => (
                        <td key={col} style={{ padding: '10px 16px', color: 'var(--text-main)', whiteSpace: 'nowrap', maxWidth: '300px', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {row[col] === null || row[col] === undefined ? <span style={{ opacity: 0.3 }}>—</span> : String(row[col])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div style={{ 
          padding: '1rem 2rem', 
          borderTop: '1px solid var(--border-color)', 
          textAlign: 'left', 
          backgroundColor: 'rgba(255,255,255,0.02)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: 0, paddingLeft: '4px' }}>
            Showing {data.length} of {dataset.rows_count || 'unknown'} total rows
          </p>
          <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.15)', textTransform: 'uppercase', letterSpacing: '1px' }}>
            Read-only Mode
          </div>
        </div>
      </div>
      
      <style>{`
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .spin { animation: spin 1s linear infinite; }
        .table-container::-webkit-scrollbar { width: 8px; height: 8px; }
        .table-container::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }
      `}</style>
    </div>
  );
};

export default DatasetPreviewModal;
