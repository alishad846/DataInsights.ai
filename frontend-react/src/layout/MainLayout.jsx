import { useState, useEffect } from 'react';
import { Activity, Database, Upload, ChevronDown } from 'lucide-react';
import { useNavigate, useLocation, Link } from 'react-router-dom';
import { getDatasets } from '../services/api';

const MainLayout = ({ children }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [datasets, setDatasets] = useState([]);
  const [isSwitcherOpen, setIsSwitcherOpen] = useState(false);
  const role = sessionStorage.getItem('role');

  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const res = await getDatasets();
        if (res.success) setDatasets(res.data || []);
      } catch (err) {
        console.error("Layout Switcher Error:", err);
      }
    };
    fetchDatasets();
  }, [location.pathname]); // Re-fetch on navigation to keep sync

  const currentDatasetId = location.pathname.split('/').pop();
  const currentDataset = datasets.find(d => d.dataset_id === currentDatasetId || d._id === currentDatasetId);

  return (
    <div className="layout-wrapper" style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <header className="glass-panel" style={{ 
        margin: '1rem', 
        padding: '0.75rem 2rem', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between',
        borderRadius: '12px',
        position: 'sticky',
        top: '1rem',
        zIndex: 1000
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
          <Link to="/" style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '0.75rem', color: '#fff' }}>
            <Activity color="var(--primary)" size={24} />
            <h2 style={{ fontSize: '1.25rem', margin: 0 }}>
              Data<span className="gradient-text">Insights.ai</span>
            </h2>
          </Link>

          {role !== 'admin' && (
            <nav style={{ display: 'flex', gap: '1.5rem', marginLeft: '1rem' }}>
              <Link to="/datasets" className={location.pathname === '/datasets' ? 'active-nav' : ''} style={{ textDecoration: 'none', color: 'var(--text-muted)', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Database size={16} /> Workspace
              </Link>
            </nav>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
          {role !== 'admin' && datasets.length > 0 && (
            <div style={{ position: 'relative' }}>
              <button 
                onClick={() => setIsSwitcherOpen(!isSwitcherOpen)}
                style={{
                  background: 'rgba(255,255,255,0.05)',
                  border: '1px solid var(--border-color)',
                  color: '#fff',
                  padding: '0.5rem 1rem',
                  borderRadius: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem',
                  cursor: 'pointer',
                  fontSize: '0.85rem'
                }}
              >
                <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: currentDataset ? 'var(--secondary)' : 'transparent', boxShadow: currentDataset ? '0 0 8px var(--secondary)' : 'none' }}></div>
                {currentDataset ? currentDataset.filename : 'Select Workspace'}
                <ChevronDown size={14} color="var(--text-muted)" />
              </button>

              {isSwitcherOpen && (
                <div className="glass-panel" style={{
                  position: 'absolute',
                  top: '110%',
                  right: 0,
                  width: '240px',
                  maxHeight: '300px',
                  overflowY: 'auto',
                  padding: '0.5rem',
                  zIndex: 2000,
                  boxShadow: '0 10px 30px rgba(0,0,0,0.5)'
                }}>
                  {datasets.map(d => (
                    <div 
                      key={d._id}
                      onClick={() => {
                        navigate(`/dashboard/${d._id}`);
                        setIsSwitcherOpen(false);
                      }}
                      style={{
                        padding: '0.75rem 1rem',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        background: d._id === currentDatasetId ? 'rgba(88, 166, 255, 0.1)' : 'transparent',
                        fontSize: '0.85rem',
                        transition: 'all 0.2s',
                        marginBottom: '0.25rem'
                      }}
                      className="hover-bg"
                    >
                      {d.filename}
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{d.status} • {new Date(d.uploadedAt).toLocaleDateString()}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </header>
      
      <main className="container" style={{ flexGrow: 1, padding: '2rem 1rem', width: '100%', maxWidth: '1200px', alignSelf: 'center' }}>
        {children}
      </main>
      
      <footer style={{ 
        textAlign: 'center', 
        padding: '2rem', 
        color: 'var(--text-muted)',
        fontSize: '0.875rem'
      }}>
        <p>© 2026 DataInsights.ai — Autonomous ML Analytics Context</p>
      </footer>
    </div>
  );
};

export default MainLayout;
