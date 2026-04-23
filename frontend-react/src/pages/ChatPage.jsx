import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, LayoutDashboard, Database } from 'lucide-react';
import QueryAssistant from '../components/QueryAssistant';

const ChatPage = () => {
    const { datasetId } = useParams();
    const navigate = useNavigate();

    return (
        <div className="view-enter" style={{ height: 'calc(100vh - 160px)', display: 'flex', flexDirection: 'column', gap: '1rem' }}>

            {/* Header */}
            <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexShrink: 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <button
                        onClick={() => navigate(`/dashboard/${datasetId}`)}
                        style={{
                            background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border-color)',
                            color: '#fff', padding: '0.5rem 0.75rem', borderRadius: '8px',
                            cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.4rem',
                            fontSize: '0.85rem', transition: 'background 0.2s',
                        }}
                    >
                        <ArrowLeft size={16} />
                        Back
                    </button>

                    <div>
                        <h2 style={{ margin: 0, fontSize: '1.3rem' }}>AI Chatbot</h2>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginTop: '2px' }}>
                            <Database size={12} color="var(--text-muted)" />
                            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                                {datasetId}
                            </span>
                        </div>
                    </div>
                </div>

                <button
                    className="btn-primary"
                    onClick={() => navigate(`/dashboard/${datasetId}`)}
                    style={{
                        display: 'flex', alignItems: 'center', gap: '0.5rem',
                        background: 'transparent', border: '1px solid var(--border-color)',
                        boxShadow: 'none', padding: '0.5rem 1rem', fontSize: '0.85rem',
                    }}
                >
                    <LayoutDashboard size={16} />
                    Dashboard
                </button>
            </header>

            {/* Chat fills remaining space */}
            <div style={{ flex: 1, minHeight: 0 }}>
                <QueryAssistant datasetId={datasetId} />
            </div>
        </div>
    );
};

export default ChatPage;
