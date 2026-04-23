import { AlertTriangle, Info, AlertCircle, Zap } from 'lucide-react';

const InsightsPanel = ({ summary, insights }) => {
  if (!summary && (!insights || insights.length === 0)) return null;

  const getSeverityColor = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'critical': return 'var(--danger)';
      case 'warning': return 'var(--warning)';
      case 'info': return 'var(--primary)';
      case 'positive': return 'var(--success)';
      default: return 'var(--text-main)';
    }
  };

  const getSeverityIcon = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'critical': return <AlertCircle size={20} color="var(--danger)" />;
      case 'warning': return <AlertTriangle size={20} color="var(--warning)" />;
      case 'positive': return <Zap size={20} color="var(--success)" />;
      default: return <Info size={20} color="var(--primary)" />;
    }
  };

  return (
    <div className="glass-panel" style={{ padding: '2rem', marginBottom: '2rem' }}>
      <h3 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <Zap color="var(--accent)" /> AI Executive Summary
      </h3>
      
      {summary && (
        <p style={{ 
          fontSize: '1.1rem', 
          lineHeight: '1.6', 
          color: 'var(--text-main)',
          paddingBottom: '1.5rem',
          borderBottom: '1px solid var(--border-color)',
          marginBottom: '1.5rem'
        }}>
          {summary}
        </p>
      )}

      {insights && insights.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {insights.map((insight, idx) => (
            <div key={idx} style={{
              display: 'flex',
              alignItems: 'flex-start',
              gap: '1rem',
              padding: '1rem',
              backgroundColor: 'rgba(0,0,0,0.2)',
              borderRadius: '8px',
              borderLeft: `4px solid ${getSeverityColor(insight.severity)}`
            }}>
              <div style={{ marginTop: '0.1rem' }}>
                {getSeverityIcon(insight.severity)}
              </div>
              <div>
                <strong style={{ display: 'block', color: getSeverityColor(insight.severity), textTransform: 'capitalize', marginBottom: '0.25rem', fontSize: '0.85rem' }}>
                  {insight.severity || 'Insight'}
                </strong>
                <span style={{ color: 'var(--text-muted)' }}>{insight.description}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default InsightsPanel;
