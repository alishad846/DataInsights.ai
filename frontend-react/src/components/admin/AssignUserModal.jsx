import React, { useState, useEffect } from 'react';
import { X, Search, UserPlus, UserMinus, Loader, Check, AlertCircle } from 'lucide-react';
import { getUsers, assignDataset, unassignDataset, getDatasetAssignments } from '../../services/api';

const AssignUserModal = ({ dataset, onClose, onUpdate }) => {
  const [employees, setEmployees] = useState([]);
  const [assignedUsers, setAssignedUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [processing, setProcessing] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      const dsId = dataset.dataset_id || dataset.id;
      if (!dsId) {
        setError('Invalid Dataset ID');
        setLoading(false);
        return;
      }

      try {
        const [allUsersRes, assignedRes] = await Promise.all([
          getUsers('employee'),
          getDatasetAssignments(dsId)
        ]);
        
        setEmployees(allUsersRes.users || []);
        setAssignedUsers(assignedRes.users || []);
      } catch (err) {
        console.error('Failed to fetch users:', err);
        setError('Failed to load user data');
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [dataset.dataset_id, dataset.id]);

  const handleAssign = async (userId) => {
    setProcessing(userId);
    const dsId = dataset.dataset_id || dataset.id;
    try {
      const res = await assignDataset(dsId, [userId]);
      if (res.success) {
        const user = employees.find(u => u.user_id === userId);
        setAssignedUsers(prev => [...prev, user]);
        if (onUpdate) onUpdate();
      }
    } catch (err) {
      console.error('Assign error:', err);
    } finally {
      setProcessing(null);
    }
  };

  const handleUnassign = async (userId) => {
    setProcessing(userId);
    const dsId = dataset.dataset_id || dataset.id;
    try {
      const res = await unassignDataset(dsId, userId);
      if (res.success) {
        setAssignedUsers(prev => prev.filter(u => u.user_id !== userId));
        if (onUpdate) onUpdate();
      }
    } catch (err) {
      console.error('Unassign error:', err);
    } finally {
      setProcessing(null);
    }
  };

  const filteredEmployees = employees.filter(emp => 
    (emp.full_name?.toLowerCase().includes(searchQuery.toLowerCase()) || 
     emp.email?.toLowerCase().includes(searchQuery.toLowerCase())) &&
    !assignedUsers.some(au => au.user_id === emp.user_id)
  );

  return (
    <div className="emp-modal-overlay" style={{ zIndex: 2000 }}>
      <div className="glass-panel emp-modal" style={{ maxWidth: 500, maxHeight: '85vh', display: 'flex', flexDirection: 'column', background: 'var(--bg-dark)', border: '1px solid var(--border-color)' }}>
        <div className="emp-modal-header" style={{ padding: '20px 24px', borderBottom: '1px solid var(--border-color)' }}>
          <div style={{ flex: 1 }}>
            <div className="emp-modal-title" style={{ fontSize: 18, color: '#fff' }}>Manage Access: {dataset.name}</div>
            <div className="emp-modal-subtitle" style={{ fontSize: 13, color: 'var(--text-muted)' }}>Assign employees to this dataset</div>
          </div>
          <button className="emp-btn emp-btn-ghost emp-btn-sm" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        <div style={{ padding: '20px 24px 16px' }}>
          <div className="emp-search-bar" style={{ width: '100%', marginBottom: 12, background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border-color)', borderRadius: 8, padding: '8px 12px', display: 'flex', alignItems: 'center', gap: 10 }}>
            <Search size={14} color="var(--text-muted)" />
            <input 
              type="text" 
              placeholder="Search employees by name or email..." 
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{ flex: 1, background: 'transparent', border: 'none', color: '#fff', fontSize: 14, outline: 'none' }}
            />
          </div>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '0 24px 24px' }}>
          {loading ? (
            <div style={{ textAlign: 'center', padding: 40 }}>
              <div className="spin" style={{ marginBottom: 12 }}>
                <Loader size={24} color="var(--primary)" />
              </div>
              <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>Loading employees...</p>
            </div>
          ) : error ? (
            <div style={{ textAlign: 'center', padding: 40, color: 'var(--danger)' }}>
              <AlertCircle size={24} style={{ marginBottom: 12 }} />
              <p>{error}</p>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
              {/* Assigned Section */}
              {assignedUsers.length > 0 && (
                <div>
                  <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 12, letterSpacing: '0.05em' }}>
                    Assigned Employees ({assignedUsers.length})
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {assignedUsers.map(user => (
                      <div key={user.user_id} className="admin-user-cell" style={{ padding: '10px 14px', background: 'rgba(255,255,255,0.03)', borderRadius: 10, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                          <div className="admin-u-avatar" style={{ width: 32, height: 32, fontSize: 12, background: 'var(--primary)', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 600 }}>
                            {user.full_name?.charAt(0) || user.email?.charAt(0)}
                          </div>
                          <div>
                            <div style={{ fontSize: 13, fontWeight: 500, color: '#fff' }}>{user.full_name}</div>
                            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{user.email}</div>
                          </div>
                        </div>
                        <button 
                          className="emp-btn emp-btn-ghost emp-btn-sm" 
                          style={{ color: 'var(--danger)', padding: 6 }}
                          onClick={() => handleUnassign(user.user_id)}
                          disabled={processing === user.user_id}
                          title="Unassign User"
                        >
                          {processing === user.user_id ? <Loader size={14} className="spin" /> : <UserMinus size={14} />}
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Available Section */}
              <div>
                <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 12, letterSpacing: '0.05em' }}>
                  Available Employees ({filteredEmployees.length})
                </div>
                {filteredEmployees.length === 0 ? (
                  <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)', fontSize: 13, border: '1px dashed var(--border-color)', borderRadius: 10 }}>
                    No more employees found
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {filteredEmployees.map(user => (
                      <div key={user.user_id} className="admin-user-cell" style={{ padding: '10px 14px', background: 'rgba(255,255,255,0.03)', borderRadius: 10, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                          <div className="admin-u-avatar" style={{ width: 32, height: 32, fontSize: 12, background: 'rgba(255,255,255,0.1)', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 600 }}>
                            {user.full_name?.charAt(0) || user.email?.charAt(0)}
                          </div>
                          <div>
                            <div style={{ fontSize: 13, fontWeight: 500, color: '#fff' }}>{user.full_name}</div>
                            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{user.email}</div>
                          </div>
                        </div>
                        <button 
                          className="emp-btn emp-btn-ghost emp-btn-sm" 
                          style={{ color: 'var(--primary)', padding: 6 }}
                          onClick={() => handleAssign(user.user_id)}
                          disabled={processing === user.user_id}
                          title="Assign User"
                        >
                          {processing === user.user_id ? <Loader size={14} className="spin" /> : <UserPlus size={14} />}
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="emp-modal-footer" style={{ padding: '16px 24px', borderTop: '1px solid var(--border-color)', background: 'rgba(0,0,0,0.2)' }}>
          <button className="emp-btn emp-btn-primary" style={{ width: '100%', padding: '10px' }} onClick={onClose}>
            Done
          </button>
        </div>
      </div>
      <style>{`
        .spin { animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
};

export default AssignUserModal;
