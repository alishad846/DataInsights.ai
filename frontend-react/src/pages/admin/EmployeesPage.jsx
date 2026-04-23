import { useEffect, useMemo, useState } from 'react';
import { Search, RefreshCw, Loader, Plus, UserMinus, X } from 'lucide-react';
import AdminLayout from '../../layout/AdminLayout';
import { createUser, getUsers, updateUserRole, updateUserStatus } from '../../services/api';

const normalizeRole = (role) => {
  return role === 'admin' ? 'admin' : 'employee';
};

const INITIAL_CREATE_FORM = {
  full_name: '',
  email: '',
  password: '',
  role: 'employee',
  department: '',
  designation: '',
  is_active: true,
};

const modalBackdropStyle = {
  position: 'fixed',
  inset: 0,
  zIndex: 50,
  background: 'rgba(4, 9, 18, 0.72)',
  backdropFilter: 'blur(10px)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  padding: 20,
};

const modalCardStyle = {
  width: '100%',
  background: 'linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(10, 16, 28, 0.98))',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 20,
  boxShadow: '0 30px 90px rgba(0,0,0,0.45)',
  padding: 20,
  color: '#fff',
};

const modalHeadStyle = {
  display: 'flex',
  alignItems: 'flex-start',
  justifyContent: 'space-between',
  gap: 12,
  marginBottom: 16,
};

const formFieldStyle = {
  display: 'grid',
  gap: 8,
};

const fieldLabelStyle = {
  fontSize: 12,
  color: 'var(--text-muted)',
  fontWeight: 600,
};

const fieldControlStyle = {
  width: '100%',
  padding: '11px 12px',
  borderRadius: 10,
  border: '1px solid rgba(255,255,255,0.1)',
  background: 'rgba(255,255,255,0.03)',
  color: '#fff',
  outline: 'none',
};

export default function EmployeesPage() {
  const [employees, setEmployees] = useState([]);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [roleFilter, setRoleFilter] = useState('all');
  const [createOpen, setCreateOpen] = useState(false);
  const [removeOpen, setRemoveOpen] = useState(false);
  const [createError, setCreateError] = useState('');
  const [removeError, setRemoveError] = useState('');
  const [createForm, setCreateForm] = useState(INITIAL_CREATE_FORM);
  const [removeEmail, setRemoveEmail] = useState('');

  useEffect(() => {
    const handler = setTimeout(() => setDebouncedSearch(searchQuery), 300);
    return () => clearTimeout(handler);
  }, [searchQuery]);

  const fetchData = async () => {
    setLoading(true);
    try {
      const res = await getUsers(roleFilter);
      const normalized = (res.users || []).map((user) => ({
        ...user,
        role: normalizeRole(user.role),
      }));
      setEmployees(normalized);
    } catch (err) {
      console.error('Failed to fetch users:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [roleFilter]);

  const handleRoleChange = async (email, newRole) => {
    const normalizedRole = normalizeRole(newRole);
    setEmployees((prev) => prev.map((emp) => (emp.email === email ? { ...emp, role: normalizedRole } : emp)));
    try {
      await updateUserRole(email, normalizedRole);
      fetchData();
    } catch (err) {
      console.error('Failed to update role:', err);
      fetchData();
    }
  };

  const handleCreateEmployee = async (e) => {
    e.preventDefault();
    setActionLoading(true);
    setCreateError('');
    try {
      await createUser({
        full_name: createForm.full_name,
        email: createForm.email,
        password: createForm.password,
        role: normalizeRole(createForm.role),
        department: createForm.department,
        designation: createForm.designation,
        is_active: createForm.is_active,
      });
      setCreateOpen(false);
      setCreateForm(INITIAL_CREATE_FORM);
      fetchData();
    } catch (err) {
      setCreateError(err.response?.data?.message || err.message || 'Failed to create employee');
    } finally {
      setActionLoading(false);
    }
  };

  const openRemoveModal = (email = '') => {
    setRemoveEmail(email);
    setRemoveError('');
    setRemoveOpen(true);
  };

  const handleRemoveEmployee = async (e) => {
    e.preventDefault();
    if (!removeEmail) {
      setRemoveError('Please select an employee to remove.');
      return;
    }

    const target = employees.find((emp) => emp.email === removeEmail);
    if (!target) {
      setRemoveError('Selected employee could not be found.');
      return;
    }

    const confirmed = window.confirm(
      `Deactivate ${target.full_name || target.email}? This will prevent the account from logging in.`
    );
    if (!confirmed) return;

    setActionLoading(true);
    setRemoveError('');
    try {
      await updateUserStatus(removeEmail, 'inactive');
      setRemoveOpen(false);
      setRemoveEmail('');
      fetchData();
    } catch (err) {
      setRemoveError(err.response?.data?.message || err.message || 'Failed to remove employee');
    } finally {
      setActionLoading(false);
    }
  };

  const filteredEmployees = useMemo(() => {
    if (!debouncedSearch.trim()) return employees;
    const lowerQ = debouncedSearch.toLowerCase();
    return employees.filter((emp) =>
      (emp.full_name && emp.full_name.toLowerCase().includes(lowerQ)) ||
      (emp.email && emp.email.toLowerCase().includes(lowerQ)) ||
      (emp.department && emp.department.toLowerCase().includes(lowerQ)) ||
      (emp.designation && emp.designation.toLowerCase().includes(lowerQ))
    );
  }, [employees, debouncedSearch]);

  const getStatusBadge = (status) => {
    if (status) return <span className="admin-badge green">● Active</span>;
    return <span className="admin-badge red">● Inactive</span>;
  };

  const roleOptions = [
    { label: 'employee', value: 'employee' },
    { label: 'admin', value: 'admin' },
  ];

  return (
    <AdminLayout title="Employees" subtitle="Manage company users and access">
      <div className="admin-section-header" style={{ marginBottom: '20px', gap: 10, flexWrap: 'wrap' }}>
        <div className="admin-search-bar" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.1)', flex: 1, minWidth: 280, maxWidth: '420px' }}>
          <Search size={14} />
          <input
            type="text"
            placeholder="Search by name, email, department, or designation..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <button className="admin-btn admin-btn-ghost admin-btn-sm" onClick={fetchData} disabled={loading}>
            <RefreshCw size={12} /> Refresh
          </button>
          <button className="admin-btn admin-btn-sm" onClick={() => setCreateOpen(true)} style={{ background: 'rgba(59, 130, 246, 0.16)', border: '1px solid rgba(59, 130, 246, 0.35)' }}>
            <Plus size={12} /> New Employee
          </button>
          <button className="admin-btn admin-btn-ghost admin-btn-sm" onClick={() => openRemoveModal()} style={{ background: 'rgba(248, 113, 113, 0.08)', border: '1px solid rgba(248, 113, 113, 0.25)' }}>
            <UserMinus size={12} /> Remove Employee
          </button>
          <select className="admin-filter-select" value={roleFilter} onChange={(e) => setRoleFilter(e.target.value)}>
            <option value="all">All Roles</option>
            <option value="employee">Employee</option>
            <option value="admin">Admin</option>
          </select>
        </div>
      </div>

      <div className="admin-table-wrap">
        <table>
          <thead>
            <tr>
              <th>User</th>
              <th>Role</th>
              <th>Department</th>
              <th>Designation</th>
              <th>Datasets</th>
              <th>Status</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              Array.from({ length: 6 }).map((_, i) => (
                <tr key={i}>
                  <td><div style={{ width: 150, height: 14, background: 'rgba(255,255,255,0.06)', borderRadius: 4 }} /></td>
                  <td><div style={{ width: 80, height: 28, background: 'rgba(255,255,255,0.04)', borderRadius: 6 }} /></td>
                  <td><div style={{ width: 100, height: 14, background: 'rgba(255,255,255,0.04)', borderRadius: 4 }} /></td>
                  <td><div style={{ width: 100, height: 14, background: 'rgba(255,255,255,0.04)', borderRadius: 4 }} /></td>
                  <td><div style={{ width: 30, height: 14, background: 'rgba(255,255,255,0.04)', borderRadius: 4 }} /></td>
                  <td><div style={{ width: 60, height: 20, background: 'rgba(255,255,255,0.04)', borderRadius: 10 }} /></td>
                  <td></td>
                </tr>
              ))
            ) : filteredEmployees.length === 0 ? (
              <tr>
                <td colSpan={7} style={{ textAlign: 'center', padding: '64px', color: 'var(--text-muted)' }}>
                  No users found
                </td>
              </tr>
            ) : (
              filteredEmployees.map((emp, i) => {
                const role = normalizeRole(emp.role);
                return (
                  <tr key={emp.email} style={{ animation: `adminSlideIn 0.4s cubic-bezier(0.16,1,0.3,1) ${0.1 + i * 0.05}s both` }}>
                    <td>
                      <div className="admin-user-cell">
                        <div className="admin-u-avatar" style={{ background: emp.color || '#58a6ff' }}>
                          {emp.initials || emp.full_name?.charAt(0).toUpperCase() || 'U'}
                        </div>
                        <div>
                          <div className="admin-u-name">{emp.full_name || 'Unknown'}</div>
                          <div className="admin-u-email">{emp.email}</div>
                        </div>
                      </div>
                    </td>
                    <td>
                      <select
                        className="admin-role-select"
                        value={role}
                        onChange={(e) => handleRoleChange(emp.email, e.target.value)}
                      >
                        {roleOptions.map((option) => (
                          <option key={option.value} value={option.value}>{option.label}</option>
                        ))}
                      </select>
                    </td>
                    <td style={{ color: 'var(--text-muted)' }}>{emp.department || '—'}</td>
                    <td style={{ color: 'var(--text-muted)' }}>{emp.designation || '—'}</td>
                    <td style={{ fontFamily: "'DM Mono', monospace", fontSize: '12px' }}>{emp.datasets_count || emp.datasets || 0}</td>
                    <td>{getStatusBadge(emp.is_active)}</td>
                    <td>
                      <button
                        className="admin-btn admin-btn-ghost admin-btn-sm"
                        title="Remove employee"
                        onClick={() => openRemoveModal(emp.email)}
                      >
                        ⋯
                      </button>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {createOpen && (
        <div style={modalBackdropStyle} onClick={() => !actionLoading && setCreateOpen(false)}>
          <div style={{ ...modalCardStyle, maxWidth: 720 }} onClick={(e) => e.stopPropagation()}>
            <div style={modalHeadStyle}>
              <div>
                <div style={{ fontSize: 18, fontWeight: 700 }}>New Employee</div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>Fill in the account fields that map to the users table.</div>
              </div>
              <button className="admin-btn admin-btn-ghost admin-btn-sm" onClick={() => setCreateOpen(false)}><X size={14} /></button>
            </div>

            {createError && <div style={{ marginBottom: 12, color: '#f87171', fontSize: 13 }}>{createError}</div>}

            <form onSubmit={handleCreateEmployee} className="admin-form-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <label style={formFieldStyle}>
                <span style={fieldLabelStyle}>Full Name</span>
                <input
                  style={fieldControlStyle}
                  value={createForm.full_name}
                  onChange={(e) => setCreateForm((prev) => ({ ...prev, full_name: e.target.value }))}
                  placeholder="Employee name"
                  required
                />
              </label>

              <label style={formFieldStyle}>
                <span style={fieldLabelStyle}>Email</span>
                <input
                  style={fieldControlStyle}
                  type="email"
                  value={createForm.email}
                  onChange={(e) => setCreateForm((prev) => ({ ...prev, email: e.target.value }))}
                  placeholder="employee@example.com"
                  required
                />
              </label>

              <label style={formFieldStyle}>
                <span style={fieldLabelStyle}>Password</span>
                <input
                  style={fieldControlStyle}
                  type="password"
                  value={createForm.password}
                  onChange={(e) => setCreateForm((prev) => ({ ...prev, password: e.target.value }))}
                  placeholder="Temporary password"
                  required
                />
              </label>

              <label style={formFieldStyle}>
                <span style={fieldLabelStyle}>Role</span>
                <select
                  style={fieldControlStyle}
                  value={createForm.role}
                  onChange={(e) => setCreateForm((prev) => ({ ...prev, role: e.target.value }))}
                >
                  {roleOptions.map((option) => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
              </label>

              <label style={formFieldStyle}>
                <span style={fieldLabelStyle}>Department</span>
                <input
                  style={fieldControlStyle}
                  value={createForm.department}
                  onChange={(e) => setCreateForm((prev) => ({ ...prev, department: e.target.value }))}
                  placeholder="Data Science"
                />
              </label>

              <label style={formFieldStyle}>
                <span style={fieldLabelStyle}>Designation</span>
                <input
                  style={fieldControlStyle}
                  value={createForm.designation}
                  onChange={(e) => setCreateForm((prev) => ({ ...prev, designation: e.target.value }))}
                  placeholder="Analyst"
                />
              </label>

              <label style={{ ...formFieldStyle, gridColumn: '1 / -1' }}>
                <span style={fieldLabelStyle}>Status</span>
                <select
                  style={fieldControlStyle}
                  value={createForm.is_active ? 'active' : 'inactive'}
                  onChange={(e) => setCreateForm((prev) => ({ ...prev, is_active: e.target.value === 'active' }))}
                >
                  <option value="active">Active</option>
                  <option value="inactive">Inactive</option>
                </select>
              </label>

              <div style={{ gridColumn: '1 / -1', display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 8 }}>
                <button type="button" className="admin-btn admin-btn-ghost" onClick={() => setCreateOpen(false)} disabled={actionLoading}>
                  Cancel
                </button>
                <button type="submit" className="admin-btn" disabled={actionLoading} style={{ background: 'linear-gradient(135deg, #3b82f6, #2563eb)' }}>
                  {actionLoading ? <Loader size={14} className="spin" /> : 'Create Employee'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {removeOpen && (
        <div style={modalBackdropStyle} onClick={() => !actionLoading && setRemoveOpen(false)}>
          <div style={{ ...modalCardStyle, maxWidth: 560 }} onClick={(e) => e.stopPropagation()}>
            <div style={modalHeadStyle}>
              <div>
                <div style={{ fontSize: 18, fontWeight: 700 }}>Remove Employee</div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>This will set the selected account to inactive.</div>
              </div>
              <button className="admin-btn admin-btn-ghost admin-btn-sm" onClick={() => setRemoveOpen(false)}><X size={14} /></button>
            </div>

            {removeError && <div style={{ marginBottom: 12, color: '#f87171', fontSize: 13 }}>{removeError}</div>}

            <form onSubmit={handleRemoveEmployee} style={{ display: 'grid', gap: 12 }}>
              <label style={formFieldStyle}>
                <span style={fieldLabelStyle}>Select employee</span>
                <select style={fieldControlStyle} value={removeEmail} onChange={(e) => setRemoveEmail(e.target.value)} required>
                  <option value="">Choose employee</option>
                  {employees.map((emp) => (
                    <option key={emp.email} value={emp.email}>
                      {emp.full_name || emp.email} ({emp.email})
                    </option>
                  ))}
                </select>
              </label>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
                <button type="button" className="admin-btn admin-btn-ghost" onClick={() => setRemoveOpen(false)} disabled={actionLoading}>
                  Cancel
                </button>
                <button type="submit" className="admin-btn" disabled={actionLoading} style={{ background: 'linear-gradient(135deg, #ef4444, #b91c1c)' }}>
                  {actionLoading ? <Loader size={14} className="spin" /> : 'Deactivate Employee'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </AdminLayout>
  );
}
