import { useState, useEffect } from "react";
import { useNavigate, Link, useLocation } from "react-router-dom";
import api from "../services/api";
import "../styles/Auth.css";

const UserIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
        <circle cx="12" cy="7" r="4" />
    </svg>
);

const MailIcon = () => (
    <svg className="field-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="2" y="4" width="20" height="16" rx="2" />
        <path d="M22 7l-10 6L2 7" />
    </svg>
);

const LockIcon = () => (
    <svg className="field-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
        <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </svg>
);

const EyeIcon = ({ off: isOff }) => isOff ? (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" />
        <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" />
        <line x1="1" y1="1" x2="23" y2="23" />
    </svg>
) : (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
        <circle cx="12" cy="12" r="3" />
    </svg>
);

const AlertIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="12" />
        <line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
);

const ChartIcon = () => (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="18" y1="20" x2="18" y2="10" />
        <line x1="12" y1="20" x2="12" y2="4" />
        <line x1="6" y1="20" x2="6" y2="14" />
    </svg>
);

const CheckCircleIcon = () => (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
        <polyline points="22 4 12 14.01 9 11.01" />
    </svg>
);

export default function Auth() {
    const nav = useNavigate();
    const location = useLocation();
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [showPassword, setShowPassword] = useState(false);
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);
    const [infoMessage, setInfoMessage] = useState("");
    const [attemptsLeft, setAttemptsLeft] = useState(null);

    useEffect(() => {
        if (location.state?.pendingApproval) {
            setInfoMessage(location.state.message || "Your account requires admin approval.");
            window.history.replaceState({}, document.title);
        }
    }, [location.state]);

    const login = async () => {
        if (!email || !password) {
            setError("Please fill in all fields");
            return;
        }
        setError("");
        setLoading(true);
        try {
            const res = await api.post("/auth/login", {
                email,
                password
            });

            if (res.data.pending) {
                setError("Your account is pending approval. Please contact your administrator to activate your account.");
                setLoading(false);
                return;
            }

            const role = res.data.role;
            const name = res.data.name || email.split('@')[0];

            sessionStorage.setItem("token", res.data.token);
            sessionStorage.setItem("role", role);
            sessionStorage.setItem("userName", name);
            sessionStorage.setItem("userEmail", res.data.email);

            setLoading(false);

            nav(role === "admin" ? "/admin" : "/employee/datasets", { replace: true });
        } catch (err) {
            const msg = err.response?.data?.message || "Invalid credentials. Please try again.";
            const left = err.response?.data?.attemptsLeft;
            
            if (left !== undefined) {
                setAttemptsLeft(left);
            }

            if (err.response?.data?.pending) {
                setError("Your account is pending approval. Please contact your administrator to activate your account.");
            } else {
                setError(msg);
            }
            setLoading(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === "Enter") login();
    };

    return (
        <div className="auth-wrapper">
            <div className="orb-1" />
            <div className="orb-2" />
            <div className="orb-3" />

            <div className="auth-card">
                <div className="auth-brand">
                    <div className="brand-icon">
                        <ChartIcon />
                    </div>
                    <h1>Data Insights</h1>
                    <p className="auth-subtitle">Sign in to your account</p>
                    <div className="auth-role-badge user">
                        <UserIcon />
                        Access
                    </div>
                </div>

                <div className="auth-form" onKeyDown={handleKeyDown}>
                    {error && (
                        <div className="auth-error">
                            <AlertIcon />
                            {error}
                        </div>
                    )}

                    {infoMessage && (
                        <div className="auth-error" style={{ background: 'rgba(88, 166, 255, 0.15)', borderColor: 'var(--primary)', color: 'var(--primary)' }}>
                            <AlertIcon />
                            {infoMessage}
                        </div>
                    )}

                    <div className="auth-field">
                        <MailIcon />
                        <input
                            type="email"
                            placeholder="you@example.com"
                            value={email}
                            onChange={e => setEmail(e.target.value)}
                            autoComplete="email"
                        />
                    </div>

                    <div className="auth-field">
                        <LockIcon />
                        <input
                            type={showPassword ? "text" : "password"}
                            placeholder="Password"
                            value={password}
                            onChange={e => setPassword(e.target.value)}
                            autoComplete="current-password"
                        />
                        <button
                            type="button"
                            className="password-toggle"
                            onClick={() => setShowPassword(!showPassword)}
                        >
                            <EyeIcon off={showPassword} />
                        </button>
                    </div>

                    {attemptsLeft !== null && (
                        <div className={`auth-attempt-flag ${attemptsLeft === 1 ? 'critical' : ''}`}>
                            <div className="flag-content">
                                <span className="flag-label">ATTEMPTS LEFT</span>
                                <div className="flag-dots">
                                    {[1, 2, 3].map(i => (
                                        <div key={i} className={`flag-dot ${i <= attemptsLeft ? 'active' : ''}`} />
                                    ))}
                                </div>
                                <span className="flag-count">{attemptsLeft}</span>
                            </div>
                        </div>
                    )}

                    <button
                        className="auth-btn primary"
                        onClick={login}
                        disabled={loading}
                    >
                        {loading ? (
                            <span className="btn-spinner" />
                        ) : (
                            "Sign In"
                        )}
                    </button>
                </div>

                <div className="auth-footer">
                    <Link to="/forgot-password">Forgot Password?</Link>
                </div>
            </div>
        </div>
    );
}
