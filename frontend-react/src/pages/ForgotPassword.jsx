import { useState } from "react";
import { Link } from "react-router-dom";
import api from "../services/api";
import "../styles/Auth.css";

const MailIcon = () => (
    <svg className="field-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="2" y="4" width="20" height="16" rx="2" />
        <path d="M22 7l-10 6L2 7" />
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

const AlertIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="12" />
        <line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
);

export default function ForgotPassword() {
    const [email, setEmail] = useState("");
    const [loading, setLoading] = useState(false);
    const [submitted, setSubmitted] = useState(false);
    const [error, setError] = useState("");
    const handleForgot = async () => {
        if (!email) {
            setError("Please enter your email address");
            return;
        }

        setError("");
        setLoading(true);

        try {
            await api.post("/auth/forgot-password", { email });
            setSubmitted(true);
        } catch (err) {
            setError(err.response?.data?.message || "Failed to connect to the server. Please check your network.");
        } finally {
            setLoading(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === "Enter" && !submitted) handleForgot();
    };

    return (
        <div className="auth-wrapper">
            <div className="orb-1" />
            <div className="orb-2" />
            <div className="orb-3" />

            {submitted ? (
                <div className="auth-card" style={{ textAlign: 'center' }}>
                    <div style={{ padding: '20px 0' }}>
                        <div style={{ 
                            width: 60, height: 60, borderRadius: '50%', 
                            background: 'linear-gradient(135deg, #3fb950 0%, #2ea043 100%)',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            margin: '0 auto 16px'
                        }}>
                            <CheckCircleIcon />
                        </div>
                        <h2 style={{ marginBottom: 8, color: '#fff' }}>Check your email</h2>
                        <p style={{ color: 'var(--text-muted)', marginBottom: 24 }}>
                            If an account exists with <strong>{email}</strong>, a password reset link has been sent to it.
                        </p>
                        <Link to="/login" style={{ textDecoration: 'none' }}>
                            <button className="auth-btn primary" style={{ width: '100%' }}>
                                Return to Login
                            </button>
                        </Link>
                    </div>
                </div>
            ) : (
                <div className="auth-card">
                    <div className="auth-brand">
                        <div className="brand-icon">
                            <ChartIcon />
                        </div>
                        <h1>Forgot Password</h1>
                        <p className="auth-subtitle">Enter your email to receive a reset link</p>
                    </div>

                    <div className="auth-form" onKeyDown={handleKeyDown}>
                        {error && (
                            <div className="auth-error">
                                <AlertIcon />
                                {error}
                            </div>
                        )}

                        <div className="auth-field">
                            <MailIcon />
                            <input
                                type="email"
                                placeholder="you@example.com"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                autoComplete="email"
                            />
                        </div>

                        <button
                            className="auth-btn primary"
                            onClick={handleForgot}
                            disabled={loading}
                        >
                            {loading ? <span className="btn-spinner" /> : "Send Reset Link"}
                        </button>
                    </div>

                    <div className="auth-footer">
                        Remembered your password?{" "}
                        <Link to="/login">Sign in</Link>
                    </div>
                </div>
            )}
        </div>
    );
}
