import React, { useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import '../styles/LandingPage.css';

export default function LandingPage() {
  const observerRef = useRef(null);

  useEffect(() => {
    observerRef.current = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.classList.add('visible');
          if (observerRef.current) {
            observerRef.current.unobserve(e.target);
          }
        }
      });
    }, { threshold: 0.12 });

    const revealElements = document.querySelectorAll('.reveal');
    revealElements.forEach(el => observerRef.current?.observe(el));

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, []);

  return (
    <div className="landing-page-container">
      {/* NAV */}
      <nav className="landing-nav">
        <Link className="landing-logo" to="/">DataInsights<span>.ai</span></Link>
        <div className="nav-actions">
          <Link to="/login" className="btn-ghost">Log in</Link>
          {/* Removed Request Access button as per new security flow */}
        </div>
      </nav>

      {/* HERO */}
      <section className="hero">
        <div className="hero-grid"></div>
        <div className="hero-glow"></div>

        <div className="hero-tag">Enterprise Intelligence Platform</div>

        <h1>Your data tells a story.<br /><em>We help you read it.</em></h1>

        <p className="hero-sub">
          DataInsights.ai gives enterprise teams a secure, intelligent layer over their data — clean it, query it, and surface decisions in plain language. No code. No risk. No guesswork.
        </p>

        <div className="hero-cta">
          <Link to="/login" className="btn-hero-outline">Explore Your Data</Link>
        </div>

        <div className="hero-stats">
          <div className="hero-stat">
            <span className="hero-stat-num">99.9%</span>
            <span className="hero-stat-label">Uptime SLA</span>
          </div>
          <div className="hero-stat-divider"></div>
          <div className="hero-stat">
            <span className="hero-stat-num">10×</span>
            <span className="hero-stat-label">Faster insights</span>
          </div>
          <div className="hero-stat-divider"></div>
          <div className="hero-stat">
            <span className="hero-stat-num">0</span>
            <span className="hero-stat-label">Raw data exposed</span>
          </div>
          <div className="hero-stat-divider"></div>
          <div className="hero-stat">
            <span className="hero-stat-num">SOC2</span>
            <span className="hero-stat-label">Ready architecture</span>
          </div>
        </div>
      </section>

      {/* VALUE STRIP */}
      <div className="value-strip">
        <div className="landing-container">
          <div className="value-item">
            <div className="value-item-icon">&#9632;</div>
            <div className="value-item-title">Role-based access</div>
            <div className="value-item-text">Admins control exactly who sees what. Employees access only their permitted datasets.</div>
          </div>
          <div className="value-item">
            <div className="value-item-icon">&#9632;</div>
            <div className="value-item-title">LLM never sees raw data</div>
            <div className="value-item-text">AI operates on statistical summaries, never your raw rows. Privacy by design.</div>
          </div>
          <div className="value-item">
            <div className="value-item-icon">&#9632;</div>
            <div className="value-item-title">Multi-tenant isolation</div>
            <div className="value-item-text">Each company is a fully isolated environment. Zero cross-tenant data bleed.</div>
          </div>
          <div className="value-item">
            <div className="value-item-icon">&#9632;</div>
            <div className="value-item-title">Audit trail built-in</div>
            <div className="value-item-text">Every query, login, and action is logged. Compliance-ready from day one.</div>
          </div>
        </div>
      </div>

      {/* HOW IT WORKS */}
      <section className="landing-section">
        <div className="landing-container">
          <div className="reveal">
            <p className="section-label">How it works</p>
            <h2 className="section-title">From raw data to<br />boardroom-ready answers</h2>
            <p className="section-sub">A five-step pipeline that turns messy datasets into confident decisions — with full governance at every stage.</p>
          </div>
          <div className="how-grid reveal">
            <div className="how-card">
              <div className="how-card-accent"></div>
              <span className="how-num">01</span>
              <h3>Upload your dataset</h3>
              <p>CSV, Excel, or JSON. Employees upload directly. Admins assign access permissions per dataset.</p>
            </div>
            <div className="how-card">
              <div className="how-card-accent"></div>
              <span className="how-num">02</span>
              <h3>Guided data cleaning</h3>
              <p>A five-step wizard handles nulls, duplicates, type mismatches, and outliers. AI suggests derived features.</p>
            </div>
            <div className="how-card">
              <div className="how-card-accent"></div>
              <span className="how-num">03</span>
              <h3>Intelligent report generation</h3>
              <p>The platform builds a statistical summary — schema, distributions, correlations — progressively in the background.</p>
            </div>
            <div className="how-card">
              <div className="how-card-accent"></div>
              <span className="how-num">04</span>
              <h3>Ask in plain English</h3>
              <p>The RAG-powered chatbot answers questions from the report first. Complex queries auto-generate validated SQL.</p>
            </div>
            <div className="how-card">
              <div className="how-card-accent"></div>
              <span className="how-num">05</span>
              <h3>Every action is audited</h3>
              <p>Admins see a complete activity log — who queried what, when, and what was returned. Full visibility.</p>
            </div>
          </div>
        </div>
      </section>

      {/* FEATURES */}
      <section className="features-section landing-section">
        <div className="landing-container">
          <div className="features-header reveal">
            <div>
              <p className="section-label">Platform capabilities</p>
              <h2 className="section-title">Built for the way<br />enterprises actually work</h2>
            </div>
            <p className="section-sub" style={{ maxWidth: '340px' }}>Every feature is designed around the reality that enterprise data is messy, sensitive, and political.</p>
          </div>
          <div className="features-grid reveal">

            <div className="feature-card">
              <div className="feature-icon">
                <svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
              </div>
              <h3>Multi-tenant data isolation</h3>
              <p>Each company gets a fully isolated data environment. No shared tables, no shared credentials, no cross-company visibility of any kind.</p>
              <span className="feature-tag">Architecture</span>
            </div>

            <div className="feature-card">
              <div className="feature-icon">
                <svg viewBox="0 0 24 24"><circle cx="12" cy="8" r="4"/><path d="M6 20v-2a6 6 0 0 1 12 0v2"/></svg>
              </div>
              <h3>Granular permission engine</h3>
              <p>Admins assign view, query, edit, and delete rights per user per dataset. Blocked queries are logged and surfaced for admin review.</p>
              <span className="feature-tag">Access control</span>
            </div>

            <div className="feature-card">
              <div className="feature-icon">
                <svg viewBox="0 0 24 24"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
              </div>
              <h3>Privacy-first AI chatbot</h3>
              <p>The LLM never receives raw data rows. It operates entirely on statistical summaries. Ask complex business questions, get safe answers.</p>
              <span className="feature-tag">AI / RAG</span>
            </div>

            <div className="feature-card">
              <div className="feature-icon">
                <svg viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
              </div>
              <h3>Text-to-SQL query engine</h3>
              <p>When the chatbot can't answer from the summary, it generates a validated SQL query — checked against permissions before a single row is returned.</p>
              <span className="feature-tag">Query engine</span>
            </div>

            <div className="feature-card">
              <div className="feature-icon">
                <svg viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/></svg>
              </div>
              <h3>AI-guided data cleaning</h3>
              <p>A five-step wizard with Ollama-powered feature engineering suggestions. Approve or reject each suggestion before any transformation runs.</p>
              <span className="feature-tag">Data ops</span>
            </div>

            <div className="feature-card">
              <div className="feature-icon">
                <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
              </div>
              <h3>Complete activity audit log</h3>
              <p>Every login, dataset access, query execution, and cleaning operation is logged with user, timestamp, and outcome. Exportable for compliance.</p>
              <span className="feature-tag">Compliance</span>
            </div>

          </div>
        </div>
      </section>

      {/* TRUST / SECURITY */}
      <section className="landing-section">
        <div className="landing-container">
          <div className="reveal">
            <p className="section-label">Security & governance</p>
            <h2 className="section-title">Your CISO will<br />actually approve this</h2>
          </div>
          <div className="trust-grid reveal">
            <ul className="trust-list">
              <li>
                <div className="trust-check"><svg viewBox="0 0 12 12"><polyline points="2,6 5,9 10,3"/></svg></div>
                <span>All AI inference runs locally via Ollama — no data leaves your infrastructure to third-party LLM providers</span>
              </li>
              <li>
                <div className="trust-check"><svg viewBox="0 0 12 12"><polyline points="2,6 5,9 10,3"/></svg></div>
                <span>Passwords bcrypt-hashed with salt rounds. JWT-based session management with configurable expiry</span>
              </li>
              <li>
                <div className="trust-check"><svg viewBox="0 0 12 12"><polyline points="2,6 5,9 10,3"/></svg></div>
                <span>Parameterized queries throughout — no raw SQL string interpolation, zero injection surface</span>
              </li>
              <li>
                <div className="trust-check"><svg viewBox="0 0 12 12"><polyline points="2,6 5,9 10,3"/></svg></div>
                <span>Dataset permissions validated server-side on every query — client-side state is never trusted</span>
              </li>
              <li>
                <div className="trust-check"><svg viewBox="0 0 12 12"><polyline points="2,6 5,9 10,3"/></svg></div>
                <span>Role-based access enforced at middleware level. Admin and employee tiers with distinct capabilities</span>
              </li>
              <li>
                <div className="trust-check"><svg viewBox="0 0 12 12"><polyline points="2,6 5,9 10,3"/></svg></div>
                <span>Full query audit trail with blocked-query logging and admin review queue</span>
              </li>
            </ul>

            <div className="trust-visual">
              <p className="trust-visual-label">Platform security posture</p>
              <div className="trust-bar-row">
                <div className="trust-bar-meta"><span>Data isolation</span><span style={{ color: 'var(--amber)' }}>100%</span></div>
                <div className="trust-bar-track"><div className="trust-bar-fill" style={{ width: '100%' }}></div></div>
              </div>
              <div className="trust-bar-row">
                <div className="trust-bar-meta"><span>Access control coverage</span><span style={{ color: 'var(--amber)' }}>100%</span></div>
                <div className="trust-bar-track"><div className="trust-bar-fill" style={{ width: '100%', animationDelay: '0.2s' }}></div></div>
              </div>
              <div className="trust-bar-row">
                <div className="trust-bar-meta"><span>Audit log completeness</span><span style={{ color: 'var(--amber)' }}>100%</span></div>
                <div className="trust-bar-track"><div className="trust-bar-fill" style={{ width: '100%', animationDelay: '0.4s' }}></div></div>
              </div>
              <div className="trust-bar-row">
                <div className="trust-bar-meta"><span>LLM data exposure</span><span style={{ color: 'var(--amber)' }}>0%</span></div>
                <div className="trust-bar-track"><div className="trust-bar-fill" style={{ width: '2%', animationDelay: '0.6s' }}></div></div>
              </div>
              <div className="trust-bar-row">
                <div className="trust-bar-meta"><span>Local inference rate</span><span style={{ color: 'var(--amber)' }}>100%</span></div>
                <div className="trust-bar-track"><div className="trust-bar-fill" style={{ width: '100%', animationDelay: '0.8s' }}></div></div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* TESTIMONIALS */}
      <section className="testimonials-section landing-section">
        <div className="landing-container">
          <div className="reveal">
            <p className="section-label">From the field</p>
            <h2 className="section-title">What decision-makers say</h2>
          </div>
          <div className="testimonials-grid reveal">
            <div className="testimonial-card">
              <div className="testimonial-quote">"</div>
              <p className="testimonial-text">We went from waiting three days for a data analyst to answer a question to getting an answer in thirty seconds — without compromising our data governance policies.</p>
              <div className="testimonial-author">
                <div className="testimonial-avatar">RK</div>
                <div>
                  <p className="testimonial-name">Rajiv Kumar</p>
                  <p className="testimonial-role">Chief Data Officer, FinanceCo</p>
                </div>
              </div>
            </div>
            <div className="testimonial-card">
              <div className="testimonial-quote">"</div>
              <p className="testimonial-text">The fact that the LLM never sees our raw rows was the dealbreaker for our security team. Every other platform we evaluated failed that test. This one passed it by design.</p>
              <div className="testimonial-author">
                <div className="testimonial-avatar">SL</div>
                <div>
                  <p className="testimonial-name">Sarah Lin</p>
                  <p className="testimonial-role">VP Engineering, RetailGroup</p>
                </div>
              </div>
            </div>
            <div className="testimonial-card">
              <div className="testimonial-quote">"</div>
              <p className="testimonial-text">Our compliance officer finally stopped worrying about our analytics stack. Audit logs, permission enforcement, local inference — it checks every box we have.</p>
              <div className="testimonial-author">
                <div className="testimonial-avatar">AM</div>
                <div>
                  <p className="testimonial-name">Alicia Montoya</p>
                  <p className="testimonial-role">CTO, HealthcareOps Inc.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="cta-section">
        <h2>Ready to give your team<br /><em style={{ fontStyle: 'italic', color: 'var(--amber)' }}>real</em> answers?</h2>
        <p>If you have an invitation, sign in to your dashboard to begin querying your data.</p>
        <div className="cta-buttons">
          <Link to="/login" className="btn-hero-outline">Sign in</Link>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="landing-footer">
        <Link className="footer-logo" to="/">DataInsights<span>.ai</span></Link>
        <div className="footer-links">
          <a href="#">Privacy</a>
          <a href="#">Security</a>
          <a href="#">Terms</a>
          <a href="#">Contact</a>
        </div>
        <p className="footer-copy">&copy; 2025 DataInsights.ai. All rights reserved.</p>
      </footer>
    </div>
  );
}
