import { useState } from 'react';
import './Contact.css';

export default function Contact() {
  const [form, setForm] = useState({ name: '', email: '', subject: '', message: '' });
  const [submitted, setSubmitted] = useState(false);

  const handleChange = (e) => {
    setForm(prev => ({ ...prev, [e.target.name]: e.target.value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setSubmitted(true);
  };

  return (
    <div className="contact">
      <section className="contact-hero">
        <div className="container">
          <div className="section__label">Get in Touch</div>
          <h1 className="contact-hero__title">Contact Us</h1>
          <p className="contact-hero__subtitle">
            Have questions about HeartLens, our methodology, or potential collaborations?
            We'd love to hear from you.
          </p>
        </div>
      </section>

      <section className="section">
        <div className="container">
          <div className="contact__grid">
            {/* Contact Info */}
            <div className="contact-info">
              <div className="contact-info__card">
                <div className="contact-info__icon">
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--hl-primary)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z" />
                  </svg>
                </div>
                <div>
                  <h3 className="contact-info__title">University of Arizona</h3>
                  <p className="contact-info__text">Department of Computer Science</p>
                  <p className="contact-info__text">Tucson, AZ 85721</p>
                </div>
              </div>

              <div className="contact-info__card">
                <div className="contact-info__icon">
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--hl-primary)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
                    <polyline points="22,6 12,13 2,6" />
                  </svg>
                </div>
                <div>
                  <h3 className="contact-info__title">Email</h3>
                  <p className="contact-info__text">asanyal@arizona.edu</p>
                  <p className="contact-info__text">cmhatre@arizona.edu</p>
                </div>
              </div>

              <div className="contact-info__card">
                <div className="contact-info__icon">
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--hl-primary)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
                    <line x1="8" y1="21" x2="16" y2="21" />
                    <line x1="12" y1="17" x2="12" y2="21" />
                  </svg>
                </div>
                <div>
                  <h3 className="contact-info__title">Course</h3>
                  <p className="contact-info__text">CSC 696D — Visual Analytics</p>
                  <p className="contact-info__text">Spring 2026</p>
                </div>
              </div>
            </div>

            {/* Form */}
            <div className="contact-form-wrap">
              {submitted ? (
                <div className="contact-success">
                  <div className="contact-success__icon">
                    <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                      <circle cx="24" cy="24" r="22" fill="#ecfdf5" stroke="#22c55e" strokeWidth="2" />
                      <path d="M15 24l6 6 12-12" stroke="#22c55e" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </div>
                  <h3>Message Sent</h3>
                  <p>Thank you for reaching out! We'll get back to you soon.</p>
                </div>
              ) : (
                <form className="contact-form" onSubmit={handleSubmit}>
                  <div className="contact-form__row">
                    <div className="contact-form__field">
                      <label>Name</label>
                      <input type="text" name="name" value={form.name} onChange={handleChange} placeholder="Your name" required />
                    </div>
                    <div className="contact-form__field">
                      <label>Email</label>
                      <input type="email" name="email" value={form.email} onChange={handleChange} placeholder="your@email.com" required />
                    </div>
                  </div>
                  <div className="contact-form__field">
                    <label>Subject</label>
                    <input type="text" name="subject" value={form.subject} onChange={handleChange} placeholder="What's this about?" required />
                  </div>
                  <div className="contact-form__field">
                    <label>Message</label>
                    <textarea name="message" value={form.message} onChange={handleChange} placeholder="Tell us more..." rows={5} required />
                  </div>
                  <button type="submit" className="btn btn--primary btn--lg contact-form__submit">
                    Send Message
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" />
                    </svg>
                  </button>
                </form>
              )}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
