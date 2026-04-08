import { useState, useRef, useEffect } from 'react';
import { usePatients } from '../../context/PatientContext';
import { sendChat } from '../../api';
import './ChatPanel.css';

export default function ChatPanel() {
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);
  const { chatMessages, addChatMessage, activePatient } = usePatients();

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  const handleSend = async () => {
    if (!input.trim() || isTyping) return;
    const userMsg = { role: 'user', text: input.trim() };
    addChatMessage(userMsg);
    setInput('');
    setIsTyping(true);

    try {
      const history = chatMessages
        .filter(m => m.role === 'user' || m.role === 'assistant')
        .slice(-6);

      const res = await sendChat(
        userMsg.text,
        activePatient?.id ?? null,
        history
      );
      addChatMessage({ role: 'assistant', text: res.response });
    } catch (err) {
      addChatMessage({
        role: 'assistant',
        text: `Sorry, I couldn't process that request. Error: ${err.message}`
      });
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const suggestions = [
    'Which patients are most at risk?',
    'Compare hospital sites',
    'What features matter most?',
  ];

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {chatMessages.map((msg, i) => (
          <div key={i} className={`chat-msg chat-msg--${msg.role}`}>
            <div className="chat-msg__avatar">
              {msg.role === 'assistant' ? (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="11" fill="#10b981" opacity="0.15" />
                  <path d="M12 16s-5-3-5-7a3.5 3.5 0 0 1 5-2.7A3.5 3.5 0 0 1 17 9c0 4-5 7-5 7z" fill="#10b981" />
                </svg>
              ) : (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="11" fill="#3b82f6" opacity="0.15" />
                  <circle cx="12" cy="10" r="3" fill="#3b82f6" />
                  <path d="M6 19c0-3.3 2.7-6 6-6s6 2.7 6 6" fill="#3b82f6" opacity="0.3" />
                </svg>
              )}
            </div>
            <div className="chat-msg__bubble">
              <div className="chat-msg__text" dangerouslySetInnerHTML={{
                __html: msg.text
                  .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                  .replace(/\n/g, '<br/>')
              }} />
            </div>
          </div>
        ))}
        {isTyping && (
          <div className="chat-msg chat-msg--assistant">
            <div className="chat-msg__avatar">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="11" fill="#10b981" opacity="0.15" />
                <path d="M12 16s-5-3-5-7a3.5 3.5 0 0 1 5-2.7A3.5 3.5 0 0 1 17 9c0 4-5 7-5 7z" fill="#10b981" />
              </svg>
            </div>
            <div className="chat-msg__bubble">
              <div className="chat-typing">
                <span /><span /><span />
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {chatMessages.length <= 1 && (
        <div className="chat-suggestions">
          {suggestions.map((s, i) => (
            <button
              key={i}
              className="chat-suggestion"
              onClick={() => { setInput(s); }}
            >
              {s}
            </button>
          ))}
        </div>
      )}

      <div className="chat-input-bar">
        <input
          className="chat-input"
          type="text"
          placeholder="Ask about patients, features, or risk factors..."
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isTyping}
        />
        <button className="chat-send" onClick={handleSend} disabled={!input.trim() || isTyping}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
            <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </div>
    </div>
  );
}
